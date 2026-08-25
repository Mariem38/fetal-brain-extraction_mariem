import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict

import torch
import wandb
import yaml
from monai.data import decollate_batch
from monai.inferers import SliceInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete, Compose
from torch.optim import SGD
from tqdm import tqdm

from data_generator_monai import FetalTrainData
from losses.hole_reg import HoleStats, hole_lambda_at, hole_penalty_with_selector
from model_zoo import get_network


def _overlay_image(image, pred, label):
    """Log the middle slice of a validation volume with pred/GT masks overlaid."""
    mid = image.shape[-1] // 2
    img = image[..., mid].float().numpy()
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo + 1e-8)

    return wandb.Image(img,
                       masks={"prediction": {"mask_data": pred[..., mid].numpy().astype("uint8"),
                                             "class_labels": {0: "background", 1: "brain"}},
                              "ground_truth": {"mask_data": label[..., mid].numpy().astype("uint8"),
                                               "class_labels": {0: "background", 1: "brain"}}})


def _foreground_logit(outputs):
    """Single-channel logit whose sigmoid equals the softmax foreground prob.

    The network emits 2 channels and the main loss uses softmax, while the hole
    term is written against one logit; softmax(z)[1] == sigmoid(z1 - z0), so this
    is exact and the gradient still reaches both channels.
    """
    return outputs[:, 1:2] - outputs[:, 0:1]


def _load_val_groups(configs):
    """Per-sample grouping keys for the validation set, from the manifest.

    Returns (keys, label_cols, extra) where keys[i] is the group of validation
    sample i in sorted-filename order. The val loader is batch_size=1, unshuffled
    and globs img_*.nii.gz sorted, which matches manifest row order; the caller
    checks the lengths agree before trusting it.
    """
    path = configs.get("val_manifest")
    if not path or not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = [c for c in ("field", "echo") if c and c in rows[0]]
    if not cols:
        return None, None, None
    keys = [tuple(r[c] for c in cols) for r in rows]
    extra = [r.get("te_ms", "") for r in rows]
    return keys, cols, extra


def _baseline_dice(configs):
    """Val dice of the checkpoint we are resuming from, for the stopping rule."""
    path, ep = configs.get("baseline_val_csv"), configs.get("baseline_epoch")
    if not path or not ep or not os.path.exists(path):
        return None
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["epoch"]) == int(ep):
                return float(r["mean_dice"])
    return None


def train(args):
    with open(args.cfg, 'r') as file:
        configs = yaml.safe_load(file)

    if not os.path.exists(configs["save_path"]):
        os.makedirs(configs["save_path"])

    logging.basicConfig(filename=configs["save_path"] + "/log_train.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(configs))

    use_wandb = configs.get("use_wandb", False)
    if use_wandb:
        wandb.init(project=configs.get("wandb_project", "fetal-brain-extraction"),
                   entity=configs.get("wandb_entity") or None,
                   name=configs.get("wandb_run_name") or None,
                   dir=configs["save_path"],
                   config={**configs, **{k: str(v) for k, v in vars(args).items()}})

    fetal_data = FetalTrainData(configs)
    train_dataloader, val_dataloader = fetal_data.load_data()

    model = get_network(configs)

    if configs.get("pretrained_path"):
        sd = torch.load(configs["pretrained_path"], map_location="cpu")
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        logging.info(f"loaded pretrained weights from {configs['pretrained_path']}")
    device = args.device
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    if use_wandb:
        wandb.watch(model, log="gradients", log_freq=500)

    if configs["optimizer"] == "SGD":
        optimizer = SGD(model.parameters(),
                        lr=configs["learning_rate"] * 1000,
                        momentum=0.9,
                        weight_decay=0.00004,
                        )
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs["learning_rate"],
                                     # weight_decay=0.00004,
                                     )

    loss_function = DiceCELoss(include_background=configs["include_background"],
                               to_onehot_y=True,
                               softmax=True,
                               squared_pred=True,
                               batch=True,
                               smooth_nr=0.00001,
                               smooth_dr=0.00001,
                               lambda_dice=0.6,
                               lambda_ce=0.4,
                               )

    dice_metric = DiceMetric(include_background=configs["include_background"],
                             reduction="mean",
                             get_not_nans=False,
                             ignore_empty=False)

    img_size = configs["img_size"]
    max_epochs = args.max_epochs or configs["max_epochs"]
    val_interval = configs["val_interval"]
    save_every = args.save_every or configs.get("save_every", 10)
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=configs["classes_num"])])
    post_label = Compose([AsDiscrete(to_onehot=configs["classes_num"])])

    val_group_keys, val_group_cols, val_group_extra = _load_val_groups(configs)
    if val_group_keys:
        logging.info(f"val grouping on {val_group_cols} from {configs['val_manifest']}: "
                     f"{len(set(val_group_keys))} groups over {len(val_group_keys)} samples")
    else:
        logging.warning("no val manifest / no field|echo columns -- validation dice will "
                        "only be available POOLED, which is not what we want")

    baseline = _baseline_dice(configs)
    if baseline is not None:
        logging.info(f"BASELINE (epoch {configs['baseline_epoch']} of "
                     f"{os.path.dirname(configs['baseline_val_csv'])}) pooled val dice = "
                     f"{baseline:.6f}  |  stop-if-below = {baseline - 0.01:.6f}")

    metrics_csv = os.path.join(configs["save_path"], "val_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w") as f:
            cols = "epoch,train_loss,dice_ce,hole_term,hole_ratio,lam,val_loss,mean_dice"
            if val_group_keys:
                cols += ",dice_by_group"
            f.write(cols + "\n")

    hole_kwargs = dict(thr=args.hole_thr,
                       max_hole_abs=args.hole_max_abs,
                       max_hole_rel=args.hole_max_rel,
                       close_iters=args.hole_close_iters,
                       min_depth_vox=args.hole_min_depth)
    hole_stats = HoleStats()
    hole_time_sum, hole_time_n = 0.0, 0
    step_time_sum = 0.0
    hole_timing_reported = False
    logging.info(f"hole regularizer: lambda={args.hole_lambda} "
                 f"warmup=[{args.hole_warmup_start}, {args.hole_warmup_end}] "
                 f"every_n_steps={args.hole_every_n_steps} {hole_kwargs} "
                 f"({'OFF, instrumentation only' if args.hole_lambda == 0.0 else 'ON'})")

    # Start training
    logging.info("-" * 30 + "training starts" + "-" * 30)

    step_start = time.time()
    global_step = 0
    for epoch in range(max_epochs):
        print("-" * 20)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_start = time.time()
        epoch_loss = 0
        epoch_base_loss = 0.0
        epoch_hole_term = 0.0
        step = 0
        hole_stats.reset()
        hole_lam = hole_lambda_at(epoch + 1, args.hole_lambda,
                                  args.hole_warmup_start, args.hole_warmup_end)
        for i, batch_data in enumerate(tqdm(train_dataloader,
                                            desc=f"epoch {epoch + 1}/{max_epochs}",
                                            leave=False, mininterval=10.0)):
            step += 1
            step_t0 = time.time()
            inputs, labels = (batch_data["image"].to(device),
                              batch_data["label"].to(device),
                              )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            base_loss_val = loss.item()
            hole_term_val = 0.0

            # hole regularizer. The selector is a non-differentiable scipy flood
            # fill on detached probabilities, so it is a constant 0/1 mask here;
            # the gradient reaches the logits only through `prob`.
            if step % args.hole_every_n_steps == 0:
                hole_t0 = time.time()
                fg_logit = _foreground_logit(outputs)
                if hole_lam > 0.0:
                    hole_pen, hole_sel = hole_penalty_with_selector(fg_logit, **hole_kwargs)
                    assert not hole_sel.requires_grad and hole_sel.grad_fn is None, \
                        "hole selector must not carry a gradient"
                    loss = loss + hole_lam * hole_pen
                    hole_term_val = hole_lam * hole_pen.item()
                else:
                    # lambda == 0: instrumentation only, never touch the loss
                    with torch.no_grad():
                        hole_pen, hole_sel = hole_penalty_with_selector(fg_logit, **hole_kwargs)
                hole_stats.update(torch.sigmoid(fg_logit.detach()), labels, hole_sel,
                                  penalty_value=hole_pen.item(),
                                  compute_leak=(step % args.hole_leak_every_n_steps == 0),
                                  **hole_kwargs)
                hole_time_sum += time.time() - hole_t0
                hole_time_n += 1

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_base_loss += base_loss_val
            epoch_hole_term += hole_term_val
            global_step += 1

            step_time_sum += time.time() - step_t0
            if not hole_timing_reported and global_step >= 50:
                hole_timing_reported = True
                hole_ms = 1000.0 * hole_time_sum / max(hole_time_n, 1)
                step_ms = 1000.0 * step_time_sum / global_step
                logging.info(f"hole selector timing over the first {global_step} steps: "
                             f"{hole_ms:.1f} ms/selector call ({hole_time_n} calls), "
                             f"{step_ms:.1f} ms/step total, "
                             f"{100.0 * hole_time_sum / max(step_time_sum, 1e-9):.1f}% of step time")

            if use_wandb and global_step % 50 == 0:
                wandb.log({"train/batch_loss": loss.item(),
                           "train/lr": optimizer.param_groups[0]["lr"],
                           "epoch": epoch + 1},
                          step=global_step)

            if (i + 1) % 500 == 0:
                print(f"  epoch {epoch + 1}  batch {i + 1}/{len(train_dataloader)}  "
                      f"loss {loss.item():.4f}", flush=True)

        epoch_loss /= step
        epoch_base_loss /= step
        epoch_hole_term /= step          # over ALL steps: skipped ones contribute 0,
        epoch_loss_values.append(epoch_loss)   # so this is the term's real contribution
        hole_ratio = epoch_hole_term / epoch_base_loss if epoch_base_loss else 0.0
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}  "
                     f"dice_ce: {epoch_base_loss:.6f}  "
                     f"lam*hole: {epoch_hole_term:.6e}  "
                     f"ratio: {hole_ratio:.6e}  lam: {hole_lam:.6e}")
        if use_wandb:
            wandb.log({"train/epoch_loss": epoch_loss,
                       "train/dice_ce": epoch_base_loss,
                       "train/hole_term": epoch_hole_term,
                       "train/hole_ratio": hole_ratio,
                       "train/lam": hole_lam,
                       "epoch": epoch + 1}, step=global_step)

        hole_summary = hole_stats.summary()
        if hole_summary:
            hole_summary["hole/lambda"] = hole_lam
            logging.info("epoch %d hole stats: " % (epoch + 1) +
                         "  ".join(f"{k.split('/')[1]}={v:.4f}"
                                   for k, v in hole_summary.items()))
            for line in hole_stats.size_report():
                logging.info(line)
            if use_wandb:
                wandb.log({**hole_summary, **hole_stats.size_summary(),
                           "epoch": epoch + 1}, step=global_step)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_sample = None
                val_loss = 0
                val_step = 0
                per_sample_dice = []
                for val_idx, val_data in enumerate(val_dataloader):
                    val_inputs, val_labels = (val_data["image"].to(device),
                                              val_data["label"].to(device),
                                              )

                    infer = SliceInferer(roi_size=(img_size, img_size), sw_batch_size=1, cval=-1, spatial_dim=2,
                                         progress=False)
                    val_outputs = infer(val_inputs, model)

                    # validation loss on the raw logits, before discretisation
                    val_loss += loss_function(val_outputs, val_labels).item()
                    val_step += 1

                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]

                    # compute metric for current iteration; keep the per-sample
                    # value so dice can be broken out per group, never only pooled
                    res = dice_metric(y_pred=val_outputs, y=val_labels)
                    per_sample_dice.append(float(res.mean().item()))

                    if use_wandb and val_idx == 0:
                        val_sample = (val_inputs[0, 0].detach().cpu(),
                                      val_outputs[0].argmax(0).detach().cpu(),
                                      val_labels[0].argmax(0).detach().cpu())

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()

                # reset the status for next validation round
                dice_metric.reset()

                val_loss /= max(val_step, 1)
                metric_values.append(metric)
                logging.info(f"epoch {epoch + 1} validation loss: {val_loss:.4f}, "
                             f"POOLED mean dice: {metric:.4f}"
                             + (f"  (baseline {baseline:.4f}, delta {metric - baseline:+.4f})"
                                if baseline is not None else ""))

                group_dice = {}
                if val_group_keys and len(per_sample_dice) == len(val_group_keys):
                    acc = defaultdict(list)
                    for idx, d in enumerate(per_sample_dice):
                        acc[val_group_keys[idx]].append(d)
                    te = {}
                    for idx, k in enumerate(val_group_keys):
                        te.setdefault(k, val_group_extra[idx])
                    for k in sorted(acc):
                        name = "_".join(f"{c}{v}" for c, v in zip(val_group_cols, k))
                        group_dice[name] = sum(acc[k]) / len(acc[k])
                        logging.info(f"    val dice [{name} te={te[k]}]: "
                                     f"{group_dice[name]:.4f}  (n={len(acc[k])})")
                elif val_group_keys:
                    logging.warning(f"val manifest has {len(val_group_keys)} rows but "
                                    f"{len(per_sample_dice)} samples were validated -- "
                                    f"NOT breaking dice out per group, alignment is unsafe")

                with open(metrics_csv, "a") as f:
                    row = (f"{epoch + 1},{epoch_loss:.6f},{epoch_base_loss:.6f},"
                           f"{epoch_hole_term:.8e},{hole_ratio:.8e},{hole_lam:.6e},"
                           f"{val_loss:.6f},{metric:.6f}")
                    if val_group_keys:
                        row += "," + ";".join(f"{k}={v:.6f}" for k, v in group_dice.items())
                    f.write(row + "\n")
                if use_wandb:
                    log_dict = {"val/loss": val_loss, "val/mean_dice_pooled": metric,
                                "epoch": epoch + 1}
                    log_dict.update({f"val/dice_{k}": v for k, v in group_dice.items()})
                    if baseline is not None:
                        log_dict["val/delta_vs_baseline"] = metric - baseline
                    if val_sample is not None:
                        log_dict["val/prediction"] = _overlay_image(*val_sample)
                    wandb.log(log_dict, step=global_step)
                # track the best independently of the periodic save, otherwise an
                # improvement landing on a multiple of 10 is never recorded as best
                is_best = metric > best_metric
                if is_best:
                    best_metric = metric
                    best_metric_epoch = epoch + 1

                if (epoch + 1) % save_every == 0 or (epoch + 1) == max_epochs or is_best:
                    save_mode_path = os.path.join(configs["save_path"], configs["model_name"] +
                                                  '_checkpoint-%s.pth' % (epoch + 1))
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"saved model at current epoch: {epoch + 1}, current mean dice: {metric:.4f}"
                                 f"{' (new best)' if is_best else ''}, best mean dice: {best_metric:.4f}"
                                 f" at epoch: {best_metric_epoch}")

        epoch_secs = time.time() - epoch_start
        logging.info(f"epoch {epoch + 1} took {epoch_secs:.1f}s, "
                     f"est. {(max_epochs - epoch - 1) * epoch_secs / 3600:.1f}h remaining")
        if use_wandb:
            wandb.log({"train/epoch_seconds": epoch_secs, "epoch": epoch + 1}, step=global_step)

    train_time = time.time() - step_start
    logging.info(f"train completed in {train_time:.4f} seconds "  f"best_metric: {best_metric:.4f} "
                 f"" f"at epoch: {best_metric_epoch}")

    if use_wandb:
        wandb.summary["best_mean_dice"] = best_metric
        wandb.summary["best_metric_epoch"] = best_metric_epoch
        wandb.summary["train_time_hours"] = train_time / 3600
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default='../configs/train_configs.yml',
                        help='path to config file')

    parser.add_argument('--n_gpu',
                        type=int,
                        default=2,
                        help='total gpu')

    parser.add_argument('--deterministic',
                        type=int,
                        default=1,
                        help='whether use deterministic training')

    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        help='random seed')

    parser.add_argument('--device',
                        type=str,
                        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help='what device to use')

    parser.add_argument('--max_epochs', type=int, default=None,
                        help='override max_epochs from the config')

    parser.add_argument('--save_every', type=int, default=None,
                        help='checkpoint every N epochs (default: config save_every, else 10); '
                             'every checkpoint is kept, the epoch is in the filename')

    # hole-filling regularizer. Every default reproduces the current behaviour:
    # hole_lambda 0.0 leaves the loss untouched and only runs the diagnostics.
    parser.add_argument('--hole_lambda', type=float, default=0.0,
                        help='weight of the hole-filling penalty (0.0 = off)')
    parser.add_argument('--hole_warmup_start', type=int, default=10,
                        help='epoch (1-based) at which the lambda ramp starts')
    parser.add_argument('--hole_warmup_end', type=int, default=20,
                        help='epoch (1-based) at which lambda reaches its full value')
    parser.add_argument('--hole_thr', type=float, default=0.5,
                        help='probability threshold for the hard mask')
    parser.add_argument('--hole_max_abs', type=int, default=200,
                        help='max hole size in pixels; larger enclosed regions are kept')
    parser.add_argument('--hole_max_rel', type=float, default=0.05,
                        help='max hole size as a fraction of the predicted mask')
    parser.add_argument('--hole_close_iters', type=int, default=0,
                        help='binary_closing iterations before the fill (0 = none)')
    parser.add_argument('--hole_min_depth', type=float, default=2.0,
                        help='min depth inside the mask, only with hole_close_iters > 0')
    parser.add_argument('--hole_every_n_steps', type=int, default=1,
                        help='run the selector every N steps; skipped steps contribute 0')

    parser.add_argument('--hole_leak_every_n_steps', type=int, default=10,
                        help='evaluate the leak_rate diagnostic every N steps; it is the '
                             'expensive stat (closing + distance transform on hole-free '
                             'slices), the other hole stats stay per-call')

    args = parser.parse_args()

    train(args)
