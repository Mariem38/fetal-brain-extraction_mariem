import argparse
import logging
import os
import sys
import time

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
    max_epochs = configs["max_epochs"]
    val_interval = configs["val_interval"]
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=configs["classes_num"])])
    post_label = Compose([AsDiscrete(to_onehot=configs["classes_num"])])

    metrics_csv = os.path.join(configs["save_path"], "val_metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w") as f:
            f.write("epoch,train_loss,val_loss,mean_dice\n")

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
        step = 0
        for i, batch_data in enumerate(tqdm(train_dataloader,
                                            desc=f"epoch {epoch + 1}/{max_epochs}",
                                            leave=False, mininterval=10.0)):
            step += 1
            inputs, labels = (batch_data["image"].to(device),
                              batch_data["label"].to(device),
                              )

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            global_step += 1

            if use_wandb and global_step % 50 == 0:
                wandb.log({"train/batch_loss": loss.item(),
                           "train/lr": optimizer.param_groups[0]["lr"],
                           "epoch": epoch + 1},
                          step=global_step)

            if (i + 1) % 500 == 0:
                print(f"  epoch {epoch + 1}  batch {i + 1}/{len(train_dataloader)}  "
                      f"loss {loss.item():.4f}", flush=True)

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logging.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        if use_wandb:
            wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch + 1}, step=global_step)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_sample = None
                val_loss = 0
                val_step = 0
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

                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

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
                logging.info(f"epoch {epoch + 1} validation loss: {val_loss:.4f}, mean dice: {metric:.4f}")
                with open(metrics_csv, "a") as f:
                    f.write(f"{epoch + 1},{epoch_loss:.6f},{val_loss:.6f},{metric:.6f}\n")
                if use_wandb:
                    log_dict = {"val/loss": val_loss, "val/mean_dice": metric, "epoch": epoch + 1}
                    if val_sample is not None:
                        log_dict["val/prediction"] = _overlay_image(*val_sample)
                    wandb.log(log_dict, step=global_step)
                # track the best independently of the periodic save, otherwise an
                # improvement landing on a multiple of 10 is never recorded as best
                is_best = metric > best_metric
                if is_best:
                    best_metric = metric
                    best_metric_epoch = epoch + 1

                if (epoch + 1) % 10 == 0 or (epoch + 1) == max_epochs or is_best:
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

    args = parser.parse_args()

    train(args)
