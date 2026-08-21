import os
import math
import sys
import time
import logging
import pandas as pd
import yaml
import argparse
import matplotlib
import numpy as np
import scipy.ndimage as ndi

import torch

import monai.transforms as tr
from matplotlib import pyplot as plt
from monai.data import decollate_batch, TestTimeAugmentation
from monai.handlers import from_engine
from monai.inferers import SliceInferer
from monai.metrics import HausdorffDistanceMetric, MeanIoU, DiceMetric, ConfusionMatrixMetric, \
    compute_confusion_matrix_metric
from monai.utils import first, set_determinism

from data_generator_monai import FetalTestData
from model_zoo import get_network


def component_scores(pred_chw, slice_axis=2):
    """Per-slice fragmentation of one prediction. Input is channel-first (C, X, Y, Z).

    Measured on the RAW prediction, before any cleanup.


    """
    m = np.asarray(pred_chw.detach().cpu())[0] > 0

    st2 = ndi.generate_binary_structure(2, 2)   # 8-connected in-plane
    n_comp_2d = max_comp = n_multi = n_fg = 0
    for k in range(m.shape[slice_axis]):
        sl = np.take(m, k, axis=slice_axis)
        if not sl.any():
            continue
        n_fg += 1
        _, c = ndi.label(sl, structure=st2)
        n_comp_2d += c
        max_comp = max(max_comp, c)
        if c > 1:
            n_multi += 1

    return dict(n_comp_2d=n_comp_2d, max_comp_slice=max_comp,
                n_slices_multi=n_multi, n_slices_fg=n_fg,
                frac_slices_multi=n_multi / n_fg if n_fg else float("nan"))


def plot_images(images, masks, gt=None,
                volume_dice=None, mean_slice_dice=None, slice_dice=None,
                save_name=None, display=None):
    slice_num = images.shape[-1]
    # Calculate the number of columns and rows based on the number of images
    n_cols = int(math.ceil(math.sqrt(slice_num)))
    n_rows = int(math.ceil(slice_num / n_cols))

    cmap_mask = matplotlib.colors.ListedColormap(['none', 'red'])
    cmap_gt = matplotlib.colors.ListedColormap(['none', 'blue'])

    # Create a grid of subplots with the calculated number of columns and rows
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for i in range(slice_num):
        # Calculate the row and column indices for the current subplot
        row = i // n_cols
        col = i % n_cols

        # Plot the image with the mask overlay
        axs[row, col].imshow(images[:, :, i], cmap='gray')
        axs[row, col].imshow(masks[:, :, i], alpha=0.6, cmap=cmap_mask)

        if not (gt is None):
            axs[row, col].imshow(gt[:, :, i], alpha=0.3, cmap=cmap_gt)

        if not (slice_dice is None):
            axs[row, col].set_title(f"dice= {slice_dice[i]:.2f}")

        axs[row, col].axis('off')

    # Remove any unused subplots
    for i in range(len(images), n_rows * n_cols):
        axs.flatten()[i].set_visible(False)

    if volume_dice and mean_slice_dice:
        fig.suptitle(
            f"red: predicted, blue: manual, volume_dice= {volume_dice:.2f}, mean_slice_dice= {mean_slice_dice:.2f}")

    elif volume_dice:
        fig.suptitle(
            f"red: predicted, blue: manual, volume_dice= {volume_dice:.2f}")

    if save_name:
        plt.savefig(save_name)

    if display:
        plt.show()


def test(args):
    with open(args.cfg, 'r') as file:
        configs = yaml.safe_load(file)

    if not os.path.exists(configs["save_path"]):
        os.makedirs(configs["save_path"])

    logging.basicConfig(filename=configs["save_path"] + "/log_test_" + configs["modality"] + ".txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(str(configs))

    device = args.device
    model = get_network(configs)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    model.load_state_dict(torch.load(configs["saved_model_path"], map_location=device))

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"total_trainable_parameters: {pytorch_total_params}")

    model.eval()

    fetal_test_data = FetalTestData(configs)
    test_dataloader, test_files, test_org_transforms_list = fetal_test_data.load_data()

    inferer = SliceInferer(
        roi_size=(configs["img_size"], configs["img_size"]),
        spatial_dim=2,
        sw_batch_size=4,
        overlap=0.50,
        progress=False
    )

    dice_metric = DiceMetric(
        include_background=configs["include_background"],
        reduction="mean",
        ignore_empty=False
    )
    iou_metric = MeanIoU(
        include_background=configs["include_background"],
        reduction="mean",
        ignore_empty=False
    )
    cm_metric = ConfusionMatrixMetric(
        include_background=configs["include_background"],
        metric_name=["sensitivity", "specificity", "precision", "f1 score"],
        compute_sample=True,   # per-volume rates, then averaged; the block below pools voxels instead
        reduction="mean"
    )

    test_time = []
    subj_list = []
    vol_list = []
    comp_list = []
    mask_list = []
    data_type = []
    if configs["test_augmentation"]:
        # not complete yet!
        spatial_transforms = [[tr.RandFlipd(keys="image", spatial_axis=1, prob=0.8)],
                              [tr.RandRotate90d(keys="image", spatial_axes=(0, 1), prob=0.6)],
                              [tr.RandRotated(keys="image", range_x=(0.2, 1.0), prob=0.5)],
                              [tr.RandZoomd(keys="image", min_zoom=0.95, max_zoom=1.2, prob=0.6)], ]

        with torch.no_grad():
            for file in test_files:
                tta_out = []
                for trans in spatial_transforms:
                    tta_transforms = tr.Compose(test_org_transforms_list + trans)
                    tta_post_transforms = tr.Compose([tr.Activations(softmax=True),
                                                      tr.AsDiscrete(argmax=True, to_onehot=None),
                                                      ])

                    tta = TestTimeAugmentation(transform=tta_transforms,
                                               batch_size=1,
                                               num_workers=0,
                                               inferrer_fn=lambda x: inferer(x, model),
                                               device=device,
                                               orig_key="image",
                                               post_func=lambda x: tta_post_transforms(x),
                                               progress=True,
                                               return_full_data=True)

                    _out = tta(file, num_examples=5)
                    tta_out.append(_out)

                tta_output = torch.vstack(tta_out)
                tta_output_mean = torch.mean(tta_output, dim=0)

                unmodified_data = tr.LoadImaged(keys=["image", "label"])(file)
                dice = dice_metric(y_pred=tta_output_mean[None, ...], y=unmodified_data["label"][None, None, ...])
                iou = iou_metric(y_pred=tta_output_mean[None, ...], y=unmodified_data["label"][None, None, ...])
                cm = cm_metric(y_pred=tta_output_mean[None, ...], y=unmodified_data["label"][None, None, ...])

    else:
        post_transforms_list = [
            tr.Invertd(
                keys="pred",
                transform=tr.Compose(test_org_transforms_list),
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            tr.Activationsd(keys="pred", softmax=True),
            tr.AsDiscreted(keys="pred", argmax=True, to_onehot=None),
            # tr.KeepLargestConnectedComponentd(keys="pred", applied_labels=1, num_components=2, is_onehot=False),
            # tr.RemoveSmallObjectsd(keys="pred", min_size=50, connectivity=1),
        ]

        if configs["save_predictions"]:
            post_transforms_list.append(
                tr.SaveImaged(
                    keys="pred",
                    meta_keys="pred_meta_dict",
                    output_dir=configs["save_path"],
                    separate_folder=False,
                    data_root_dir=configs["data_root_dir"],
                    output_postfix="pred",
                    resample=True
                )
            )

        post_transforms = tr.Compose(post_transforms_list)

        with torch.no_grad():
            for i, test_data in enumerate(test_dataloader):
                test_inputs, test_labels = test_data["image"].to(device), test_data["label"].to(device)

                start_time = time.time()

                test_data["pred"] = inferer(test_inputs, model)
                test_data = [post_transforms(i) for i in decollate_batch(test_data)]

                test_time.append((time.time() - start_time) / test_inputs.shape[-1])

                test_outputs = from_engine(["pred"])(test_data)

                if test_outputs[0].ndim == 4:
                    comp_list.append(component_scores(test_outputs[0]))
                else:
                    logging.info(f"component scores skipped: expected (C, X, Y, Z), "
                                 f"got {tuple(test_outputs[0].shape)}")
                    comp_list.append(dict(n_comp_2d=float("nan"),
                                          max_comp_slice=float("nan"),
                                          n_slices_multi=float("nan"),
                                          n_slices_fg=float("nan"),
                                          frac_slices_multi=float("nan")))

                dice = dice_metric(y_pred=test_outputs, y=test_labels)
                iou = iou_metric(y_pred=test_outputs, y=test_labels)
                cm = cm_metric(y_pred=test_outputs, y=test_labels)

                tp, fp, tn, fn = (int(v) for v in cm[0, 0].tolist())
                logging.info(f"confusion matrix tp={tp} fp={fp} tn={tn} fn={fn}")

                # outside the modality branches: an empty vol_list would silently
                # zero out the zip below and write a rowless csv
                vol_list.append(os.path.basename(
                    test_inputs.meta["filename_or_obj"][0]).replace(".nii.gz", ""))

                if configs["modality"] == "T2W" or configs["modality"] == "otherscanners":
                    subject_id = os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0]))
                    if subject_id.endswith('s1') or subject_id.endswith('s2'):
                        subject_id = subject_id[:-2]
                    subj_list.append(subject_id)
                    data_type.append(os.path.basename(os.path.dirname(os.path.dirname(
                        test_inputs.meta["filename_or_obj"][0]))))

                elif configs["modality"] in ("lowfield", "highfield"):
                    # BIDS layout: the parent dir is "anat", so parse the filename
                    fname = os.path.basename(test_inputs.meta["filename_or_obj"][0])
                    subj_list.append(fname.split('_')[0])
                    data_type.append(configs["modality"])

                elif configs["modality"] == "DWI":
                    subject_id = os.path.basename(test_inputs.meta["filename_or_obj"][0]).split('_')[0]
                    if subject_id.endswith('s1') or subject_id.endswith('s2'):
                        subject_id = subject_id[:-2]
                    subj_list.append(subject_id)
                    data_type.append(os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0])))

                elif configs["modality"] == "fMRI":
                    subject_id = os.path.dirname(test_inputs.meta["filename_or_obj"][0]).split('/')[-2][:-2]
                    subj_list.append(subject_id)
                    data_type.append('fMRI')

                elif configs["modality"] == "fMRI-ON":
                    subject_id = os.path.dirname(test_inputs.meta["filename_or_obj"][0]).split('/')[6]
                    subj_list.append(subject_id)
                    data_type.append('fMRI-ON')

                if configs["plot_results"]:
                    if dice.item() < 0.85:
                        original_image = tr.LoadImage()(test_outputs[0].meta["filename_or_obj"])[0]
                        original_label = tr.LoadImage()(test_labels[0].meta["filename_or_obj"])[0]

                        filename_without_extension = os.path.splitext(os.path.basename
                                                                      (test_inputs.meta["filename_or_obj"][0]))[0]
                        parent_folder = os.path.basename(os.path.dirname(test_inputs.meta["filename_or_obj"][0]))

                        new_filename = f"{parent_folder}_{filename_without_extension}.png"
                        save_dir = os.path.join(configs["save_path"], 'bad_dice_figs')
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        save_name = os.path.join(save_dir, new_filename)

                        plot_images(
                            original_image,
                            test_outputs[0].detach().cpu()[0],
                            gt=original_label,
                            volume_dice=dice.item(),
                            mean_slice_dice=None,
                            slice_dice=None,
                            save_name=save_name,
                            display=False
                        )
                        print(test_inputs.meta["filename_or_obj"][0])
                        print(dice.item())
                        print(iou.item())

    if configs["save_metrics"]:
        header = ["Method", "Modality", "Type", "Subject", "Volume", "Dice", "IoU",
                  "n_comp_2d", "max_comp_slice", "n_slices_multi", "n_slices_fg",
                  "frac_slices_multi"]
        dice_list = (dice_metric.get_buffer().detach().cpu().numpy()[:, 0]).tolist()
        iou_list = (iou_metric.get_buffer().detach().cpu().numpy()[:, 0]).tolist()
        modality = [configs["modality"]] * len(dice_list)
        method = [os.path.splitext(os.path.basename(configs["saved_model_path"]))[0]] * len(dice_list)

        # a silent length mismatch here would misalign every row
        assert len(comp_list) == len(dice_list), \
            f"comp_list has {len(comp_list)} entries but dice_list has {len(dice_list)}"
        data = list(zip(method, modality, data_type, subj_list, vol_list, dice_list, iou_list,
                        [c["n_comp_2d"] for c in comp_list],
                        [c["max_comp_slice"] for c in comp_list],
                        [c["n_slices_multi"] for c in comp_list],
                        [c["n_slices_fg"] for c in comp_list],
                        [c["frac_slices_multi"] for c in comp_list]))

        file_path = os.path.join(configs["save_path"], configs["modality"] + "_" + method[0] + ".csv")
        df = pd.DataFrame(data, columns=header)
        df.to_csv(file_path, index=False)

        logging.info(f"evaluation metric dice mean: {np.mean(dice_list)}")
        logging.info(f"evaluation metric dice std: {np.std(dice_list)}")

        logging.info(f"evaluation metric iou mean: {np.mean(iou_list)}")
        logging.info(f"evaluation metric iou std: {np.std(iou_list)}")

    logging.info(f"evaluation metric dice: {dice_metric.aggregate()}")
    logging.info(f"evaluation metric iou: {iou_metric.aggregate()}")

    if comp_list:
        fracs = np.array([c["frac_slices_multi"] for c in comp_list], dtype=float)
        n_any = sum(1 for c in comp_list if c["n_slices_multi"] > 0)
        # a single fragmented slice trips the raw count, so the fraction is the meaningful threshold
        logging.info(f"stacks with a fragmented slice: {n_any} / {len(comp_list)}")
        logging.info(f"stacks with frac_slices_multi > 0.05: {int((fracs > 0.05).sum())} / {len(comp_list)}")
        logging.info(f"frac_slices_multi mean: {np.nanmean(fracs)}")
        logging.info(f"frac_slices_multi std: {np.nanstd(fracs)}")
        logging.info(f"worst slice component count: {np.nanmax([c['max_comp_slice'] for c in comp_list])}")

    # buffer is [N, C, 4] holding tp, fp, tn, fn per volume
    cm_buffer = cm_metric.get_buffer().detach().cpu().numpy()
    tp, fp, tn, fn = cm_buffer[:, 0, :].sum(axis=0)

    logging.info(f"confusion matrix summed over {cm_buffer.shape[0]} volumes (foreground = brain):")
    logging.info(f"{'':>10}{'pred bg':>18}{'pred fg':>18}")
    logging.info(f"{'true bg':>10}{int(tn):>18d}{int(fp):>18d}")
    logging.info(f"{'true fg':>10}{int(fn):>18d}{int(tp):>18d}")

    logging.info(f"voxel-weighted sensitivity: {tp / (tp + fn)}")
    logging.info(f"voxel-weighted specificity: {tn / (tn + fp)}")
    logging.info(f"voxel-weighted precision: {tp / (tp + fp)}")
    logging.info(f"voxel-weighted f1 score: {2 * tp / (2 * tp + fp + fn)}")

    for name, value in zip(["sensitivity", "specificity", "precision", "f1 score"], cm_metric.aggregate()):
        logging.info(f"per-volume mean {name}: {value.item()}")

    logging.info(f"latency mean: {np.mean(test_time[1:])}")
    logging.info(f"latency std: {np.std(test_time[1:])}")

    dice_metric.reset()
    iou_metric.reset()
    cm_metric.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg',
                        type=str,
                        default='../configs/test_configs.yml',
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

    if args.deterministic:
        set_determinism(seed=12345)
    else:
        set_determinism(seed=None)

    test(args)
