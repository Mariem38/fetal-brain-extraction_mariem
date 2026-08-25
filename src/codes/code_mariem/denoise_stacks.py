"""Rician denoising of test stacks


"""
import argparse
import csv
import os
import time

import ants


def denoised_path(image_path, out_root, src_root):
    """Mirror the source tree under out_root so subject/session structure survives."""
    rel = os.path.relpath(image_path, src_root)
    return os.path.join(out_root, rel)


def denoise_csv(in_csv, out_root, out_csv, src_root, noise_model="Rician", overwrite=False):
    with open(in_csv) as fh:
        rows = list(csv.reader(fh))[1:]

    out_rows = []
    start = time.time()
    for i, (image, label) in enumerate(rows, 1):
        target = denoised_path(image, out_root, src_root)
        os.makedirs(os.path.dirname(target), exist_ok=True)

        if overwrite or not os.path.exists(target):
            img = ants.image_read(image)
            ants.image_write(ants.denoise_image(img, noise_model=noise_model), target)

        out_rows.append((target, label))
        if i % 10 == 0 or i == len(rows):
            per = (time.time() - start) / i
            print(f"  {i}/{len(rows)}  {per:.1f}s/vol  eta {per * (len(rows) - i) / 60:.1f} min",
                  flush=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "label"])
        writer.writerows(out_rows)

    return out_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv",
                        type=str,
                        default="/home/mial/Documents/for_mariem/lowfield_test.csv")

    parser.add_argument("--src_root",
                        type=str,
                        default="/home/mial/Documents/for_mariem")

    parser.add_argument("--out_root",
                        type=str,
                        default="/home/mial/Documents/for_mariem/for_mariem_denoised",
                        help="where the denoised copies go")

    parser.add_argument("--out_csv",
                        type=str,
                        default="/home/mial/Documents/for_mariem/for_mariem_denoised/lowfield_denoised_test.csv")

    parser.add_argument("--noise_model",
                        type=str,
                        default="Rician",
                        help="ants denoise_image noise model")

    parser.add_argument("--overwrite",
                        action="store_true",
                        help="redo volumes that already exist in out_root")

    args = parser.parse_args()

    rows = denoise_csv(args.csv, args.out_root, args.out_csv, args.src_root,
                       args.noise_model, args.overwrite)
    print(f"\nwrote {args.out_csv}: {len(rows)} volumes")
