"""Summary tables of the multi-echo test runs, low field and high field.

Reads the per-volume csv each test.py run wrote under results/runs/<family>/
and reports mean +- std across volumes, raw against Rician-denoised input.
Renders booktabs-style tables (horizontal rules only, no fill) and also writes
LaTeX so the numbers can go straight into a paper.
"""
import argparse
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "runs")

# field -> rows of (model label, input label, family, run folder, csv name)
LAYOUT = {
    "Low field": [
        ("Pretrained", "Raw",      "pretrained", "pretrained_lowfield",          "lowfield_AttUNet.csv"),
        ("v2 ep48",    "Raw",      "v2",         "v2_ep48_lowfield",             "lowfield_AttUNet_checkpoint-48.csv"),
        ("v3 ep70",    "Raw",      "v3",         "v3_ep70_lowfield",             "lowfield_AttUNet_checkpoint-70.csv"),
        ("Pretrained", "Denoised", "pretrained", "lowfield_denoised_pretrained", "lowfield_AttUNet.csv"),
        ("v2 ep48",    "Denoised", "v2",         "lowfield_denoised_v2_ep48",    "lowfield_AttUNet_checkpoint-48.csv"),
        ("v3 ep70",    "Denoised", "v3",         "lowfield_denoised_v3_ep70",    "lowfield_AttUNet_checkpoint-70.csv"),
    ],
    "High field": [
        ("Pretrained", "Raw",      "pretrained", "pretrained_highfield",          "highfield_AttUNet.csv"),
        ("v2 ep48",    "Raw",      "v2",         "v2_ep48_highfield",             "highfield_AttUNet_checkpoint-48.csv"),
        ("v3 ep70",    "Raw",      "v3",         "v3_ep70_highfield",             "highfield_AttUNet_checkpoint-70.csv"),
        ("Pretrained", "Denoised", "pretrained", "highfield_denoised_pretrained", "highfield_AttUNet.csv"),
        ("v2 ep48",    "Denoised", "v2",         "highfield_denoised_v2_ep48",    "highfield_AttUNet_checkpoint-48.csv"),
        ("v3 ep70",    "Denoised", "v3",         "highfield_denoised_v3_ep70",    "highfield_AttUNet_checkpoint-70.csv"),
    ],
}

METRICS = [("Dice", "Dice"), ("IoU", "IoU"), ("frac_slices_multi", "Frag.")]


def _find_csv(family, folder, csv_name):
    """Preferred name first, then any csv in the folder (the modality prefix varies)."""
    path = os.path.join(RUNS_ROOT, family, folder, csv_name)
    if os.path.exists(path):
        return path
    folder_path = os.path.join(RUNS_ROOT, family, folder)
    if os.path.isdir(folder_path):
        found = [f for f in sorted(os.listdir(folder_path)) if f.endswith(".csv")]
        if found:
            return os.path.join(folder_path, found[0])
    return None


def collect(field):
    rows = []
    for model, prep, family, folder, csv_name in LAYOUT[field]:
        path = _find_csv(family, folder, csv_name)
        if path is None:
            print(f"missing, skipped: {family}/{folder}")
            continue
        d = pd.read_csv(path)
        if d.empty:
            print(f"empty csv, skipped: {path}")
            continue

        row = {"Model": model, "Input": prep}
        for col, label in METRICS:
            row[label] = f"{d[col].mean():.3f} ± {d[col].std():.3f}"
        row["Dice < 0.5"] = f"{int((d.Dice < 0.5).sum())}/{len(d)}"
        rows.append(row)
    return pd.DataFrame(rows)


def _draw(ax, table, title):
    """Booktabs: top rule, header rule, midrule between blocks, bottom rule."""
    ax.axis("off")
    tab = ax.table(cellText=table.values, colLabels=table.columns,
                   cellLoc="center", loc="upper center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1, 1.5)

    n_rows = len(table)
    # data row i sits at table row i+1, so the rule goes on boundary+1
    boundary = next((i for i, v in enumerate(table["Input"]) if v == "Denoised"), None)

    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0)
        cell.set_facecolor("none")
        cell.PAD = 0.04
        if c == 0:
            cell.get_text().set_ha("left")
            cell._text.set_x(0.06)
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"          # top rule + header rule
            cell.set_linewidth(0.9)
        elif r == n_rows:
            cell.visible_edges = "B"           # bottom rule
            cell.set_linewidth(0.9)
        elif boundary is not None and r == boundary + 1:
            cell.visible_edges = "T"           # midrule between raw and denoised
            cell.set_linewidth(0.5)

    ax.set_title(title, fontsize=10.5, pad=10, loc="left")


def render(tables, out_path):
    fig, axs = plt.subplots(len(tables), 1,
                            figsize=(7.6, 2.5 * len(tables)))
    axs = [axs] if len(tables) == 1 else list(axs)
    for ax, (title, table) in zip(axs, tables):
        _draw(ax, table, title)
    fig.tight_layout(h_pad=2.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"

    parser = argparse.ArgumentParser()

    parser.add_argument("--out",
                        type=str,
                        default=os.path.join(RUNS_ROOT, "..", "figures", "multi_echo_table.png"),
                        help="where to write the table image")

    parser.add_argument("--tex_out",
                        type=str,
                        default=os.path.join(RUNS_ROOT, "..", "figures", "multi_echo_table.tex"),
                        help="LaTeX version of the same tables")

    args = parser.parse_args()

    tables, tex = [], []
    for field in LAYOUT:
        table = collect(field)
        if table.empty:
            print(f"no results for {field}, skipped")
            continue
        n = {"Low field": 63, "High field": 48}[field]
        tables.append((f"{field} ({n} volumes) — mean ± std across volumes", table))
        tex.append(table.to_latex(index=False, escape=False,
                                  caption=f"{field}, {n} volumes. Mean $\\pm$ std across volumes.",
                                  label=f"tab:{field.lower().replace(' ', '_')}"))
        print(f"\n=== {field} ===")
        print(table.to_string(index=False))

    if not tables:
        raise SystemExit("no results found under results/runs/")

    render(tables, args.out)
    os.makedirs(os.path.dirname(args.tex_out), exist_ok=True)
    with open(args.tex_out, "w") as fh:
        fh.write("\n\n".join(tex))

    print(f"\nwrote {os.path.normpath(args.out)}")
    print(f"wrote {os.path.normpath(args.tex_out)}")
