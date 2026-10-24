"""
Regenerate the three benchmark figures from the unified re-scoring.

Reads output/{class,global}_benchmarks_{alg}.csv written by unified_benchmark.py
and produces figures visually matching the Snakemake pipeline's
(scripts/data_analysis/plotting.py), so they can be swapped in directly:

  output/figures/detection_selection_bars_unified.png   (Figure 1, 3 panels)
  output/figures/recall_precision_plot_unified.png      (Figure 2)
  output/figures/metrics_table_unified.png              (Table 1)

Style, colours and markers are carried over deliberately: DeepFRET blue circle,
Papylio orange square, SCP pink triangle. The error-bar branches of the original
are dropped -- one movie, one repeat, so there is nothing to put on them.

`--out DIR` points the reader at a different results directory -- e.g.
output_x256_512/ for the band-restricted run. Figures land in DIR/figures/.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"          # set from --out in main()
FIGS = OUT / "figures"

ALGORITHMS = [("deepfret", "DeepFRET"), ("papylio", "Papylio"), ("scp", "SCP")]
CLASS_ORDER = ["Target Interactions", "Off-Target Interactions", "Pure Noise"]

OKABE_ITO = ['#0072B2', '#D55E00', '#CC79A7', '#009E73', '#E69F00']
ALG_NAMES = [d for _, d in ALGORITHMS]
ALG_COLOR_MAP = dict(zip(ALG_NAMES, OKABE_ITO[:len(ALG_NAMES)]))
MARKERS = ['o', 's', '^', 'D', 'p'][:len(ALG_NAMES)]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 24,
    'axes.titlesize': 48,
    'axes.labelsize': 40,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 28,
    'figure.titlesize': 52,
})


def load():
    dfs_class, dfs_global = {}, {}
    for alg, display in ALGORITHMS:
        dfs_class[display] = pd.read_csv(OUT / f"class_benchmarks_{alg}.csv")
        g = pd.read_csv(OUT / f"global_benchmarks_{alg}.csv", index_col=0)
        dfs_global[display] = g
    return dfs_class, dfs_global


def plot_counts(class_name, dfs_class, ax, show_ylabel=True):
    x = np.arange(len(ALG_NAMES))
    width = 0.6
    raw, filt = [], []
    for name in ALG_NAMES:
        row = dfs_class[name].loc[dfs_class[name]['Class'] == class_name]
        raw.append(float(row['Raw_Count'].iloc[0]) if not row.empty else 0.0)
        filt.append(float(row['Filtered_Count'].iloc[0]) if not row.empty else 0.0)

    colors = [ALG_COLOR_MAP[a] for a in ALG_NAMES]
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.bar(x, raw, width, color=colors, alpha=0.25, edgecolor=None, zorder=2)
    ax.bar(x, filt, width, color=colors, alpha=0.6, edgecolor='black',
           linewidth=1, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ALG_NAMES, rotation=30, ha='right', fontsize=32)
    ax.set_ylim(bottom=0)
    if show_ylabel:
        ax.set_ylabel('Spot Count', fontweight='bold', labelpad=15)


def figure_bars(dfs_class, classes=None, filename="detection_selection_bars_unified.png"):
    """
    Stacked detection/selection bars, one panel per class.

    classes=None gives all three panels (target, off-target, pure noise); pass a
    subset for the two-panel version that omits pure noise.
    """
    classes = CLASS_ORDER if classes is None else classes
    n = len(classes)
    fig = plt.figure(figsize=(8 * n, 10))
    gs = gridspec.GridSpec(2, n, wspace=0.3, hspace=0.3, height_ratios=[1, 0.3])
    for i, c_name in enumerate(classes):
        ax = fig.add_subplot(gs[0, i])
        plot_counts(c_name, dfs_class, ax, show_ylabel=(i == 0))
        ax.set_title(c_name, weight='bold', fontsize=48, pad=90)
        ax.text(-0.25, 1.05, chr(65 + i), transform=ax.transAxes, size=40, weight='bold')
        if i == 0:
            ax.yaxis.set_label_coords(-0.3, 0.5)
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    ax_leg.legend(handles=[
        Patch(facecolor='gray', alpha=0.25, label='Detection (Total)'),
        Patch(facecolor='gray', alpha=0.6, edgecolor='black', label='Selection (Retained)'),
    ], loc='center', fontsize=32, frameon=False, ncol=2)
    path = FIGS / filename
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def figure_pr(dfs_global):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal', adjustable='box')
    X, Y = np.meshgrid(np.linspace(0.01, 1, 100), np.linspace(0.01, 1, 100))
    Z = 2 * X * Y / (X + Y)
    ax.clabel(ax.contour(X, Y, Z, levels=[0.2, 0.4, 0.6, 0.7, 0.8, 0.9],
                         colors='lightgray', linestyles='--', alpha=0.6),
              inline=True, fmt='F1=%.1f', fontsize=22)

    for i, name in enumerate(ALG_NAMES):
        g = dfs_global[name]
        c, m = ALG_COLOR_MAP[name], MARKERS[i]
        p_r = float(g.loc['Global_Precision', 'Raw'])
        p_f = float(g.loc['Global_Precision', 'Filtered'])
        r_r = float(g.loc['Global_Recall', 'Raw'])
        r_f = float(g.loc['Global_Recall', 'Filtered'])
        ax.plot([r_r, r_f], [p_r, p_f], color=c, ls='--', lw=4, alpha=0.6, zorder=1)
        ax.scatter(r_r, p_r, edgecolors=c, marker=m, facecolors='none', s=350, lw=4, zorder=3)
        ax.scatter(r_f, p_f, color=c, marker=m, s=350, zorder=3)

    ax.set_xlabel('Recall', fontweight='bold', labelpad=15)
    ax.set_ylabel('Precision', fontweight='bold', labelpad=15)
    ax.set_title('Global Performance', fontweight='bold', pad=25)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, ls='--', alpha=0.5)

    alg_handles = [Line2D([0], [0], marker=MARKERS[i], color='w',
                          mfc=ALG_COLOR_MAP[n], label=n, ms=18)
                   for i, n in enumerate(ALG_NAMES)]
    state_handles = [
        Line2D([0], [0], marker='o', color='w', markeredgecolor='k',
               markerfacecolor='none', label='Detection', ms=18, markeredgewidth=4),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
               label='Selection', ms=18),
    ]
    leg1 = ax.legend(handles=alg_handles, loc='upper left', title="Algorithm",
                     fontsize=24, bbox_to_anchor=(0.02, 0.98), title_fontsize=28,
                     framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=state_handles, loc='upper left', title="State", fontsize=24,
              bbox_to_anchor=(0.02, 0.68), title_fontsize=28, framealpha=0.9)
    path = FIGS / "recall_precision_plot_unified.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def figure_table(dfs_class, dfs_global):
    data_left, data_right, row_counts = [], [], []
    for name in ALG_NAMES:
        dfc = dfs_class[name].set_index('Class').loc[CLASS_ORDER].reset_index()
        row_counts.append(len(dfc))
        for _, row in dfc.iterrows():
            data_left.append(["", row['Class'], f"{row['Raw_Count']:.0f}",
                              f"{row['Filtered_Count']:.0f}",
                              f"{row['Retention'] * 100:.1f}%"])
        g = dfs_global[name]
        for stage, col in (("Detection", "Raw"), ("Selection", "Filtered")):
            data_right.append([stage,
                               f"{float(g.loc['Global_Precision', col]):.3f}",
                               f"{float(g.loc['Global_Recall', col]):.3f}",
                               f"{float(g.loc['Global_F1_Score', col]):.3f}"])

    total_rows = sum(row_counts)
    base_h = 1.0 / (total_rows + 1)
    fig = plt.figure(figsize=(26, (total_rows + 1) * 0.6 + 2.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8], wspace=0.02)
    ax_l, ax_r = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_l.axis('off')
    ax_r.axis('off')

    cols_l = ['Algorithm', 'Class', 'Raw Count', 'Sel. Count', 'Retention']
    tbl_l = ax_l.table(cellText=data_left, colLabels=cols_l, loc='upper center',
                       cellLoc='center', colWidths=[0.15, 0.25, 0.2, 0.2, 0.2])
    cols_r = ['State', 'Precision', 'Recall', 'F1 Score']
    tbl_r = ax_r.table(cellText=data_right, colLabels=cols_r, loc='upper center',
                       cellLoc='center', colWidths=[0.2, 0.26, 0.26, 0.26])

    for tbl in (tbl_l, tbl_r):
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(16)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0)
            if r == 0:
                cell.set_text_props(weight='bold', fontsize=18)
                cell.set_height(base_h)

    for r in range(1, len(data_left) + 1):
        for c in range(len(cols_l)):
            tbl_l[r, c].set_height(base_h)
    idx = 1
    for n_rows in row_counts:
        rh = (n_rows * base_h) / 2
        for c in range(len(cols_r)):
            tbl_r[idx, c].set_height(rh)
            tbl_r[idx + 1, c].set_height(rh)
        idx += 2

    accum = 0
    for i, (n_rows, name) in enumerate(zip(row_counts, ALG_NAMES)):
        y = 1.0 - base_h - (accum * base_h) - (n_rows * base_h) / 2
        ax_l.text(0.07, y, name, ha='center', va='center', weight='bold',
                  transform=ax_l.transAxes, fontsize=18)
        accum += n_rows
        if i < len(ALG_NAMES) - 1:
            y_line = 1.0 - base_h - (accum * base_h)
            ax_l.axhline(y_line, c='gray', lw=0.5)
            ax_r.axhline(y_line, c='gray', lw=0.5)

    ax_l.set_title("Class Specific", weight='bold', pad=20, fontsize=24)
    ax_r.set_title("Global", weight='bold', pad=20, fontsize=24)
    plt.suptitle("Table 1. Comprehensive Benchmarking Statistics.",
                 x=0.1, y=0.98, ha='left', weight='bold', fontsize=28)
    path = FIGS / "metrics_table_unified.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    return path


def main():
    global OUT, FIGS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=HERE / "output",
                    help="directory holding the benchmark CSVs "
                         "(default: output/; band runs live in output_x{MIN}_{MAX}/)")
    args = ap.parse_args()
    OUT = args.out
    FIGS = OUT / "figures"
    FIGS.mkdir(parents=True, exist_ok=True)

    dfs_class, dfs_global = load()
    outputs = (
        figure_bars(dfs_class),
        figure_bars(dfs_class, classes=CLASS_ORDER[:2],
                    filename="detection_selection_bars_unified_no_noise.png"),
        figure_pr(dfs_global),
        figure_table(dfs_class, dfs_global),
    )
    for p in outputs:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
