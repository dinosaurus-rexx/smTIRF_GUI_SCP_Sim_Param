"""Figure 1 -- target and off-target panels only (no Pure Noise).

The benchmark pipeline's own master panel
(BachelorThesis-FRETAlgorithms/results/detection_selection_bars.png) plots every
class it finds, which includes "Pure Noise". Noise rejection is not the question
under study here, and its ~1600 y-axis makes the other two panels hard to read
side by side, so this regenerates the figure with the two annotated classes only.

Style is matched to the pipeline's plotting.py (same Okabe-Ito colours, bar
alphas, fonts and panel-letter placement) so this figure sits beside the
untouched Figure 2 without looking out of place.

Run with an interpreter whose matplotlib actually renders -- on this machine the
conda-forge build segfaults in Agg and the pip wheel does not. See
BachelorThesis-FRETAlgorithms/envs/plotting.yaml.

    <plotpip-env>/python.exe figures/make_figure1.py
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

RESULTS = Path(r"C:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms\results")
OUT = Path(__file__).resolve().parent / "figure1_target_offtarget.png"

# Order and display names as the pipeline emits them (config order of active algorithms).
ALGS = [("deepfret", "DeepFRET"), ("papylio", "Papylio"), ("scp", "SCP")]
CLASSES = ["Target Interactions", "Off-Target Interactions"]   # Pure Noise deliberately omitted

OKABE_ITO = ["#0072B2", "#D55E00", "#CC79A7"]
COLOR = {name: c for (_, name), c in zip(ALGS, OKABE_ITO)}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 24,
    "axes.titlesize": 48,
    "axes.labelsize": 40,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 28,
})

# --- load the per-repeat class benchmarks (repeats: 1, so these are the values) ---
data = {}
for key, name in ALGS:
    df = pd.read_csv(RESULTS / key / "class_benchmarks" / "rep1.csv").set_index("Class")
    data[name] = df

names = [n for _, n in ALGS]

# Dedicated bottom row for the legend, as the pipeline's master panel does --
# a fig.legend() anchored below the axes collides with the rotated tick labels.
fig = plt.figure(figsize=(16, 11))
# hspace generous: the 30-degree tick labels at fontsize 32 hang well below the
# axes and will sit on top of the legend row otherwise.
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.12], wspace=0.3, hspace=0.75)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

for i, (ax, cls) in enumerate(zip(axes, CLASSES)):
    x = np.arange(len(names))
    raw = [data[n].loc[cls, "Raw_Count"] for n in names]
    filt = [data[n].loc[cls, "Filtered_Count"] for n in names]
    colors = [COLOR[n] for n in names]

    ax.yaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    # pale = everything detected; solid = what survived that pipeline's filtering
    ax.bar(x, raw, 0.6, color=colors, alpha=0.25, edgecolor=None, zorder=2)
    ax.bar(x, filt, 0.6, color=colors, alpha=0.6, edgecolor="black", linewidth=1, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=32)
    ax.set_ylim(bottom=0)
    ax.set_title(cls, weight="bold", fontsize=44, pad=45)
    ax.text(-0.16, 1.12, chr(65 + i), transform=ax.transAxes, size=40, weight="bold",
            va="top", ha="right")
    if i == 0:
        ax.set_ylabel("Spot Count", fontweight="bold", labelpad=15)

handles = [
    Patch(facecolor="gray", alpha=0.25, label="Detection (Total)"),
    Patch(facecolor="gray", alpha=0.6, edgecolor="black", label="Selection (Retained)"),
]
ax_legend = fig.add_subplot(gs[1, :])
ax_legend.axis("off")
ax_legend.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=28)

OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=300, bbox_inches="tight")
plt.close()
print(f"wrote {OUT}")
for n in names:
    t, o = data[n].loc[CLASSES[0]], data[n].loc[CLASSES[1]]
    print(f"  {n:9s} target {int(t.Raw_Count):3d}->{int(t.Filtered_Count):3d}   "
          f"off-target {int(o.Raw_Count):3d}->{int(o.Filtered_Count):3d}")
