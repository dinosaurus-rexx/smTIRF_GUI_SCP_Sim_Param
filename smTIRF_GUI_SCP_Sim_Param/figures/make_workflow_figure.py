"""Workflow diagram: real movie + simulators -> SCP detector -> tuning arms -> results.

Two levels of the same pipeline, so the figure can be either a one-glance overview or a
methods-appendix reference:

    python figures/make_workflow_figure.py                  # overview (default)
    python figures/make_workflow_figure.py --level detailed # every script and file

Each level is emitted three ways from one declaration, so the topology can be pasted
into a thesis, a README, or a slide without redrawing it:

    figures/pipeline_workflow.png / .svg   matplotlib render (no external tools needed)
    figures/pipeline_workflow.dot          Graphviz source (rendered if `dot` is on PATH)
    figures/pipeline_workflow.mmd          Mermaid source (VS Code / GitHub / docs)

Every F1 quoted in the boxes comes from METHODS_SPEC.md 4.3 and 5 -- this script draws
the numbers, it does not compute them. Update them here only when that table changes.

Coordinates are hand-placed in a 0-100 x space (y up) because the lane structure *is*
the figure; Graphviz and Mermaid get the lanes as clusters and lay themselves out.
"""
import argparse
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent
STEM = "pipeline_workflow"

# Okabe-Ito hues, light fill + saturated edge, so the kinds stay distinguishable in
# greyscale print as well as on screen. Matched to the palette in make_figure1.py.
KINDS = {
    "real":   ("#E7F0F8", "#0072B2"),   # the real recording and its annotation
    "sim":    ("#FDF0E6", "#D55E00"),   # simulated data
    "detect": ("#F3EDF7", "#8E5BA6"),   # the detector
    "tune":   ("#EAF4F0", "#009E73"),   # search / scoring machinery
    "arm":    ("#EFF6EA", "#5C8A2E"),   # an evaluation design + its held-out number
    "note":   ("#F1F1F1", "#5A5A5A"),   # framing, constraints, artifacts
}
# Edge styles: (colour, linewidth, dash). "feedback" is reserved for the paths the
# caption calls out by number -- they are the non-obvious part of the workflow, so they
# are drawn on top of everything else.
EDGE_STYLES = {
    "solid":    ("#4D4D4D", 1.2, "-"),
    "dashed":   ("#8A8A8A", 1.2, "--"),
    "feedback": ("#B02418", 2.0, "-"),
}

KIND_LEGEND = {
    "real": "real data",
    "sim": "simulated data",
    "detect": "SCP detector",
    "tune": "search / scoring",
    "arm": "evaluation design",
    "note": "constraint / output",
}


class Node(NamedTuple):
    label: str
    sub: str
    foot: str       # bold line pinned to the bottom of the box (the takeaway)
    kind: str
    lane: int
    x: float
    y: float
    w: float
    h: float


class Edge(NamedTuple):
    src: str
    dst: str
    label: str
    style: str
    rad: float      # arc3 curvature; bow height ~ rad * chord / 2, so keep it small
    t: float        # *preferred* label position along the curve (see _place_label)


class Graph(NamedTuple):
    title: str
    figsize: tuple
    lanes: list      # (id, title, y_bottom, y_top)
    nodes: dict
    edges: list


def N(label, sub="", foot="", kind="real", lane=1, x=0.0, y=0.0, w=10.0, h=7.0):
    return Node(label, sub, foot, kind, lane, x, y, w, h)


def E(src, dst, label="", style="solid", rad=0.0, t=0.5):
    return Edge(src, dst, label, style, rad, t)


# ===========================================================================
# OVERVIEW -- the four stages of the methodology figure, matching the caption's
# structure and its section references. Section numbers (2.2 - 2.8) are the ones
# used in the Methods text, NOT the METHODS_SPEC.md lettering.
#
# The four protocols are the four evaluation *designs*; the random-search draw and
# the leak-free recalibration are the two *controls* (one box), which is how
# METHODS_SPEC.md 4.3 labels them -- (a)-(d) designs, (e)-(f) controls.
#
# The two feedback paths the caption calls out are drawn in red and numbered:
#   (1) the detector runs *inside* the simulator's calibration, because two
#       calibration targets are detector-mediated;
#   (2) the simulator's output feeds the parameter search, whose winner is then
#       judged on the real recording.
# ===========================================================================

OVERVIEW = Graph(
    title="Workflow of the methodology",
    figsize=(21, 14.5),
    lanes=[
        (1, "Stage 1\nReal recording  \u00a72.2, \u00a72.5", 77.0, 99.0),
        (2, "Stage 2\nSimulator  \u00a72.5, \u00a72.6", 54.0, 75.0),
        (3, "Stage 3\nDetect + score  \u00a72.3, \u00a72.4", 33.0, 52.0),
        (4, "Stage 4\nSearch + protocols  \u00a72.7, \u00a72.8", 0.0, 31.0),
    ],
    nodes={
        # ---- Stage 1: annotate, then measure ------------------------------
        "real": N("Real smTIRF recording",
                  "one dual-channel movie \u00b7 501 frames\n"
                  "512 px wide \u00b7 analysed as a single channel",
                  "the only real data available",
                  "real", 1, 8, 81, 25, 13),
        "annot": N("Manual annotation  \u00a72.2",
                   "GUI2 \u00b7 243 target and 91 off-target spots\n"
                   "stored [y, x], converted to the detector's (x, y)",
                   "partitioned into disjoint x-bands",
                   "real", 1, 36, 81, 29, 13),
        "measure": N("Measurement  \u00a72.5",
                     "spot photometry, background structure, noise\n"
                     "and spot density measured from the recording",
                     "reference file of target statistics\n(machine-readable)",
                     "real", 1, 68, 81, 31, 13),

        # ---- Stage 2: configure from the reference, then judge against it --
        "sim": N("Simulator configured from the reference  \u00a72.5",
                 "per-emitter sub-pixel PSF \u00b7 blinking and\n"
                 "bleaching \u00b7 structured background \u00b7\n"
                 "soft-core placement",
                 "fully seeded: one seed \u2192 one movie, exactly",
                 "sim", 2, 8, 58, 29, 13),
        "movies": N("Synthetic movies + ground truth",
                    "ground truth emitted in two formats \u00a72.5:\n"
                    "(x, y) label JSON  and  [y, x] coord/class .npy",
                    "one dataset per calibration variant",
                    "sim", 2, 40, 58, 28, 13),
        "harness": N("Validation harness  \u00a72.6",
                     "compares the simulated movies against the\n"
                     "same reference file, through one shared\n"
                     "estimator module for both sides",
                     "acceptance gates T1\u2013T5, T7",
                     "sim", 2, 71, 58, 28, 13),

        # ---- Stage 3: detect on either data source, then score -------------
        "detector": N("SCP detector  \u00a72.3",
                      "bandpass background subtraction  \u2192  Laplacian-of-Gaussian\n"
                      "blob detection  \u2192  KD-tree linking across frames  \u2192  event\n"
                      "classification  \u2192  duplicate suppression",
                      "run on simulated or real movies \u00b7 7 free parameters",
                      "detect", 3, 8, 36, 45, 13),
        "scorer": N("Scoring against annotations  \u00a72.4",
                    "greedy one-to-one assignment within a fixed pixel\n"
                    "margin; every detection not matched to a target\n"
                    "counts against precision",
                    "recall \u00b7 precision \u00b7 F1",
                    "detect", 3, 56, 36, 43, 13),

        # ---- Stage 4: search, four protocols, and the two controls ---------
        "search": N("Bayesian optimization  \u00a72.7",
                    "a Tree-structured Parzen Estimator samples all seven parameters "
                    "jointly, so coupled optima stay reachable; seeded, with a "
                    "per-run timeout",
                    "objective = F1 of one detector run",
                    "tune", 4, 8, 22.5, 44, 8),
        "controls": N("Two controls bound the claims",
                      "random configurations drawn without the simulator, and the\n"
                      "simulator recalibrated with the test band withheld entirely",
                      "separate the simulator's contribution from selection",
                      "note", 4, 55, 22.5, 44, 8),

        "p1": N("(1)  tune on the real\n       recording",
                "search   real train band\nselect    real validation band\n"
                "report   real test band\n\nspends real labels and\n100+ real detector runs",
                "highest held-out F1",
                "arm", 4, 8, 1.5, 21, 18),
        "p2": N("(2)  tune on simulation\n       only",
                "search   simulated train movies\nselect    simulated validation\n"
                "report   real recording, once\n\nno real annotation enters\nthe loop at any point",
                "the classic design",
                "arm", 4, 31.3, 1.5, 21, 18),
        "p3": N("(3)  search on simulation,\n       select on real",
                "search   simulated movies\nselect    real validation band\n"
                "report   real test band\n\nsimulation screens candidates\ncheaply; few reach real data",
                "the headline protocol",
                "arm", 4, 54.6, 1.5, 21, 18),
        "p4": N("(4)  transfer shape, then\n       recalibrate on real",
                "search   simulated movies\nthen      re-fit the intensity\n"
                "              threshold on real data\nselect    real validation band\n"
                "report   real test band",
                "the threshold is movie-specific",
                "arm", 4, 77.9, 1.5, 21, 18),
    },
    edges=[
        E("real", "annot", "annotate by hand"),
        E("annot", "measure", "annotated coordinates"),
        E("real", "measure", "pixels", rad=-0.42, t=0.5),

        E("measure", "sim", "configure the simulator", rad=-0.04, t=0.42),
        E("sim", "movies", "generate, seeded"),
        E("movies", "harness", "check"),
        E("measure", "harness", "the same reference, both sides", "dashed", rad=0.28, t=0.5),

        E("movies", "detector", "simulated movies", rad=0.05, t=0.40),
        E("real", "detector", "or the real recording", rad=-0.06, t=0.45),
        E("detector", "scorer", "detected events"),
        E("annot", "scorer", "annotations as ground truth", "dashed", rad=-0.16, t=0.5),

        E("scorer", "search", "F1 of this trial", rad=0.0, t=0.45),
        E("search", "detector", "next parameter set", "dashed", rad=0.28, t=0.5),

        E("search", "p1", "", rad=0.05),
        E("search", "p2", "", rad=0.03),
        E("search", "p3", "", rad=0.0),
        E("search", "p4", "", rad=-0.03),
        E("p3", "controls", "", "dashed", rad=0.0),

        # The two feedback paths named in the caption.
        E("detector", "measure",
          "\u2460  two calibration targets are detector-mediated,\n"
          "so the detector runs inside the calibration",
          "feedback", rad=0.34, t=0.5),
        E("movies", "search",
          "\u2461  the simulator's output feeds the search,\n"
          "whose winner is then judged on real data",
          "feedback", rad=-0.28, t=0.52),
    ],
)



# ===========================================================================
# DETAILED -- the same pipeline down to script and file names
# ===========================================================================

DETAILED = Graph(
    title="smTIRF SCP pipeline — scripts, files, and the six evaluation designs",
    figsize=(26, 17),
    # Lane titles are set rotated in the left gutter, so their text length is bounded by
    # the band HEIGHT, not the width -- keep them short.
    lanes=[
        (1, "1  Real data", 85.5, 99.5),
        (2, "2  Simulator\nmeasure → build → judge", 60.0, 84.0),
        (3, "3  SCP detector", 46.5, 58.5),
        (4, "4  Optimization", 31.5, 45.0),
        (5, "5  Evaluation designs", 12.0, 30.0),
        (6, "6  Held-out results", 0.0, 10.5),
        (7, "7  Artifacts", -8.5, -1.5),
    ],
    nodes={
        # ---- lane 1: the real recording and its annotation -----------------
        "raw": N("realdata/vid.tiff", "dual-channel · 2×256×512 · 501 frames · 16-bit",
                 kind="real", lane=1, x=8, y=88.5, w=17, h=7),
        "prep": N("Parameter_opt/prepare_real_movie.py",
                  "pick the annotated channel (median SNR ≈ 11)",
                  kind="sim", lane=1, x=27.5, y=88.5, w=17, h=7),
        "ch1": N("realdata/vid_ch1.tiff", "(T, Y, X) single-channel stack",
                 kind="real", lane=1, x=47, y=88.5, w=15, h=7),
        "gui": N("GUI2/smTIRF_tool.py", "Flask annotation GUI",
                 kind="sim", lane=1, x=64.5, y=88.5, w=14, h=7),
        "labels": N("coord.npy [y, x]  +  class.npy",
                    "243 target · 91 off-target · moving",
                    kind="real", lane=1, x=81, y=88.5, w=18, h=7),

        # ---- lane 2 row A: full-movie calibration --------------------------
        "meas_full": N("measure_real_movie.py", "whole-movie statistics",
                       kind="sim", lane=2, x=8, y=75.5, w=16, h=7),
        "ref_full": N("config/real_movie_reference.json", "calibration targets",
                      kind="real", lane=2, x=27, y=75.5, w=16, h=7),
        "gen_full": N("generate_dataset.py", "config/real_calibrated.yaml · seeded",
                      kind="sim", lane=2, x=46, y=75.5, w=16, h=7),
        "ds_full": N("dataset/{train,val}/ + manifest.json",
                     "movies · (x,y) label JSON · [y,x] npy",
                     kind="real", lane=2, x=65, y=75.5, w=16, h=7),
        "gates": N("validate_against_real.py",
                   "gates T1–T5, T7  ·  tests/test_contracts.py",
                   kind="note", lane=2, x=84, y=75.5, w=15, h=7),

        # ---- lane 2 row B: leak-free recalibration + legacy baseline -------
        "meas_band": N("measure_real_movie.py --band 0 256",
                       "leak-free: validation-band labels only",
                       kind="sim", lane=2, x=8, y=62.5, w=16, h=8),
        "ref_band": N("real_movie_reference_valband.json", "test band influenced nothing",
                      kind="real", lane=2, x=27, y=62.5, w=16, h=8),
        "gen_band": N("generate_dataset.py\n--config real_calibrated_valband.yaml",
                      "same seeds as the full arm",
                      kind="sim", lane=2, x=46, y=62.5, w=16, h=8),
        "ds_band": N("dataset_valband/", "leak-free simulated movies",
                     kind="real", lane=2, x=65, y=62.5, w=16, h=8),
        "ds_bt": N("generate_dataset_bt.py → dataset_bt/",
                   "legacy simulator · no real calibration\nkept only to reproduce the baseline",
                   kind="real", lane=2, x=84, y=62.5, w=15, h=8),

        # ---- lane 3: the detector, in processing order ---------------------
        "bp": N("bandpass", "− Gaussian σ=8, clip, min–max",
                kind="detect", lane=3, x=8, y=49.5, w=12.4, h=7),
        "log": N("LoG  blob_log", "min_sigma … max_sigma, threshold",
                 kind="detect", lane=3, x=23.4, y=49.5, w=12.4, h=7),
        "link": N("KD-tree linking", "max_displacement",
                  kind="detect", lane=3, x=38.8, y=49.5, w=12.4, h=7),
        "cls": N("event classification", "circularity · min_pulsation_frames · CV",
                 kind="detect", lane=3, x=54.2, y=49.5, w=12.4, h=7),
        "nms": N("event NMS", "3.0 px duplicate suppression",
                 kind="detect", lane=3, x=69.6, y=49.5, w=12.4, h=7),
        "events": N("events JSON", "(x, y) per selected event",
                    kind="real", lane=3, x=85, y=49.5, w=12.4, h=7),

        # ---- lane 4: search, scorer, selection -----------------------------
        "opt": N("SCP_parametrizer_002.py\nOptuna TPE search",
                 "7 parameters · seed 42 · per-run timeout",
                 kind="tune", lane=4, x=8, y=34, w=21, h=8),
        "scorer": N("scorer: greedy 1:1 match, |dx|,|dy| ≤ 3 px",
                    "recall · precision · F1 (β = 1)",
                    kind="tune", lane=4, x=32, y=34, w=23, h=8),
        "topk": N("top-K candidates", "K = 6 … 10",
                  kind="tune", lane=4, x=58, y=34, w=15, h=8),
        "method": N("METHOD switch", "sim_only | sim_guided_real | direct_real",
                    kind="tune", lane=4, x=76, y=34, w=23, h=8),

        # ---- lane 5: the six evaluation designs (METHODS_SPEC 4.3 a-f) -----
        "arm_a": N("(a)  direct on real",
                   "search   real train [0,170)\nselect    real val   [170,340)\n"
                   "report   real test  [340,512)\n\n100+ real detector runs",
                   kind="arm", lane=5, x=8, y=13, w=13.6, h=15),
        "arm_b": N("(b)  sim only",
                   "search   simulated train\nselect    simulated val\n"
                   "report   real, scored once\n\nclassic design, completeness",
                   kind="arm", lane=5, x=23.4, y=13, w=13.6, h=15),
        "arm_c": N("(c)  sim-guided  ← headline",
                   "search   dataset/ (simulated)\nselect    real val   [0,256)\n"
                   "report   real test  [256,512)\n\n30 trials · K=6 · 6 real runs",
                   kind="arm", lane=5, x=38.8, y=13, w=13.6, h=15),
        "arm_f": N("(f)  sim-guided, leak-free",
                   "simulator calibrated on the\nvalidation band only\n(dataset_valband/)\n\n"
                   "same search + selection as (c)",
                   kind="arm", lane=5, x=54.2, y=13, w=13.6, h=15),
        "arm_d": N("(d)  shape transfer +\n      threshold recalibration",
                   "search   simulated (7 params)\nthen      real threshold grid\n"
                   "              0.20 / 0.26 / 0.32\nselect on real val band",
                   kind="arm", lane=5, x=69.6, y=13, w=13.6, h=15),
        "arm_e": N("(e)  random-search control",
                   "6 uniform draws · no simulator\nsame scorer, bands, selection,\n"
                   "real-run budget as (c)\n\nisolates the simulator's value",
                   kind="arm", lane=5, x=85, y=13, w=13.6, h=15),

        # ---- lane 6: numbers that were used for no decision ---------------
        "r_a": N("test F1  0.800",
                 "recall 94.9% · precision 69.2%\ntrain 0.839 · val 0.806\n→ reference config 0.847",
                 kind="note", lane=6, x=8, y=0.5, w=13.6, h=9.5),
        "r_b": N("real, scored once", "reported for completeness",
                 kind="note", lane=6, x=23.4, y=0.5, w=13.6, h=9.5),
        "r_c": N("test F1  0.7143",
                 "recall 90.1% · precision 59.2%\nvs 0.256 legacy simulator\n(2.8× · above the 0.70 bar)",
                 kind="note", lane=6, x=38.8, y=0.5, w=13.6, h=9.5),
        "r_f": N("test F1  0.6852",
                 "recall 66.7% · precision 70.5%\ngap to (c) = 0.029 F1\n— an upper bound on leakage",
                 kind="note", lane=6, x=54.2, y=0.5, w=13.6, h=9.5),
        "r_d": N("real threshold 0.20",
                 "F1 0.699 in 184 s\n0.15 → 0.213 (blobs hijack\ntracking) · 0.12 times out",
                 kind="note", lane=6, x=69.6, y=0.5, w=13.6, h=9.5),
        "r_e": N("test F1  0.4505",
                 "6 random configs + real selection\n0.5591 picking by simulated\nscore alone · simulator = +0.26",
                 kind="note", lane=6, x=85, y=0.5, w=13.6, h=9.5),

        # ---- lane 7: where it all lands -----------------------------------
        "artifacts": N("experiments/<arm>/optimization_results*.json   →   "
                       "unified_benchmark_rescoring/{unified_benchmark,plot_unified,rescore_all}.py",
                       "one scorer for every quoted number  ·  output/*.csv  ·  "
                       "output/figures/*.png  ·  figures/make_figure1.py",
                       kind="note", lane=7, x=8, y=-8, w=91, h=6),
    },
    edges=[
        E("raw", "prep"), E("prep", "ch1"), E("ch1", "gui"), E("gui", "labels"),

        E("ch1", "meas_full", "pixels: whole frame", rad=0.05, t=0.30),
        E("labels", "meas_full", "labels: whole movie", rad=0.02, t=0.35),
        E("labels", "meas_band", "labels: x < 256 only", "dashed", rad=-0.05, t=0.20),

        E("meas_full", "ref_full"), E("ref_full", "gen_full"),
        E("gen_full", "ds_full"), E("ds_full", "gates"),
        E("meas_band", "ref_band"), E("ref_band", "gen_band"),
        E("gen_band", "ds_band"), E("ds_band", "gates", "", "dashed", rad=-0.10),

        E("ds_full", "bp", "simulated", rad=0.05, t=0.22),
        E("ds_band", "bp", "leak-free", "dashed", rad=0.08, t=0.14),
        E("ds_bt", "bp", "baseline", "dashed", rad=0.11, t=0.10),
        E("ch1", "bp", "the real movie", rad=-0.05, t=0.18),

        E("bp", "log"), E("log", "link"), E("link", "cls"),
        E("cls", "nms"), E("nms", "events"),

        E("events", "scorer", "detections", rad=0.06, t=0.30),
        E("labels", "scorer", "ground truth, sliced by x band", "dashed", rad=0.06, t=0.09),
        E("scorer", "opt", "F1 objective", rad=0.16),
        E("opt", "log", "7 parameters per trial", "dashed", rad=-0.18, t=0.55),
        E("opt", "topk", "ranked trials", rad=-0.08, t=0.62),
        E("topk", "method"),

        E("method", "arm_a", rad=0.05), E("method", "arm_b", rad=0.04),
        E("method", "arm_c", rad=0.03), E("method", "arm_f", rad=0.02),
        E("method", "arm_d", rad=0.01),
        E("method", "arm_e", "no simulator", "dashed", t=0.72),

        E("arm_a", "r_a"), E("arm_b", "r_b"), E("arm_c", "r_c"),
        E("arm_f", "r_f"), E("arm_d", "r_d"), E("arm_e", "r_e"),

        E("r_a", "artifacts"), E("r_c", "artifacts"),
        E("r_f", "artifacts"), E("r_e", "artifacts"),
    ],
)

GRAPHS = {"overview": OVERVIEW, "detailed": DETAILED}


# ===========================================================================
# Rendering
# ===========================================================================

def _anchors(g, src, dst):
    """Pick exit/entry points: vertical when the lanes differ, horizontal when not.

    The exit x aims at the destination (clamped inside the source box), so fan-outs like
    select -> five arms do not all leave from the same pixel.
    """
    a, b = g.nodes[src], g.nodes[dst]
    c1, c2 = a.x + a.w / 2, b.x + b.w / 2

    if (a.y < b.y + b.h) and (b.y < a.y + a.h):            # side by side
        if c2 >= c1:
            return a.x + a.w, a.y + a.h / 2, b.x, b.y + b.h / 2
        return a.x, a.y + a.h / 2, b.x + b.w, b.y + b.h / 2

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    ax = clamp(c2, a.x + 1.5, a.x + a.w - 1.5)
    bx = clamp(c1, b.x + 1.5, b.x + b.w - 1.5)
    if b.y + b.h <= a.y:                                   # destination below
        return ax, a.y, bx, b.y + b.h
    return ax, a.y + a.h, bx, b.y                          # destination above


def _arc3_point(p0, p2, rad, t):
    """Point at parameter `t` on the quadratic Bezier that arc3 actually draws.

    matplotlib's ConnectionStyle.Arc3 puts the control point at
    midpoint + rad * (dy, -dx) -- the RIGHT-hand perpendicular, note, not the left --
    so a label offset guessed from the chord lands on the wrong side of the curve.
    Evaluating the same Bezier keeps every label sitting on its own arrow.
    """
    (x1, y1), (x2, y2) = p0, p2
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    cx, cy = mx + rad * dy, my - rad * dx
    u = 1 - t
    return (u * u * x1 + 2 * u * t * cx + t * t * x2,
            u * u * y1 + 2 * u * t * cy + t * t * y2)


def _rect_clearance(px, py, node, pad=0.6):
    """Distance from a point to a padded node box; 0 when the point is inside it."""
    dx = max(node.x - pad - px, px - (node.x + node.w + pad), 0.0)
    dy = max(node.y - pad - py, py - (node.y + node.h + pad), 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _place_label(g, p0, p2, rad, preferred_t, placed):
    """Slide an edge label along its own curve to the clearest spot.

    Hand-tuned label positions rot the moment a box moves, so `t` in the edge list is
    only a preference: sample the curve, score each candidate on distance from every
    node box and from labels already placed, and keep the best. Clearances are capped so
    a label prefers "clear enough, near where I asked" over drifting to the emptiest
    corner of the figure.
    """
    best, best_score = None, -1e9
    for t in [preferred_t] + [0.06 + 0.04 * i for i in range(23)]:
        px, py = _arc3_point(p0, p2, rad, t)
        box_clear = min((_rect_clearance(px, py, n) for n in g.nodes.values()), default=99.0)
        lbl_clear = min(((px - lx) ** 2 + (py - ly) ** 2) ** 0.5
                        for lx, ly in placed) if placed else 99.0
        score = (min(box_clear, 1.4) + 0.45 * min(lbl_clear, 5.0)
                 - 0.9 * abs(t - preferred_t))
        if score > best_score:
            best, best_score = (px, py), score
    return best


def render_png(g, out_png, out_svg, dpi=150):
    fig, ax = plt.subplots(figsize=g.figsize)
    y_lo = min(y0 for _, _, y0, _ in g.lanes)
    y_hi = max(y1 for _, _, _, y1 in g.lanes)
    span = y_hi - y_lo
    ax.set_xlim(-1.5, 100.5)
    ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.05 * span)
    ax.axis("off")

    ax.text(50, y_hi + 0.030 * span, g.title, ha="center", va="center",
            fontsize=18, fontweight="bold", color="#222222")

    # Lane bands first, so nodes and arrows sit on top of them.
    for lane_id, title, y0, y1 in g.lanes:
        ax.add_patch(FancyBboxPatch(
            (-1.0, y0), 101.0, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.8",
            facecolor="#FAFAFA" if lane_id % 2 else "#F2F4F5",
            edgecolor="#DCDCDC", linewidth=0.8, zorder=0))
        ax.text(3.0, (y0 + y1) / 2, title, rotation=90, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#666666", zorder=1,
                linespacing=1.35)

    for node in g.nodes.values():
        face, edge = KINDS[node.kind]
        ax.add_patch(FancyBboxPatch(
            (node.x, node.y), node.w, node.h,
            boxstyle="round,pad=0,rounding_size=0.7",
            facecolor=face, edgecolor=edge, linewidth=1.7, zorder=3))

        cx = node.x + node.w / 2
        n_title = node.label.count("\n") + 1
        ax.text(cx, node.y + node.h - 0.9 - 0.85 * (n_title - 1), node.label,
                ha="center", va="top", fontsize=9.6, fontweight="bold",
                color="#1A1A1A", zorder=4, linespacing=1.25)
        if node.sub:
            wrapped = "\n".join(
                "\n".join(textwrap.wrap(line, width=max(20, int(node.w * 2.7)))) or ""
                for line in node.sub.split("\n"))
            ax.text(cx, node.y + node.h - 1.2 - 2.3 * n_title, wrapped,
                    ha="center", va="top", fontsize=7.4, color="#3C3C3C",
                    zorder=4, linespacing=1.5)
        if node.foot:
            ax.text(cx, node.y + 0.8, node.foot, ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold", color=edge,
                    zorder=4, linespacing=1.35)

    placed = []
    for e in g.edges:
        x1, y1, x2, y2 = _anchors(g, e.src, e.dst)
        colour, lw, dash = EDGE_STYLES[e.style]
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={e.rad}",
            arrowstyle="-|>", mutation_scale=15 if e.style == "feedback" else 14,
            linewidth=lw, linestyle=dash, color=colour,
            shrinkA=1.0, shrinkB=2.0,
            zorder=6 if e.style == "feedback" else 2))
        if e.label:
            lx, ly = _place_label(g, (x1, y1), (x2, y2), e.rad, e.t, placed)
            placed.append((lx, ly))
            feedback = e.style == "feedback"
            ax.text(lx, ly, e.label, ha="center", va="center",
                    fontsize=7.8 if feedback else 7.4,
                    fontweight="bold" if feedback else "normal",
                    color=colour if feedback else "#333333",
                    zorder=7 if feedback else 5, linespacing=1.4,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF",
                              edgecolor=colour if feedback else "none",
                              linewidth=0.9, alpha=0.94))

    used = [k for k in KINDS if any(n.kind == k for n in g.nodes.values())]
    handles = [FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0",
                              facecolor=KINDS[k][0], edgecolor=KINDS[k][1], linewidth=1.7)
               for k in used]
    labels = [KIND_LEGEND[k] for k in used]
    if any(e.style == "feedback" for e in g.edges):
        colour, lw, _ = EDGE_STYLES["feedback"]
        handles.append(Line2D([0], [0], color=colour, linewidth=lw))
        labels.append("feedback path (① ②)")
    ax.legend(handles, labels,
              loc="upper center", bbox_to_anchor=(0.5, 0.015), ncol=len(handles),
              frameon=False, fontsize=9.5, handlelength=1.7, handleheight=1.0,
              columnspacing=2.0)

    fig.tight_layout()
    for path in (out_png, out_svg):
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _flat(text):
    return text.replace("\n", "\\n").replace('"', "'")


def render_dot(g, out_dot):
    lines = [
        "digraph smtirf_pipeline {",
        '  graph [rankdir=TB, fontname="Helvetica", fontsize=11, compound=true,',
        f'         labelloc=t, label="{_flat(g.title)}"];',
        '  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9];',
        '  edge  [fontname="Helvetica", fontsize=8, color="#4D4D4D"];',
    ]
    for lane_id, title, _y0, _y1 in g.lanes:
        lines.append(f"  subgraph cluster_{lane_id} {{")
        lines.append(f'    label="{title.replace(chr(10), " — ")}"; style="rounded,filled"; '
                     'fillcolor="#F7F7F7"; color="#DCDCDC"; fontsize=11;')
        for key, n in g.nodes.items():
            if n.lane != lane_id:
                continue
            face, edge = KINDS[n.kind]
            text = _flat(n.label)
            for extra in (n.sub, n.foot):
                if extra:
                    text += "\\n" + _flat(extra)
            lines.append(f'    {key} [label="{text}", fillcolor="{face}", color="{edge}"];')
        lines.append("  }")
    for e in g.edges:
        attrs = []
        if e.label:
            attrs.append(f'label="{_flat(e.label)}"')
        if e.style == "dashed":
            attrs.append('style=dashed, color="#8A8A8A"')
        elif e.style == "feedback":
            colour = EDGE_STYLES["feedback"][0]
            attrs.append(f'color="{colour}", penwidth=2.0, fontcolor="{colour}"')
        suffix = f' [{", ".join(attrs)}]' if attrs else ""
        lines.append(f"  {e.src} -> {e.dst}{suffix};")
    lines.append("}")
    out_dot.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_mermaid(g, out_mmd):
    lines = ["%% generated by figures/make_workflow_figure.py", "flowchart TB"]
    for lane_id, title, _y0, _y1 in g.lanes:
        lines.append(f'  subgraph L{lane_id}["{title.replace(chr(10), " — ")}"]')
        lines.append("    direction LR")
        for key, n in g.nodes.items():
            if n.lane != lane_id:
                continue
            text = n.label.replace("\n", "<br/>")
            if n.sub:
                text += "<br/><i>" + n.sub.replace("\n", "<br/>") + "</i>"
            if n.foot:
                text += "<br/><b>" + n.foot.replace("\n", "<br/>") + "</b>"
            lines.append(f'    {key}["{text.replace(chr(34), chr(39))}"]')
        lines.append("  end")
    # linkStyle indices are the order in which edges are declared, which is exactly
    # the order of g.edges -- so enumerate rather than trying to count emitted lines.
    for i, e in enumerate(g.edges):
        arrow = {"dashed": "-.->", "feedback": "==>"}.get(e.style, "-->")
        if e.label:
            label = e.label.replace("\n", " ").replace('"', "'")
            lines.append(f'  {e.src} {arrow}|"{label}"| {e.dst}')
        else:
            lines.append(f"  {e.src} {arrow} {e.dst}")
    for i, e in enumerate(g.edges):
        if e.style == "feedback":
            colour = EDGE_STYLES["feedback"][0]
            lines.append(f"  linkStyle {i} stroke:{colour},stroke-width:2.5px,color:{colour};")
    for kind, (face, edge) in KINDS.items():
        members = [k for k, n in g.nodes.items() if n.kind == kind]
        if members:
            lines.append(f"  classDef {kind} fill:{face},stroke:{edge},stroke-width:2px,color:#1a1a1a;")
            lines.append(f"  class {','.join(members)} {kind};")
    out_mmd.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", choices=sorted(GRAPHS), default="overview",
                    help="overview (default) = 14 boxes; detailed = scripts and files")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--stem", default=None,
                    help="output basename (default: pipeline_workflow[_detailed])")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--only", choices=["png", "dot", "mermaid"], action="append",
                    help="restrict output formats (repeatable); default is all three")
    args = ap.parse_args()

    g = GRAPHS[args.level]
    stem = args.stem or (STEM if args.level == "overview" else f"{STEM}_{args.level}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ["png", "dot", "mermaid"])
    base = args.out_dir / stem

    if "png" in want:
        png, svg = base.with_suffix(".png"), base.with_suffix(".svg")
        render_png(g, png, svg, dpi=args.dpi)
        print(f"wrote {png}")
        print(f"wrote {svg}")

    if "dot" in want:
        dot = base.with_suffix(".dot")
        render_dot(g, dot)
        print(f"wrote {dot}")
        exe = shutil.which("dot")
        if exe:
            target = base.with_name(stem + "_graphviz.png")
            subprocess.run([exe, "-Tpng", str(dot), "-o", str(target)], check=True)
            print(f"wrote {target}")
        else:
            print("  (graphviz `dot` not on PATH; skipped its auto-layout render — "
                  "the .dot itself is written and valid)")

    if "mermaid" in want:
        mmd = base.with_suffix(".mmd")
        render_mermaid(g, mmd)
        print(f"wrote {mmd}")

    print(f"\n{args.level}: {len(g.nodes)} nodes, {len(g.edges)} edges, {len(g.lanes)} lanes.")


if __name__ == "__main__":
    main()
