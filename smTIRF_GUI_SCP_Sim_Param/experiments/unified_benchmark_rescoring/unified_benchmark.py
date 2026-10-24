"""
Unified re-scoring of SCP, Papylio and DeepFRET with the parameter-tuning scorer.

Writes per-class and global benchmark CSVs to output/, in the same shape as the
Snakemake pipeline's class_benchmarks/global_benchmarks, so plot_unified.py can
regenerate the thesis figures from them.

Scoring rule (identical to the one used for every tuning number in the thesis):
  * greedy 1:1 assignment, closest pair first
  * a detection matches an annotation when |dx| <= 3 AND |dy| <= 3 px (per axis)
  * every detection not matched to a TARGET annotation is a false positive,
    including detections sitting on annotated off-target spots
  * recall = TP / 243 targets, precision = TP / all detections, F1 = harmonic mean

Two matchings are computed and reconciled:
  A) targets-only  -- pm.compute_metrics, byte-for-byte the tuning code path.
     This is the source of the reported global precision/recall/F1.
  B) all-annotation -- the same greedy matcher run against targets AND off-target
     spots at once, needed to split detections three ways for the bar chart
     (target / off-target / pure noise).
Any disagreement in the target count between A and B is printed; the two can
differ only where a detection falls within margin of both an annotated target and
an annotated off-target spot.

Region. By default the whole movie, x[0,512). `--band XMIN XMAX` restricts both the
annotations and the detections to a half-open x band (the same convention as the
parametrizer's `_in_band`), so the benchmark can be reported on the band a tuning
number was reported on -- notably x[256,512), the reference detector's held-out
band (F1 0.847). Band runs write to output_x{XMIN}_{XMAX}/ so the whole-movie
output is never overwritten.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import SCP_parametrizer_002 as pm  # noqa: E402

BT = Path(r"c:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms")
RESULTS = BT / "results"

# display_name values from config/realdata.yaml
ALGORITHMS = [("deepfret", "DeepFRET"), ("papylio", "Papylio"), ("scp", "SCP")]
TARGET_NAME = "Target Interactions"
OFFTARGET_NAME = "Off-Target Interactions"
NOISE_NAME = "Pure Noise"

TARGET_CLASS = pm.TEST_POSITIVE_CLASS      # 0
OFFTARGET_CLASS = pm.TEST_NEGATIVE_CLASS   # 1
MARGIN = pm.MATCH_MARGIN                   # 3


def load_xy(path):
    """(N, 2) in [y, x] -> list of (x, y)."""
    arr = np.load(path)
    if arr.size == 0:
        return []
    return [(float(r[1]), float(r[0])) for r in arr]


def load_annotations():
    coord = np.load(pm.TEST_COORD_NPY)
    cls = np.load(pm.TEST_CLASS_NPY)
    spots = [(int(c[1]), int(c[0])) for c in coord]     # [y, x] -> (x, y)
    return spots, cls


def in_band(spot, xband):
    """Half-open on x, matching pm._in_band."""
    return xband[0] <= spot[0] < xband[1]


def split_by_class(spots, cls, detections):
    """
    Greedy 1:1 match detections against ALL annotations, then split the detections
    three ways by the class of whatever each one matched.
    """
    pairs, _, unmatched_det = pm.match_detections_to_positives(spots, detections, margin=MARGIN)
    index_of = {tuple(s): i for i, s in enumerate(spots)}
    n_target = n_offtarget = 0
    for p in pairs:
        c = cls[index_of[tuple(p['positive'])]]
        if c == TARGET_CLASS:
            n_target += 1
        elif c == OFFTARGET_CLASS:
            n_offtarget += 1
    return n_target, n_offtarget, len(unmatched_det)


def safe_div(n, d):
    return n / d if d > 0 else 0.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--band", nargs=2, type=int, metavar=("XMIN", "XMAX"),
                    default=None,
                    help="restrict scoring to the half-open x band [XMIN, XMAX). "
                         "Default: the whole movie, 0 512.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory. Default: output/ for the whole movie, "
                         "output_x{XMIN}_{XMAX}/ for a band.")
    return ap.parse_args()


def main():
    args = parse_args()
    xband = tuple(args.band) if args.band else (0, 512)
    whole = xband == (0, 512)
    out = args.out or (HERE / ("output" if whole else f"output_x{xband[0]}_{xband[1]}"))
    out.mkdir(parents=True, exist_ok=True)

    all_spots, all_cls = load_annotations()
    keep = [i for i, s in enumerate(all_spots) if in_band(s, xband)]
    spots = [all_spots[i] for i in keep]
    cls = np.asarray([all_cls[i] for i in keep])

    positives = [s for s, c in zip(spots, cls) if c == TARGET_CLASS]
    negatives = [s for s, c in zip(spots, cls) if c == OFFTARGET_CLASS]
    n_targets, n_offtargets = len(positives), len(negatives)

    region = (f"whole movie, columns 0-512" if whole
              else f"held-out band x[{xband[0]},{xband[1]})")
    print(f"region: {region}")
    print(f"annotations in region: {n_targets} target, {n_offtargets} off-target, "
          f"{len(spots)} total  (of {len(all_spots)} on the whole movie)")
    print(f"scorer: greedy 1:1, per-axis margin {MARGIN} px, F{pm.FBETA:g}, "
          f"every unmatched detection a false positive\n")

    summary_rows = []
    reconciliation = []

    for alg, display in ALGORITHMS:
        det = [d for d in load_xy(RESULTS / alg / "detections" / "rep1.npy")
               if in_band(d, xband)]
        sel = [d for d in load_xy(RESULTS / alg / "selections" / "rep1.npy")
               if in_band(d, xband)]

        # (A) tuning code path -- the source of the global numbers
        m_det = pm.compute_metrics(positives, negatives, det)
        m_sel = pm.compute_metrics(positives, negatives, sel)

        # (B) three-way split for the bar chart
        t_det, o_det, noise_det = split_by_class(spots, cls, det)
        t_sel, o_sel, noise_sel = split_by_class(spots, cls, sel)

        for stage, m, t in (("detection", m_det, t_det), ("selection", m_sel, t_sel)):
            if m['true_positives'] != t:
                reconciliation.append(
                    f"{display} {stage}: targets-only match {m['true_positives']}, "
                    f"all-annotation match {t} (difference {m['true_positives'] - t})")

        df_class = pd.DataFrame([
            {"Class": TARGET_NAME, "Raw_Count": t_det, "Filtered_Count": t_sel,
             "Retention": safe_div(t_sel, t_det),
             "Recall_Raw": safe_div(t_det, n_targets),
             "Recall_Filtered": safe_div(t_sel, n_targets)},
            {"Class": OFFTARGET_NAME, "Raw_Count": o_det, "Filtered_Count": o_sel,
             "Retention": safe_div(o_sel, o_det),
             "Recall_Raw": safe_div(o_det, n_offtargets),
             "Recall_Filtered": safe_div(o_sel, n_offtargets)},
            {"Class": NOISE_NAME, "Raw_Count": noise_det, "Filtered_Count": noise_sel,
             "Retention": safe_div(noise_sel, noise_det),
             "Recall_Raw": 0.0, "Recall_Filtered": 0.0},
        ])
        df_class.to_csv(out / f"class_benchmarks_{alg}.csv", index=False)

        df_global = pd.DataFrame(
            {"Raw": [m_det['precision'], m_det['recall'], m_det['fbeta']],
             "Filtered": [m_sel['precision'], m_sel['recall'], m_sel['fbeta']]},
            index=["Global_Precision", "Global_Recall", "Global_F1_Score"])
        df_global.to_csv(out / f"global_benchmarks_{alg}.csv")

        for stage, m in (("detection", m_det), ("selection", m_sel)):
            summary_rows.append({
                "algorithm": display, "stage": stage,
                "detections": m['detected_count'],
                "targets_found": m['true_positives'],
                "off_target_hit": m['negatives_hit'],
                "false_positives": m['false_positives'],
                "false_negatives": m['false_negatives'],
                "recall": m['recall'], "precision": m['precision'], "f1": m['fbeta'],
            })

        print(f"{display}")
        print(f"  detection  {m_det['detected_count']:>5} dets  "
              f"R {m_det['recall']:>6.1%}  P {m_det['precision']:>6.1%}  "
              f"F1 {m_det['fbeta']:.4f}   "
              f"[target {t_det}, off-target {o_det}, noise {noise_det}]")
        print(f"  selection  {m_sel['detected_count']:>5} dets  "
              f"R {m_sel['recall']:>6.1%}  P {m_sel['precision']:>6.1%}  "
              f"F1 {m_sel['fbeta']:.4f}   "
              f"[target {t_sel}, off-target {o_sel}, noise {noise_sel}]")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out / "summary.csv", index=False)

    meta = {
        "scorer": "greedy 1:1, per-axis margin 3 px (SCP_parametrizer_002.compute_metrics)",
        "match_margin_px": MARGIN,
        "fbeta": pm.FBETA,
        "region": region,
        "x_band": list(xband),
        "annotations": {"target": n_targets, "off_target": n_offtargets},
        "annotations_whole_movie": len(all_spots),
        "labels": {"coord": str(pm.TEST_COORD_NPY), "class": str(pm.TEST_CLASS_NPY)},
        "detection_arrays": str(RESULTS / "{alg}" / "{detections,selections}" / "rep1.npy"),
        "false_positive_rule": "every detection not matched to a target annotation",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print()
    if reconciliation:
        print("target-count reconciliation (targets-only vs all-annotation matching):")
        for line in reconciliation:
            print(f"  {line}")
    else:
        print("target-count reconciliation: the two matchings agree exactly.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
