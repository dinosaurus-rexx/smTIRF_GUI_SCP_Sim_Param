"""
Re-score SCP, Papylio and DeepFRET with the SAME scorer used for parameter tuning.

Motivation: the Snakemake benchmark scores with Hungarian assignment at a 2 px
Euclidean threshold over the whole movie, while every tuning number in the thesis
comes from the parametrizer's greedy per-axis +/-3 px scorer on a held-out band.
The same SCP configuration therefore has two F1 values on record (0.850 and 0.847),
which is confusing to read. This script puts all three algorithms through the
tuning scorer so one table can be reported.

Nothing is re-implemented: compute_metrics / match_detections_to_positives are
imported from the parametrizer, so this is the same code path the search used.

Inputs are the detection/selection arrays the Snakemake pipeline already wrote,
shape (N, 2) in [y, x] order.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import SCP_parametrizer_002 as pm  # noqa: E402

BT = Path(r"c:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms")
RESULTS = BT / "results"

ALGORITHMS = ["scp", "papylio", "deepfret"]
STAGES = ["detections", "selections"]

# (label, (x_min, x_max)) -- half-open on x, the same convention as _in_band().
BANDS = [
    ("whole movie      x[0,512)", (0, 512)),
    ("direct-CV train  x[0,170)", pm.DIRECT_TRAIN_XBAND),
    ("direct-CV val    x[170,340)", pm.DIRECT_VAL_XBAND),
    ("direct-CV TEST   x[340,512)", pm.DIRECT_TEST_XBAND),
    ("sim-guided TEST  x[256,512)", pm.REAL_TEST_XBAND),
]


def load_xy(npy_path):
    """(N, 2) in [y, x] -> list of (x, y), matching the parametrizer's spot format."""
    arr = np.load(npy_path)
    if arr.size == 0:
        return []
    return [(float(r[1]), float(r[0])) for r in arr]


def in_band(spot, xband):
    return xband[0] <= spot[0] < xband[1]


def main():
    positives, negatives = pm.load_test_labels(pm.TEST_COORD_NPY, pm.TEST_CLASS_NPY)
    print(f"annotations: {len(positives)} target, {len(negatives)} off-target")
    print(f"scorer: greedy 1:1, per-axis margin {pm.MATCH_MARGIN} px, "
          f"F{pm.FBETA:g}; every unmatched detection counts as FP\n")

    loaded = {}
    for alg in ALGORITHMS:
        for stage in STAGES:
            p = RESULTS / alg / stage / "rep1.npy"
            loaded[(alg, stage)] = load_xy(p) if p.exists() else None
            if loaded[(alg, stage)] is None:
                print(f"  MISSING: {p}")

    for band_label, xband in BANDS:
        pos_b = [s for s in positives if in_band(s, xband)]
        neg_b = [s for s in negatives if in_band(s, xband)]
        print("=" * 88)
        print(f"{band_label}   |   {len(pos_b)} target, {len(neg_b)} off-target annotations")
        print("=" * 88)
        print(f"{'algorithm':<10} {'stage':<11} {'det':>6} {'TP':>5} {'FP':>6} "
              f"{'FN':>5} {'recall':>8} {'precis':>8} {'F1':>7} {'offtgt hit':>11}")
        for alg in ALGORITHMS:
            for stage in STAGES:
                det = loaded[(alg, stage)]
                if det is None:
                    continue
                det_b = [d for d in det if in_band(d, xband)]
                m = pm.compute_metrics(pos_b, neg_b, det_b)
                print(f"{alg:<10} {stage:<11} {m['detected_count']:>6} "
                      f"{m['true_positives']:>5} {m['false_positives']:>6} "
                      f"{m['false_negatives']:>5} {m['recall']:>7.1%} "
                      f"{m['precision']:>7.1%} {m['fbeta']:>7.4f} "
                      f"{m['negatives_hit']:>11}")
        print()


if __name__ == "__main__":
    main()
