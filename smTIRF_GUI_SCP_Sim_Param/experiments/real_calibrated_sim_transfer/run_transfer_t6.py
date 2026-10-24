"""
Gate T6: does tuning on the real-calibrated simulator transfer to the real movie?

This is the acceptance test the whole simulator exists to pass (SIMULATOR_SPEC.md
sections 1 and 7-T6). Everything else in section 7 measures whether the movies LOOK
right; only this measures whether parameters tuned on them WORK.

Protocol -- deliberately the existing one, unchanged:
    1. Optuna TPE search on the simulated train movies.
    2. Top-K configs each run ONCE on the real movie, sliced by x into a validation
       band (0-256, selects the winner) and a held-out test band (256-512, reported).
Nothing in the scorer or the bands is touched: METHOD = "sim_guided_real" and
SCP_parametrizer_002 is driven as-is, so the result is directly comparable to the
two rows in spec section 1.

    tuned on BT sim, applied to real    F1 0.256   (R 18.0% / P 44.4%)
    tuned directly on the real movie    F1 0.847   (R 94.6% / P 76.6%)
    target for this run                 F1 >= 0.70

Note which of those is the competitor: 0.256. The 0.847 comes from tuning on the
real movie's own labels and is an oracle, not a benchmark -- a surrogate is not
expected to beat the real thing. What T6 asks is whether the simulator has stopped
being useless for parameter selection.

TWO ARMS, and the second one matters:

    --arm full   the simulator calibrated against the WHOLE real movie (the default
                 reference). This is what spec T6 literally asks for.
    --arm valband the simulator calibrated against the VALIDATION BAND ONLY
                 (x 0-256), so the held-out test band never influenced the
                 simulator's parameters either.

The `full` arm has a subtle leak. Detector parameters are still selected on the
validation band only, so it is not circular in the usual sense -- but the simulator
itself was fitted to statistics of the entire movie, test band included. Its test
number is therefore not fully held out. The gap between the two arms is the honest
measure of how much of the result is real transfer rather than leakage, and it is
why both are run.

Cost. Each trial runs the detector on every train movie; a 501x256x512 movie takes
roughly 1-3 minutes depending on how many blobs the parameters produce. Size the run
with --trials and --train-movies, and expect hours. Run it in the background.

    python run_transfer_t6.py --arm full --trials 40 --train-movies 2
    python run_transfer_t6.py --arm valband --trials 40 --train-movies 2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")          # trap 2
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent                          # smTIRF_GUI_SCP_Sim_Param
SIMDIR = PROJ / "Simulation"
sys.path.insert(0, str(SIMDIR))
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import numpy as np                                 # noqa: E402

import SCP_parametrizer_002 as pm                  # noqa: E402

# Datasets built by Simulation/generate_dataset.py, one per arm.
ARM_DATASETS = {
    "full": SIMDIR / "dataset" / "manifest.json",
    "valband": SIMDIR / "dataset_valband" / "manifest.json",
}
ARM_NOTES = {
    "full": "simulator calibrated against the whole real movie (spec T6 as written; "
            "the test band influenced the simulator's calibration)",
    "valband": "simulator calibrated against the real validation band only "
               "(x 0-256), so the test band influenced nothing",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARM_DATASETS), default="full")
    ap.add_argument("--manifest", default=None, help="override the arm's dataset")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=6,
                    help="configs carried from the sim search to the real movie")
    ap.add_argument("--train-movies", type=int, default=2,
                    help="how many simulated train movies each trial is scored on")
    ap.add_argument("--timeout", type=int, default=900,
                    help="seconds per detector run before it is abandoned")
    args = ap.parse_args()

    manifest = Path(args.manifest) if args.manifest else ARM_DATASETS[args.arm]
    if not manifest.exists():
        sys.exit(f"dataset manifest not found: {manifest}\n"
                 "Build it first with Simulation/generate_dataset.py "
                 f"(--out-dir {manifest.parent.name} for this arm).")

    # ---- drive the parametrizer, changing nothing that affects scoring ----
    pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
    pm.PYTHON_CMD = [sys.executable]
    pm.DATASET_MANIFEST = str(manifest)
    pm.METHOD = "sim_guided_real"
    pm.N_TRIALS = args.trials
    pm.TOP_K = args.top_k
    pm.DETECTION_TIMEOUT = args.timeout
    pm.RESULTS_FILE = str(HERE / f"optimization_results_{args.arm}.json")
    # USE_BANDPASS, BANDPASS_BG_SIGMA, EVENT_NMS_RADIUS, INTENSITY_APERTURE,
    # MATCH_MARGIN, FBETA and the two x-bands are left at their module defaults on
    # purpose: those are the settings every number on record was produced with.

    print("=" * 78)
    print(f"GATE T6 -- ARM '{args.arm}'")
    print("=" * 78)
    print(f"  {ARM_NOTES[args.arm]}")
    print(f"  dataset   : {manifest}")
    print(f"  trials    : {args.trials}   top-k: {args.top_k}")
    print(f"  bands     : val x{pm.REAL_VAL_XBAND}  test x{pm.REAL_TEST_XBAND}")
    print(f"  scorer    : compute_metrics, MATCH_MARGIN={pm.MATCH_MARGIN}, "
          f"F-beta={pm.FBETA}  (untouched)")
    print(f"  detector  : bandpass={pm.USE_BANDPASS} sigma={pm.BANDPASS_BG_SIGMA} "
          f"event_nms={pm.EVENT_NMS_RADIUS} aperture={pm.INTENSITY_APERTURE}")
    print(f"  benchmarks: BT-sim-tuned 0.256 | direct-real 0.847 | T6 target >= 0.70")

    # Cap the number of train movies: cost is linear in it and every trial pays.
    meta, train, val = pm.load_movies_from_manifest(str(manifest))
    if args.train_movies and len(train) > args.train_movies:
        train = train[: args.train_movies]
    print(f"  sim train : {len(train)} movie(s) -> {[m['name'] for m in train]}")
    if not train:
        sys.exit("manifest has no train movies")

    # Patch the loader so run_sim_guided() sees exactly this truncated set. Cleaner
    # than editing the manifest, and it leaves the dataset on disk untouched.
    def _loader(_path, _meta=meta, _train=train, _val=val):
        return _meta, _train, _val
    pm.load_movies_from_manifest = _loader

    t0 = time.time()
    payload = pm.run_sim_guided()
    elapsed = time.time() - t0

    if payload is None:
        sys.exit("run_sim_guided() returned nothing -- see the log above")

    test_f1 = payload["real_test_metrics"]["fbeta"]
    val_f1 = payload["real_val_metrics"]["fbeta"]
    payload.update({
        "arm": args.arm,
        "arm_note": ARM_NOTES[args.arm],
        "dataset_manifest": str(manifest),
        "n_train_movies": len(train),
        "elapsed_seconds": round(elapsed, 1),
        "python": platform.python_version(),
        "benchmarks": {"bt_sim_tuned_real_test_f1": 0.256,
                       "direct_real_test_f1": 0.847,
                       "t6_target_f1": 0.70},
        "t6_pass": bool(test_f1 >= 0.70),
    })
    Path(pm.RESULTS_FILE).write_text(json.dumps(payload, indent=2, default=str),
                                     encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"T6 RESULT -- arm '{args.arm}'")
    print("=" * 78)
    m = payload["real_test_metrics"]
    print(f"  real VAL  band F1 {val_f1:.4f}   (selection band)")
    print(f"  real TEST band F1 {test_f1:.4f}   R {m['recall']:.1%} P {m['precision']:.1%}")
    print(f"  vs BT-sim-tuned 0.256 | direct-real 0.847 | target 0.70")
    print(f"  T6: {'PASS' if test_f1 >= 0.70 else 'FAIL'}"
          f"   ({elapsed / 3600:.2f} h)")
    print(f"  results -> {pm.RESULTS_FILE}")
    return 0 if test_f1 >= 0.70 else 1


if __name__ == "__main__":
    sys.exit(main())
