"""
The control gate T6 is missing: does the SIMULATOR help, or does the real-labelled
selection step do all the work?

Every configuration with a real-movie score on record was first chosen by its
simulated score, so the existing results cannot separate those two contributions.
This draws configurations at RANDOM from the same search space, with no simulator
involved at all, and puts them through the identical second half of the protocol:

    run on the real movie -> score the validation band (x 0-256) -> pick the winner
    -> report the held-out test band (x 256-512)

Same bounds (`active_param_bounds`), same scorer (`compute_metrics`, MATCH_MARGIN=3),
same detector switches, same bands, same number of candidates as TOP_K. The only
thing removed is the simulator.

Interpretation:
  * random ~ 0.71  -> the simulator's candidate pool is no better than random draws;
                      the real-label selection step is doing the work, and the
                      simulator is not earning its keep for parameter tuning.
  * random << 0.71 -> the simulator genuinely concentrates the search on a useful
                      region, and 0.7143 reflects that.

Comparison numbers, all on the same band x 256-512:
    0.256   tuned on the old BT simulator
    0.559   this simulator, picking by SIMULATED score alone (no real labels)
    0.7143  this simulator, sim search + real-val selection   <- what this tests
    0.8468  tuned directly on the real movie, hand-improved

    python run_random_control.py --candidates 6 --seed 0
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
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import numpy as np                                  # noqa: E402
import SCP_parametrizer_002 as pm                   # noqa: E402


def draw(rng, bounds):
    """One uniform draw from the search space, honouring max_sigma > min_sigma.

    Mirrors suggest_params() exactly, but with a uniform sampler instead of TPE --
    so the only difference from the real protocol is where candidates come from.
    """
    lo, hi = bounds["min_sigma"]
    min_sigma = rng.uniform(lo, hi)
    ms_lo, ms_hi = bounds["max_sigma"]
    max_sigma = rng.uniform(max(ms_lo, min_sigma + 0.1), ms_hi)
    return {
        "min_sigma": min_sigma,
        "max_sigma": max_sigma,
        "intensity_threshold": rng.uniform(*bounds["intensity_threshold"]),
        "circular_threshold": rng.uniform(*bounds["circular_threshold"]),
        "max_displacement": rng.uniform(*bounds["max_displacement"]),
        "min_pulsation_frames": int(rng.integers(bounds["min_pulsation_frames"][0],
                                                 bounds["min_pulsation_frames"][1] + 1)),
        "intensity_variation_threshold": rng.uniform(*bounds["intensity_variation_threshold"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=int, default=6, help="match TOP_K")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
    pm.PYTHON_CMD = [sys.executable]
    pm.METHOD = "sim_guided_real"        # same bounds the sim-guided search used
    pm.DETECTION_TIMEOUT = args.timeout

    test_movie = pm.load_test_movie()
    if test_movie is None:
        sys.exit("real movie/labels not found")
    pos, neg = test_movie["positive_spots"], test_movie["negative_spots"]
    bounds = pm.active_param_bounds()
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("RANDOM-SEARCH CONTROL -- no simulator, real-val selection only")
    print("=" * 78)
    print(f"  candidates : {args.candidates} (uniform draws, seed {args.seed})")
    print(f"  bounds     : {bounds}")
    print(f"  bands      : val x{pm.REAL_VAL_XBAND}  test x{pm.REAL_TEST_XBAND}")
    print(f"  scorer     : compute_metrics, MATCH_MARGIN={pm.MATCH_MARGIN} (untouched)")

    t0 = time.time()
    scored = []
    for i in range(args.candidates):
        p = draw(rng, bounds)
        out_json = str(HERE / f"_random_cand{i}.json")
        print(f"\n[{i + 1}/{args.candidates}] "
              + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in p.items()))
        bands = pm.evaluate_real_bands(p, pos, neg, out_json, verbose=True)
        if bands.get("error"):
            print(f"    FAILED: {bands['error']}")
            continue
        scored.append({"index": i, "params": p,
                       "val": bands["val"], "test": bands["test"]})

    if not scored:
        sys.exit("every random candidate failed to produce detections")

    best = max(scored, key=lambda s: s["val"]["fbeta"])
    elapsed = time.time() - t0

    print("\n" + "=" * 78)
    print("RANDOM-SEARCH CONTROL RESULT")
    print("=" * 78)
    print(f"  {'cand':>4} {'VAL F1':>9} {'TEST F1':>9} {'TEST R':>8} {'TEST P':>8}")
    for s in scored:
        mark = "  <- selected on VAL" if s is best else ""
        print(f"  {s['index']:>4} {s['val']['fbeta']:9.4f} {s['test']['fbeta']:9.4f} "
              f"{100 * s['test']['recall']:7.1f}% {100 * s['test']['precision']:7.1f}%{mark}")
    t = best["test"]
    print(f"\n  RANDOM + real-val selection -> held-out TEST F1 {t['fbeta']:.4f} "
          f"(R {t['recall']:.1%} P {t['precision']:.1%})")
    print(f"  sim-guided + real-val selection    0.7143")
    print(f"  sim score alone, no real labels    0.5591")
    print(f"  old BT simulator                   0.2560")
    print(f"  ({elapsed / 60:.1f} min)")
    verdict = ("the simulator's candidate pool is NO BETTER than random draws"
               if t["fbeta"] >= 0.68 else
               "the simulator's candidate pool IS better than random draws")
    print(f"\n  Reading: {verdict}.")

    payload = {
        "experiment": "random_search_control",
        "note": "no simulator; uniform draws from the same bounds, real-val selection",
        "n_candidates": args.candidates, "seed": args.seed,
        "bounds": {k: list(v) for k, v in bounds.items()},
        "val_band": list(pm.REAL_VAL_XBAND), "test_band": list(pm.REAL_TEST_XBAND),
        "candidates": scored, "selected": best,
        "real_test_fbeta": t["fbeta"],
        "benchmarks": {"sim_guided": 0.7143, "sim_score_only": 0.5591,
                       "bt_sim": 0.256, "direct_real_handtuned": 0.8468},
        "elapsed_seconds": round(elapsed, 1),
        "python": platform.python_version(),
    }
    p = HERE / "random_control_results.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"  results -> {p}")


if __name__ == "__main__":
    main()
