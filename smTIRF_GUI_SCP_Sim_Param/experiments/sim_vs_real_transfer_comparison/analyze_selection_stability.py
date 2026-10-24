"""
Is arm B's 0.803 vs arm C's 0.770 a real difference?

The spec's rule of thumb was "with 40 trials and a single seed, differences below
~0.02 F1 are not interpretable". The measured gap is 0.033, nominally above that --
so the rule of thumb alone would license claiming a simulator effect. This script
replaces the rule of thumb with two measurements from the experiment's own data.

1. SELECTION STABILITY. Each arm picks one config from its top-8-by-TRAIN using the
   45-target VAL band. If configs that VAL cannot separate have widely different TEST
   scores, the reported number is a draw from that spread rather than a property of
   the protocol.

2. PAIRED BOOTSTRAP on the 111 test-band targets, to put an interval on B - C.

Run:
    python -u analyze_selection_stability.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))
sys.path.insert(0, str(HERE))

import SCP_parametrizer_002 as pm
from run_transfer import TEST_BAND, VAL_BAND, real_labels

ARMS = {'B': 'armB_sim_then_real.json', 'C': 'armC_real_only.json'}
SEED = 12345
N_BOOT = 20000


def band_outcomes(dets, positives, negatives, band):
    """Instance-level outcomes in one band: which targets were hit, and how many FPs."""
    p = [s for s in positives if pm._in_band(s, band)]
    n = [s for s in negatives if pm._in_band(s, band)]
    d = [s for s in dets if pm._in_band(s, band)]
    matched, unmatched_p, unmatched_d = pm.match_detections_to_positives(
        p, d, margin=pm.MATCH_MARGIN)
    # Identify hit targets by coordinate. Safe only if annotated positives are
    # distinct, which is asserted rather than assumed -- a duplicate would silently
    # mark two targets hit off one match.
    assert len(set(map(tuple, p))) == len(p), "duplicate annotated positives in band"
    hit = {tuple(m['positive']) for m in matched}
    per_target = np.array([1 if tuple(s) in hit else 0 for s in p], dtype=int)
    assert per_target.sum() == len(matched)
    return per_target, len(unmatched_d), len(p), len(n)


def f1_from(tp, fp, fn):
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def load_arm(tag):
    payload = json.loads((HERE / ARMS[tag]).read_text())
    det_dir = HERE / f"detections_arm{tag}"
    return payload, det_dir


def main():
    pos, neg = real_labels()
    print("=" * 78)
    print("1. SELECTION STABILITY -- what the 45-target VAL band can actually separate")
    print("=" * 78)

    selected = {}
    for tag in ('B', 'C'):
        payload, det_dir = load_arm(tag)
        trials = [t for t in payload['all_trials'] if t['train_fbeta'] is not None]
        trials.sort(key=lambda t: -t['train_fbeta'])
        pool = trials[:payload['top_k']]
        pool.sort(key=lambda t: -(t['val_fbeta'] or 0))
        best_val = pool[0]['val_fbeta']
        print(f"\nARM {tag}: selection pool = top {payload['top_k']} by TRAIN, ranked by VAL")
        print(f"  {'trial':>6} {'train':>7} {'val':>7} {'test':>7}   val gap to winner")
        for t in pool:
            gap = best_val - (t['val_fbeta'] or 0)
            mark = "  <-- SELECTED" if t['number'] == payload['final_trial_number'] else ""
            print(f"  {t['number']:>6} {t['train_fbeta']:>7.4f} {t['val_fbeta']:>7.4f} "
                  f"{t['test_fbeta']:>7.4f}   {gap:+.4f}{mark}")
        selected[tag] = payload['final_trial_number']

        # How many val-band spots separate the top two? F1 differences on 45 targets
        # are quantised; a "gap" of a few thousandths can be a single detection.
        near = [t for t in pool if best_val - (t['val_fbeta'] or 0) <= 0.02]
        tests = [t['test_fbeta'] for t in near]
        print(f"  configs within 0.02 VAL of the winner: {len(near)} "
              f"(trials {[t['number'] for t in near]})")
        print(f"  their TEST F1 spread: {min(tests):.4f} - {max(tests):.4f} "
              f"(range {max(tests) - min(tests):.4f})")

        for t in near[:2]:
            dets = pm.extract_coordinates(str(det_dir / f"_scp_detected_direct_t{t['number']}.json"))
            pt, fp, npos, _ = band_outcomes(dets, pos, neg, VAL_BAND)
            print(f"    trial {t['number']:>2} VAL counts: TP={pt.sum()} FN={npos - pt.sum()} FP={fp}")

    print("\n" + "=" * 78)
    print("2. PAIRED BOOTSTRAP on the 111 test-band targets")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    out = {}
    for tag in ('B', 'C'):
        payload, det_dir = load_arm(tag)
        dets = pm.extract_coordinates(
            str(det_dir / f"_scp_detected_direct_t{selected[tag]}.json"))
        per_target, fp, npos, nneg = band_outcomes(dets, pos, neg, TEST_BAND)
        out[tag] = (per_target, fp)
        print(f"  arm {tag} trial {selected[tag]}: TP={per_target.sum()} "
              f"FN={npos - per_target.sum()} FP={fp} over {npos} targets "
              f"-> F1={f1_from(per_target.sum(), fp, npos - per_target.sum()):.4f}")

    nb, nc = len(out['B'][0]), len(out['C'][0])
    assert nb == nc, "arms scored different target counts -- not comparable"
    diffs, fb, fc = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, nb, nb)                    # paired: same targets both arms
        vals = []
        for tag in ('B', 'C'):
            per_target, fp = out[tag]
            tp = int(per_target[idx].sum())
            fn = nb - tp
            fp_b = int(rng.binomial(fp, 1.0) if fp == 0 else rng.poisson(fp))
            vals.append(f1_from(tp, fp_b, fn))
        fb.append(vals[0]); fc.append(vals[1]); diffs.append(vals[0] - vals[1])
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n  B F1  95% CI: [{np.percentile(fb, 2.5):.4f}, {np.percentile(fb, 97.5):.4f}]")
    print(f"  C F1  95% CI: [{np.percentile(fc, 2.5):.4f}, {np.percentile(fc, 97.5):.4f}]")
    print(f"  B - C = {np.mean(diffs):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(B>C) = {(diffs > 0).mean():.3f}")
    print(f"  {'CI excludes zero' if lo > 0 or hi < 0 else 'CI INCLUDES ZERO -- difference not resolved'}")
    print("\n  (Bootstrap over spots within one movie. It bounds sampling noise on this")
    print("   field of view only -- it says nothing about a second recording, and it")
    print("   does not include the selection variance measured in part 1, which is")
    print("   larger.)")


if __name__ == "__main__":
    main()
