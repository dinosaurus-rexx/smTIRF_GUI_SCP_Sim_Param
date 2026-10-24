"""
Recover arm B's trial 39, the one trial lost to a machine sleep.

Arm B ran overnight and the laptop slept during trial 39. subprocess timeouts are
wall-clock, so the detector was killed at the 600 s limit after 44380 s of elapsed
time and the trial was scored 0.0. Every other B trial ran 93-241 s (arm C, run
uninterrupted, ran 100-416 s), so exactly one trial is affected.

The loss cannot have changed arm B's outcome -- trial 39 is the last trial, and the
winner (trial 34, train F1 0.8144) was already fixed -- but leaving a spuriously
zeroed trial in the record would misrepresent the search. This re-runs it with the
identical parameters and scores it on the same bands with the same scorer.

Run:
    python -u recover_armB_trial39.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))
sys.path.insert(0, str(HERE))

import SCP_parametrizer_002 as pm
from run_transfer import TRAIN_BAND, VAL_BAND, TEST_BAND, real_labels, score_band, brief

pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]
pm.OUTPUT_DIR = HERE / "detections_armB"
pm.DETECTION_TIMEOUT = 1800          # generous: the point is to get a number at all

RESULTS = HERE / "armB_sim_then_real.json"
TRIAL = 39

if __name__ == "__main__":
    payload = json.loads(RESULTS.read_text())
    entry = next(t for t in payload['all_trials'] if t['number'] == TRIAL)
    params = dict(entry['params'])
    print(f"re-running arm B trial {TRIAL}: {params}")
    assert entry['train_fbeta'] == 0.0, "trial 39 is not the zeroed one -- check the file"

    out_json = str(pm.OUTPUT_DIR / f"_scp_detected_direct_t{TRIAL}.json")
    t0 = time.time()
    ok, _ = pm.run_detection(params, pm.TEST_TIF, out_json)
    dt = time.time() - t0
    print(f"detector returned after {dt:.1f}s (success={ok})")
    if not ok:
        print("still failing -- the trial is genuinely intractable, not a sleep artifact")
        sys.exit(1)

    pos, neg = real_labels()
    dets = pm.extract_coordinates(out_json)
    recovered = {'wall_clock_s': round(dt, 1)}
    for name, band in (('train', TRAIN_BAND), ('val', VAL_BAND), ('test', TEST_BAND)):
        m = score_band(pos, neg, dets, band)
        recovered[name] = m
        print(f"  {name:5s} x{band}: {brief(m)}")

    best_train = max(t['train_fbeta'] for t in payload['all_trials']
                     if t['train_fbeta'] is not None)
    would_change = recovered['train']['fbeta'] > best_train
    print(f"\nrecovered train F1 {recovered['train']['fbeta']:.4f} vs arm B's winning "
          f"{best_train:.4f} -> selection {'CHANGES' if would_change else 'unaffected'}")

    # Recorded alongside, not merged into all_trials: the search really did see a 0.0
    # for this trial, and rewriting that would misrepresent what TPE was told.
    payload['recovered_trial_39'] = {
        'reason': 'original run killed by the 600s wall-clock timeout during a machine '
                  'sleep (44380s elapsed); re-run afterwards with identical parameters',
        'params': params,
        'metrics': recovered,
        'changes_selection': bool(would_change),
        'note': 'the search itself was scored with the 0.0; this is the honest value the '
                'trial would have contributed had the machine stayed awake',
    }
    RESULTS.write_text(json.dumps(payload, indent=2))
    print(f"\nwritten to {RESULTS}")
