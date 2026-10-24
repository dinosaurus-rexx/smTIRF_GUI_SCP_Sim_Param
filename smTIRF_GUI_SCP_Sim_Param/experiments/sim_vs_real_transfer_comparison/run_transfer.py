"""
How far does BachelorThesis-simulator pretraining carry?

Three protocols, one fixed held-out band (x >= 256), one number each. See
../../EXPERIMENT_SPEC_sim_transfer.md for the design and for why arm C exists.

    A   sim only        BT-sim-tuned params verbatim, no real labels consulted
    A2  sim + pixels    A's six shape params, threshold calibrated label-free
    B   sim -> real     BT-sim warm start, then Optuna on x < 170
    C   real only       cold start, otherwise identical to B  (the control)

Run one arm at a time:

    cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/experiments/sim_vs_real_transfer_comparison
    python -u run_transfer.py preflight
    python -u run_transfer.py A
    python -u run_transfer.py A2
    python -u run_transfer.py B
    python -u run_transfer.py C

B and C must not run concurrently unless each gets its own OUTPUT_DIR -- which
this driver does give them, so their per-trial detection JSONs cannot collide
with each other or overwrite the historical direct_run_results detections in
Parameter_opt/ (one of which, _scp_detected_direct_t22.json, is the pre-flight
witness for the scoring path). They are still run sequentially here because two
concurrent SCP searches would contend for CPU and distort the per-trial timings.

Every arm scores through pm.compute_metrics at MATCH_MARGIN = 3 -- the same
scorer behind every number on record. No second scorer is introduced anywhere.
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import numpy as np
import SCP_parametrizer_002 as pm

# --- shared configuration ---------------------------------------------------
pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]

TRAIN_BAND = (0, 170)      # Optuna objective, arms B and C
VAL_BAND = (170, 256)      # model selection, arms B and C
TEST_BAND = (256, 512)     # reported; never consulted by any arm
WHOLE_MOVIE = (0, 512)

N_TRIALS = 40              # matches the run that produced F1 0.800
TOP_K = 8                  # ditto (experiments/direct_real_spatial_crossval/run_direct.py)

BT_RESULTS = PROJ / "experiments" / "bt_simulator_threshold_calibration" / "optimization_results.json"
BT_DATASET = PROJ / "experiments" / "bt_simulator_threshold_calibration" / "dataset"

# The sim search's top-4 by sim F-beta. Arm A takes the first verbatim; arm B
# warm-starts from all four.
SIM_TOP4 = (25, 22, 11, 13)

# Enqueued warm-start threshold for arm B. The sim's own ~0.05 is outside
# PARAM_BOUNDS['intensity_threshold'] = (0.25, 0.5); Optuna clamps out-of-range
# enqueued values with a warning, which would silently collapse all four
# warm-start trials onto the same threshold. 0.30 is chosen deliberately: inside
# bounds, near the 0.323 the direct-on-real search settled on, and measured
# tractable (~160 s). intensity_threshold is precisely the parameter that does
# not transfer (SCP min-max normalises each frame, so it is a fraction of the
# frame's brightest object), so nothing is lost by not transferring it.
ARM_B_WARM_START_THRESHOLD = 0.30


# ============================================================================
# helpers
# ============================================================================

def sim_trial_params(number):
    """The BT sim search's parameters for one trial number, verbatim."""
    trials = json.loads(BT_RESULTS.read_text())['all_trials']
    for t in trials:
        if t['number'] == number:
            return dict(t['params']), t['values'][0]
    raise KeyError(f"trial {number} not in {BT_RESULTS}")


def real_labels():
    """(positives, negatives) for the real movie, as (x, y) tuples."""
    return pm.load_test_labels(pm.TEST_COORD_NPY, pm.TEST_CLASS_NPY)


def score_band(positives, negatives, detections, band):
    """pm.compute_metrics restricted to one x-band. The only scorer used here."""
    p = [s for s in positives if pm._in_band(s, band)]
    n = [s for s in negatives if pm._in_band(s, band)]
    d = [s for s in detections if pm._in_band(s, band)]
    return pm.compute_metrics(p, n, d)


def brief(m):
    return (f"F1={m['fbeta']:.4f} R={m['recall']:.2%} P={m['precision']:.2%} "
            f"TP={m['true_positives']} FP={m['false_positives']} "
            f"neg-hit={m['negatives_hit']} dets={m['detected_count']}")


def instrument_timing():
    """Wrap pm.run_detection so every detector call reports its wall-clock.

    Pre-flight item 4 of the spec: if trials start exceeding ~400 s the search
    has wandered to low thresholds and will not finish in the estimate.
    """
    original = pm.run_detection
    if getattr(original, '_timed', False):
        return
    times = []

    def timed(params, tif_path, output_json):
        t0 = time.time()
        ok, out = original(params, tif_path, output_json)
        dt = time.time() - t0
        times.append(dt)
        flag = "  <-- SLOW" if dt > 400 else ""
        print(f"    [detect {dt:6.1f}s | thr={params['intensity_threshold']:.3f} | "
              f"{'ok' if ok else 'FAILED'} | mean so far {np.mean(times):.0f}s]{flag}",
              flush=True)
        return ok, out

    timed._timed = True
    timed.times = times
    pm.run_detection = timed


def save(payload, filename):
    path = HERE / filename
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {path}")


# ============================================================================
# pre-flight
# ============================================================================

def preflight():
    """Cheap checks from spec section 6. Any failure voids every number below."""
    ok = True
    pos, neg = real_labels()

    print("=== 1. band spot counts ===")
    expect = {(0, 170): (87, 33), (170, 256): (45, 16),
              (256, 512): (111, 42), (0, 512): (243, 91)}
    for band, (ep, en) in expect.items():
        p = sum(pm._in_band(s, band) for s in pos)
        n = sum(pm._in_band(s, band) for s in neg)
        good = (p, n) == (ep, en)
        ok &= good
        print(f"  x{band}: targets {p}/{ep}, off-target {n}/{en}  "
              f"{'OK' if good else 'MISMATCH -- label file changed'}")

    print("\n=== 2. scoring path: re-score _scp_detected_direct_t22.json on x[340,512) ===")
    dets = pm.extract_coordinates(str(pm.OUTPUT_DIR / "_scp_detected_direct_t22.json"))
    m = score_band(pos, neg, dets, (340, 512))
    good = (m['true_positives'] == 74 and m['false_positives'] == 33
            and abs(m['recall'] - 74 / 78) < 1e-9
            and abs(m['precision'] - 74 / 107) < 1e-9
            and abs(m['fbeta'] - 0.8) < 1e-9)
    ok &= good
    print(f"  TP {m['true_positives']}/74  FP {m['false_positives']}/33  "
          f"R {m['recall']:.4%}/94.8718%  P {m['precision']:.4%}/69.1589%  "
          f"F1 {m['fbeta']:.4f}/0.8000")
    print(f"  {'OK -- scoring path intact end to end' if good else 'MISMATCH'}")

    print("\n=== 3. BT sim dataset ===")
    tifs = sorted(BT_DATASET.rglob("*.tif*"))
    good = len(tifs) == 5
    ok &= good
    print(f"  {len(tifs)}/5 TIFFs  {'OK' if good else 'MISSING'}")

    print("\n=== 4. module state ===")
    print(f"  PARAM_BOUNDS['intensity_threshold'] = {pm.PARAM_BOUNDS['intensity_threshold']} (exp (0.25, 0.5))")
    print(f"  active min_pulsation_frames bound = {pm.active_param_bounds()['min_pulsation_frames']} "
          f"(METHOD={pm.METHOD!r}; (3, 15) default, (3, 60) for direct_real -- the "
          f"track-length filter only discriminates on the real movie, see the note "
          f"on MIN_PULSATION_FRAMES_BY_METHOD)")
    print(f"  MATCH_MARGIN={pm.MATCH_MARGIN} SEED={pm.SEED} FBETA={pm.FBETA} "
          f"USE_BANDPASS={pm.USE_BANDPASS} bg_sigma={pm.BANDPASS_BG_SIGMA} "
          f"EVENT_NMS_RADIUS={pm.EVENT_NMS_RADIUS} "
          f"INTENSITY_APERTURE={pm.INTENSITY_APERTURE}"
          f"{' (legacy int(sigma*1.5) photometry)' if pm.INTENSITY_APERTURE == 0 else ''}")
    if pm.EVENT_NMS_RADIUS > 0:
        print("  NB: EVENT_NMS_RADIUS > 0 -- detections are duplicate-suppressed, so "
              "numbers from this run are NOT directly comparable to those recorded "
              "before it was added. Set it to 0.0 to reproduce those.")
    hist = ({'min_sigma': 0.5, 'max_sigma': 2.5, 'intensity_threshold': 0.30,
             'circular_threshold': 0.10, 'max_displacement': 3.0,
             'min_pulsation_frames': 3, 'intensity_variation_threshold': 0.05},
            {'min_sigma': 0.5, 'max_sigma': 2.5, 'intensity_threshold': 0.35,
             'circular_threshold': 0.30, 'max_displacement': 2.0,
             'min_pulsation_frames': 5, 'intensity_variation_threshold': 0.10},
            {'min_sigma': 0.7, 'max_sigma': 2.0, 'intensity_threshold': 0.40,
             'circular_threshold': 0.25, 'max_displacement': 2.0,
             'min_pulsation_frames': 4, 'intensity_variation_threshold': 0.08})
    # The historical three must still be present and unaltered -- that is what makes
    # a new arm C comparable to the recorded one in every respect except the
    # deliberate changes. A fourth seed was added with the (3, 60) track-length
    # ceiling, because all three historical seeds sit at min_pulsation_frames 3-5
    # and would leave the newly opened region unexplored early on.
    good = tuple(pm.DIRECT_WARM_START[:3]) == hist
    ok &= good
    print(f"  DIRECT_WARM_START[:3] == historical three configs: {good}  "
          f"{'OK' if good else 'REFACTOR REGRESSION'}")

    extra = list(pm.DIRECT_WARM_START[3:])
    expected_extra = [pm.DIRECT_WARM_START_TRACK_LENGTH_SEED]
    good = extra == expected_extra
    ok &= good
    print(f"  plus {len(extra)} track-length seed(s) at min_pulsation_frames="
          f"{[c['min_pulsation_frames'] for c in extra]}: {good}  "
          f"{'OK' if good else 'UNEXPECTED WARM START'}")

    # Every enqueued seed must be inside the space actually being searched, or
    # Optuna silently clamps it and the warm start is not what it claims to be.
    # DIRECT_WARM_START is only ever consumed by a direct_real search, so check it
    # against THAT method's bounds -- not against whatever METHOD happens to be set
    # at pre-flight time (the module default is a sim method, whose (3, 15) ceiling
    # would flag the track-length seed as clamped even though no run would clamp it).
    _saved_method = pm.METHOD
    try:
        pm.METHOD = "direct_real"
        _b = pm.active_param_bounds()
    finally:
        pm.METHOD = _saved_method
    bad = [(i, k, v) for i, c in enumerate(pm.DIRECT_WARM_START) for k, v in c.items()
           if not (_b[k][0] <= v <= _b[k][1])]
    good = not bad
    ok &= good
    print(f"  all warm-start values inside the direct_real bounds "
          f"(min_pulsation_frames {_b['min_pulsation_frames']}): {good}"
          + ("" if good else f"  WOULD BE CLAMPED: {bad}"))

    print("\n" + ("ALL PRE-FLIGHT CHECKS PASSED" if ok else "PRE-FLIGHT FAILED -- DO NOT LAUNCH"))
    return ok


# ============================================================================
# arm A -- sim only
# ============================================================================

def arm_A():
    """
    Selection uses the sim score and nothing else, so it picks trial 25 -- the
    top sim scorer. Its parameters go to the real movie verbatim, INCLUDING its
    own intensity_threshold of ~0.051. No real label is consulted anywhere, so
    the whole movie is a legitimate test set for this arm as well as x >= 256.

    This is expected to time out: on this movie threshold 0.05 exceeded 300 s and
    0.12 exceeded 420 s. The timeout is raised to 1800 s so the outcome is a
    measurement rather than an inference. A timeout IS the result and is recorded
    as one -- no workable threshold is substituted.
    """
    params, sim_f1 = sim_trial_params(25)
    print("=" * 78)
    print("ARM A -- sim only. Trial 25 verbatim (sim F1 = %.4f)" % sim_f1)
    print("=" * 78)
    for k, v in params.items():
        print(f"  {k}: {v}")

    pm.DETECTION_TIMEOUT = 1800
    print(f"\nDETECTION_TIMEOUT = {pm.DETECTION_TIMEOUT}s (raised for this arm only)")

    out_json = str(HERE / "_armA_detections.json")
    t0 = time.time()
    success, output = pm.run_detection(params, pm.TEST_TIF, out_json)
    elapsed = time.time() - t0
    print(f"\nDetector returned after {elapsed:.1f}s  (success={success})")

    payload = {
        'arm': 'A', 'description': 'sim only -- BT-sim trial 25 verbatim, no real labels',
        'source_trial': 25, 'sim_fbeta': sim_f1, 'parameters': params,
        'detection_timeout_s': pm.DETECTION_TIMEOUT,
        'wall_clock_s': round(elapsed, 1),
        'timed_out': (not success and output == 'timeout'),
        'success': success,
        'match_margin': pm.MATCH_MARGIN, 'use_bandpass': pm.USE_BANDPASS,
        'event_nms_radius': pm.EVENT_NMS_RADIUS,
    }

    if not success:
        reason = 'timeout' if output == 'timeout' else 'detector failed'
        print(f"\nRESULT: {reason} after {elapsed:.1f}s.")
        print("Pure transfer is not merely inaccurate here, it is intractable on this")
        print("movie: the simulator's usable threshold range (<=0.10) and the real")
        print("movie's (>=0.20) do not overlap. Reported as a timeout, not replaced")
        print("with a workable threshold.")
        payload['result'] = reason
        payload['detail'] = output if output != 'timeout' else None
    else:
        pos, neg = real_labels()
        dets = pm.extract_coordinates(out_json)
        print(f"\n{len(dets)} detections")
        payload['result'] = 'completed'
        for name, band in (('test_x256_512', TEST_BAND), ('whole_movie', WHOLE_MOVIE)):
            m = score_band(pos, neg, dets, band)
            payload[name] = m
            print(f"  {name:16s} x{band}: {brief(m)}")
        print("\nThe whole-movie number is directly comparable to the thesis's Table A.8")
        print("(scored there on all 243 targets) -- though the thesis used Hungarian")
        print("matching at threshold 2, not this greedy matcher at margin 3.")

    save(payload, "armA_sim_only.json")
    return payload


# ============================================================================
# arm A2 -- label-free threshold calibration
# ============================================================================

def _blob_count(frame, params, bandpass):
    """One frame through SCP's exact blob-detection front end. No tracking."""
    from scipy import ndimage
    from skimage.feature import blob_log
    f = ndimage.gaussian_filter(frame.astype(np.float64), sigma=0.5)   # preprocess_frame
    if bandpass:
        bg = ndimage.gaussian_filter(f, pm.BANDPASS_BG_SIGMA)
        f = np.clip(f - bg, 0, None)
    f = (f - f.min()) / (f.max() - f.min() + 1e-10)                    # min-max per frame
    blobs = blob_log(f, min_sigma=params['min_sigma'], max_sigma=params['max_sigma'],
                     threshold=params['intensity_threshold'], overlap=0.5, log_scale=False)
    return len(blobs)


def _mean_density(tif, params, n_frames=10):
    """Mean blobs per 1000 px over evenly sampled frames.

    Density, not raw count: the sim movies are 256x256 and the real movie is
    256x512, so equal blobs-per-frame would mean half the areal density. Matching
    density is the faithful version of the spec's blobs-per-frame criterion.
    """
    import tifffile
    with tifffile.TiffFile(tif) as tf:
        n = tf.series[0].shape[0]
        idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)
        counts, px = [], None
        for i in idx:
            frame = np.asarray(tf.asarray(key=int(i)))
            px = frame.size
            counts.append(_blob_count(frame, params, pm.USE_BANDPASS))
    return float(np.mean(counts)) / (px / 1000.0), float(np.mean(counts)), px


def arm_A2():
    """
    Keep trial 25's six shape parameters; set intensity_threshold from the real
    movie's PIXELS but not its LABELS -- the protocol that would actually be
    available on a new recording.

    Criterion: pick the threshold whose mean blob density on the real movie
    matches the mean blob density the sim produced at its own chosen threshold.
    This is unsupervised calibration, and is reported separately from arm A
    proper; it is not a sim-only number and not a supervised one either.
    """
    params, sim_f1 = sim_trial_params(25)
    print("=" * 78)
    print("ARM A2 -- trial 25 shape params + label-free threshold calibration")
    print("=" * 78)

    # --- 1. sim reference density at the sim's own chosen threshold -----------
    sim_tifs = sorted((BT_DATASET / "train").glob("*.tif*"))
    print(f"\n[1] sim reference density at threshold {params['intensity_threshold']:.4f}")
    ref = []
    for t in sim_tifs:
        d, c, px = _mean_density(str(t), params)
        ref.append(d)
        print(f"    {t.name}: {c:.1f} blobs/frame over {px} px -> {d:.3f} per 1000 px")
    target = float(np.mean(ref))
    print(f"    target density = {target:.3f} blobs per 1000 px")

    # --- 2. bracket then bisect on the real movie ----------------------------
    # Blob count falls monotonically with threshold, so walk down from a high,
    # cheap threshold and stop at the first one that overshoots the target. That
    # avoids ever evaluating a pathologically low threshold on the real movie.
    print(f"\n[2] real-movie density sweep (sigma {params['min_sigma']:.2f}-{params['max_sigma']:.2f}, "
          f"bandpass={pm.USE_BANDPASS})")
    probe = dict(params)
    curve = []

    def density_at(thr):
        probe['intensity_threshold'] = thr
        d, c, px = _mean_density(pm.TEST_TIF, probe)
        curve.append({'threshold': round(thr, 5), 'density_per_1000px': round(d, 4),
                      'blobs_per_frame': round(c, 2)})
        print(f"    thr {thr:.4f} -> {c:8.1f} blobs/frame = {d:7.3f} per 1000 px")
        return d

    # Walk down from a high, cheap threshold. The grid reaches below the sim's own
    # 0.05 because it has to: measured beforehand, the real movie's blob density at
    # trial 25's sigmas and threshold 0.0505 (1.45 per 1000 px) is already just
    # BELOW the sim's (1.59), so the crossing point is lower still. A grid that
    # stopped at 0.10 would bottom out without bracketing and silently return its
    # floor. Tractability is not a concern down here at these sigmas -- the narrow
    # high-sigma window holds the count near 200 blobs/frame, which is what let arm
    # A finish in 202 s.
    coarse = [0.60, 0.50, 0.40, 0.32, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.035, 0.02]
    hi, lo = None, None          # hi: density below target, lo: density above
    d_hi = d_lo = None
    for thr in coarse:
        d = density_at(thr)
        if d < target:
            hi, d_hi = thr, d
        else:
            lo, d_lo = thr, d
            break

    if lo is None:
        chosen, note = coarse[-1], ("density never reached the sim's even at the lowest "
                                    "probed threshold -- calibration hit its floor")
    elif hi is None:
        chosen, note = coarse[0], ("density already exceeded the sim's at the highest "
                                   "probed threshold -- calibration hit its ceiling")
    else:
        print(f"    bracketed: [{lo:.4f} (d={d_lo:.3f}) , {hi:.4f} (d={d_hi:.3f})], target {target:.3f}")
        for _ in range(6):
            mid = 0.5 * (lo + hi)
            d = density_at(mid)
            if d < target:
                hi, d_hi = mid, d
            else:
                lo, d_lo = mid, d
        chosen = lo if abs(d_lo - target) <= abs(d_hi - target) else hi
        d_chosen = d_lo if chosen == lo else d_hi
        note = (f"bisected to match the sim's blob density; achieved "
                f"{d_chosen:.3f} vs target {target:.3f} "
                f"({abs(d_chosen - target) / target:.1%} off)")
    print(f"\n    chosen threshold = {chosen:.4f}  ({note})")

    # --- 3. full SCP run at the calibrated threshold -------------------------
    final = dict(params)
    final['intensity_threshold'] = float(chosen)
    pm.DETECTION_TIMEOUT = 1800
    print(f"\n[3] full SCP run at the calibrated threshold (timeout {pm.DETECTION_TIMEOUT}s)")
    out_json = str(HERE / "_armA2_detections.json")
    t0 = time.time()
    success, output = pm.run_detection(final, pm.TEST_TIF, out_json)
    elapsed = time.time() - t0
    print(f"    detector returned after {elapsed:.1f}s (success={success})")

    payload = {
        'arm': 'A2',
        'description': 'sim shape params + intensity_threshold calibrated on real '
                       'PIXELS only (unsupervised); distinct from arm A proper',
        'source_trial': 25, 'sim_fbeta': sim_f1,
        'shape_params_from_sim': {k: v for k, v in params.items()
                                  if k != 'intensity_threshold'},
        'sim_threshold': params['intensity_threshold'],
        'calibration': {'criterion': 'match mean blob density (blobs per 1000 px) of '
                                     'the sim at its own threshold',
                        'note': note,
                        'target_density_per_1000px': round(target, 4),
                        'sim_movies': [t.name for t in sim_tifs],
                        'frames_sampled_per_movie': 10,
                        'sweep': curve},
        'calibrated_threshold': float(chosen),
        'parameters': final,
        'wall_clock_s': round(elapsed, 1),
        'timed_out': (not success and output == 'timeout'),
        'success': success,
        'match_margin': pm.MATCH_MARGIN, 'use_bandpass': pm.USE_BANDPASS,
        'event_nms_radius': pm.EVENT_NMS_RADIUS,
    }

    if not success:
        payload['result'] = 'timeout' if output == 'timeout' else 'detector failed'
        print(f"\nRESULT: {payload['result']}")
    else:
        pos, neg = real_labels()
        dets = pm.extract_coordinates(out_json)
        payload['result'] = 'completed'
        print(f"\n{len(dets)} detections")
        for name, band in (('test_x256_512', TEST_BAND), ('whole_movie', WHOLE_MOVIE)):
            m = score_band(pos, neg, dets, band)
            payload[name] = m
            print(f"  {name:16s} x{band}: {brief(m)}")

    save(payload, "armA2_labelfree_threshold.json")
    return payload


# ============================================================================
# arms B and C -- 40-trial Optuna searches on x < 170, selected on 170-256
# ============================================================================

def _configure_search(tag):
    """Everything B and C share. They must search an identical space."""
    pm.METHOD = "direct_real"
    pm.N_TRIALS = N_TRIALS
    pm.TOP_K = TOP_K
    pm.SEED = 42
    pm.DIRECT_TRAIN_XBAND = TRAIN_BAND
    pm.DIRECT_VAL_XBAND = VAL_BAND
    pm.DIRECT_TEST_XBAND = TEST_BAND
    pm.DETECTION_TIMEOUT = 600
    # Per-arm detection directory: run_direct_real names its per-trial output by
    # trial number only, so a shared directory would have B and C overwrite each
    # other and clobber the historical direct_run_results detections.
    out = HERE / f"detections_arm{tag}"
    out.mkdir(exist_ok=True)
    pm.OUTPUT_DIR = out
    instrument_timing()
    print(f"  bands: train x{TRAIN_BAND} | val x{VAL_BAND} | test x{TEST_BAND}")
    print(f"  N_TRIALS={pm.N_TRIALS} TOP_K={pm.TOP_K} SEED={pm.SEED} "
          f"MATCH_MARGIN={pm.MATCH_MARGIN} timeout={pm.DETECTION_TIMEOUT}s")
    print(f"  PARAM_BOUNDS = {json.dumps({k: list(v) for k, v in pm.active_param_bounds().items()})}")
    print(f"  detections -> {out}")


def arm_B():
    """
    Sim -> continue on half the movie. Identical to arm C except that the four
    enqueued warm-start configs are the sim's top-4 rather than the module's
    three recall-preserving seeds. PARAM_BOUNDS is NOT widened: B and C must
    search the same space or the comparison means nothing.
    """
    print("=" * 78)
    print("ARM B -- BT sim warm start, then Optuna on x < 170")
    print("=" * 78)
    _configure_search("B")

    warm = []
    for n in SIM_TOP4:
        p, f1 = sim_trial_params(n)
        p['intensity_threshold'] = ARM_B_WARM_START_THRESHOLD
        warm.append(p)
        print(f"  warm start <- sim trial {n:2d} (sim F1 {f1:.4f}), "
              f"threshold set to {ARM_B_WARM_START_THRESHOLD} deliberately "
              f"(sim's own ~0.05 is outside bounds and would be silently clamped)")
    # Sanity: every enqueued value must be inside bounds, or Optuna clamps it and
    # the warm start is not the warm start we think it is.
    _bounds = pm.active_param_bounds()   # METHOD is set by _configure_search above
    for p in warm:
        for k, v in p.items():
            lo, hi = _bounds[k]
            assert lo <= v <= hi, f"enqueued {k}={v} outside bounds {(lo, hi)}"
    pm.DIRECT_WARM_START = tuple(warm)
    pm.RESULTS_FILE = str(HERE / "armB_sim_then_real.json")

    t0 = time.time()
    payload = pm.run_direct_real()
    print(f"\nARM B wall clock: {(time.time() - t0) / 60:.1f} min")
    return payload


def arm_C():
    """
    Real only, cold start -- the control. Identical to B in every respect except
    the warm-start configs, which stay at the module default (the three
    recall-preserving seeds). Without this arm a good B number cannot be
    attributed to the simulator rather than to the half-movie tuning.
    """
    print("=" * 78)
    print("ARM C -- real only, cold start (control)")
    print("=" * 78)
    _configure_search("C")
    print(f"  warm start <- module default DIRECT_WARM_START "
          f"({len(pm.DIRECT_WARM_START)} recall-preserving seeds, unchanged)")
    pm.RESULTS_FILE = str(HERE / "armC_real_only.json")

    t0 = time.time()
    payload = pm.run_direct_real()
    print(f"\nARM C wall clock: {(time.time() - t0) / 60:.1f} min")
    return payload


# ============================================================================

ARMS = {'preflight': preflight, 'A': arm_A, 'A2': arm_A2, 'B': arm_B, 'C': arm_C}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        print(f"usage: python -u run_transfer.py [{' | '.join(ARMS)}]")
        sys.exit(2)
    name = sys.argv[1]
    print(f"### {name} ###", flush=True)
    result = ARMS[name]()
    if name == 'preflight' and result is not True:
        sys.exit(1)
    print(f"### {name} DONE ###")
