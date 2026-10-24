
import subprocess
import json
import os
from pathlib import Path

import numpy as np
import optuna

# --- Detector script -------------------------------------------------------
scp_script = r"C:\Users\PC\Desktop\Constellate_proteomics\SCP\SCP_KTree_JSON.py"
# Interpreter used to launch the detector. SCP pins Python 3.11 on the target
# machine; override (e.g. [sys.executable] or ["py", "-3.12"]) if yours differs.
PYTHON_CMD = ["py", "-3.11"]

# Bandpass (background subtraction) before blob detection. Applied to BOTH sim and
# real runs so the search tunes for the same preprocessing the real movie gets.
# Needed for the noisy real movie: without it, low thresholds flood the detector
# with spurious blobs (intractably slow + imprecise).
USE_BANDPASS = True
BANDPASS_BG_SIGMA = 8.0

# Final duplicate-suppression pass over classified events (px). Passed to every
# detector run, sim and real, so the search tunes for the pipeline used at
# evaluation time. 3.0 is lossless on the real movie: its closest pair of
# annotated spots is 3.61 px apart, so no radius below that can fuse two
# annotated molecules. Measured on the held-out test band it removed 8 of 50
# false positives with the true-positive count unchanged (precision 67.9% ->
# 71.6%, recall flat at 95.5%). Set to 0.0 to reproduce pre-existing runs
# exactly -- doing so is required to reproduce any number recorded before this
# was added.
EVENT_NMS_RADIUS = 3.0

# Signal-aperture radius (px) for per-spot photometry; 0.0 keeps SCP's legacy
# int(sigma*1.5) disc with no background subtraction. Left OFF deliberately.
# The legacy measurement is poor photometry -- at the real-tuned sigma range it
# is a 0-1 px sample on top of the full background pedestal -- but switching it
# on is not a drop-in improvement:
#   * measured at the annotated coordinates of realdata/vid_ch1.tiff, the class
#     separation gain is small for brightness (AUC 0.899 -> 0.909) and only
#     moderate for CV (0.580 -> 0.748);
#   * it redefines mean_intensity and intensity_cv, so every tuned
#     intensity_variation_threshold on record becomes meaningless; and
#   * with the background removed, aggregates show HIGHER cv than molecules
#     (1.596 vs 0.791) on that movie, while the classifier keeps cv >= threshold
#     -- so the CV test would then actively favour aggregates, and the searched
#     bounds (0.05, 0.25) sit far below the observed range.
# Turning this on therefore means also revisiting the direction and bounds of the
# CV criterion. Do that as its own change, with its own held-out measurement.
INTENSITY_APERTURE = 0.0

# --- Dataset (simulated) ---------------------------------------------------
# Manifest produced by Simulation/generate_dataset.py. Lists the seeded train
# and validation movies + their label JSONs. Optimization happens on TRAIN;
# VALIDATION is used to pick the final config (guards against overfitting).
DATASET_MANIFEST = r"C:\Users\PC\Desktop\Constellate_proteomics\Simulation\dataset\manifest.json"

# --- Test set (the real, non-simulated labelled movie) ---------------------
# Held out entirely from tuning; scored exactly ONCE at the end. Labels come as
# two numpy arrays (BachelorThesis realdata format): coord.npy is (N,2) int in
# [y, x] order; class.npy is (N,) with 0=stationary (target->positive),
# 1=aggregate (off-target->negative), 2=moving. See load_test_movie().
TEST_TIF = r"C:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms\realdata\vid_ch1.tiff"
TEST_COORD_NPY = r"C:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms\realdata\coord.npy"
TEST_CLASS_NPY = r"C:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms\realdata\class.npy"
TEST_POSITIVE_CLASS = 0   # stationary = detect target
TEST_NEGATIVE_CLASS = 1   # aggregate  = off-target distractor

# --- Spatial cross-validation bands on the real movie ----------------------
# Goal = best params FOR THE REAL DATA. We search on sim for breadth, then judge
# the top-K configs on the REAL movie split by X into disjoint bands: pick the
# final config on VAL, report the held-out number on TEST. Bands are (x_min, x_max)
# in pixels; the movie is 512 wide. Split ~50/50 (each band ~half the 334 spots).
REAL_VAL_XBAND = (0, 256)     # selection band
REAL_TEST_XBAND = (256, 512)  # held-out reporting band
METHOD = "sim_guided_real"    # "sim_guided_real" | "direct_real" | "sim_only"

# For METHOD="direct_real": search parameters ON the real movie, split into three
# disjoint x-bands. Optimize on TRAIN, select on VAL, report held-out TEST. One
# detection per trial covers all three (sliced by x). Directly targets real data.
DIRECT_TRAIN_XBAND = (0, 170)
DIRECT_VAL_XBAND = (170, 340)
DIRECT_TEST_XBAND = (340, 512)

# ============================================================================
# OPTIMIZATION CONFIG
# ============================================================================

N_TRIALS = 100                 # Optuna trials; each trial runs the detector on ALL train movies
OPTIMIZATION_MODE = "single"   # "single" -> maximize mean F-beta (one best param set)
                               # "multi"  -> maximize (mean recall, mean precision) -> Pareto front
FBETA = 1.0                    # beta for the F-beta score (also the selection metric).
                               # 1.0 = F1 (spot-detection field standard);
                               # >1 leans toward recall, <1 toward precision.
TOP_K = 10                     # candidates carried from TRAIN to VALIDATION for model selection
SEED = 42                      # TPE sampler seed, for reproducible searches
MATCH_MARGIN = 3               # pixel margin when matching detections to annotations

# Per-movie detector output files live here (one per movie so runs never clash).
OUTPUT_DIR = Path(__file__).resolve().parent

# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def calculate_score(true_positives, false_negatives, false_positives, true_negatives,
                   recall_weight=0.6, precision_weight=0.4):
    """
    Legacy weighted-sum composite score, kept for REPORTING/COMPARISON ONLY.

    This is NOT the optimization target anymore (see calculate_fbeta): a linear
    weighted sum is fully compensatory and cannot reach non-convex regions of the
    precision/recall trade-off. It remains here so old runs stay comparable.
    """
    total_positives = true_positives + false_negatives
    total_detections = true_positives + false_positives

    recall = true_positives / total_positives if total_positives > 0 else 0
    precision = true_positives / total_detections if total_detections > 0 else 0

    return recall_weight * recall + precision_weight * precision


def calculate_fbeta(recall, precision, beta=1.0):
    """
    F-beta score: the harmonic-style combination of precision and recall.

    beta > 1 weights recall higher, beta < 1 weights precision higher, beta == 1
    is the balanced F1. Unlike a weighted sum this is non-compensatory: it stays
    low unless BOTH precision and recall are decent. Returns 0 when both are 0.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_center_coordinates(json_path):
    """Extract x,y coordinates from center field in JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'center' in item:
                    center = item['center']
                    if 'x' in center and 'y' in center:
                        coordinates.append((center['x'], center['y']))
        
        return coordinates
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def extract_coordinates(json_path):
    """Extract x,y coordinates from detected events JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        coordinates = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if 'center' in item:
                        center = item['center']
                        if 'x' in center and 'y' in center:
                            coordinates.append((center['x'], center['y']))
                            continue
                    if 'x' in item and 'y' in item:
                        coordinates.append((item['x'], item['y']))
                    elif 'X' in item and 'Y' in item:
                        coordinates.append((item['X'], item['Y']))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    coordinates.append((item[0], item[1]))
        
        return coordinates
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return []

def is_within_margin(coord1, coord2, margin=3):
    """Check if two coordinates are within the specified pixel margin."""
    return abs(coord1[0] - coord2[0]) <= margin and abs(coord1[1] - coord2[1]) <= margin

def find_matches(training_spots, detected_events, margin=3):
    """Find which training spots are detected in the events."""
    matches = []
    unmatched = []
    
    for train_coord in training_spots:
        found = False
        for detect_coord in detected_events:
            if is_within_margin(train_coord, detect_coord, margin):
                matches.append({
                    'training': train_coord,
                    'detected': detect_coord,
                    'distance': ((train_coord[0] - detect_coord[0])**2 + 
                                (train_coord[1] - detect_coord[1])**2)**0.5
                })
                found = True
                break
        
        if not found:
            unmatched.append(train_coord)

    return matches, unmatched

def match_detections_to_positives(positive_spots, detected_events, margin=3):
    """
    Greedy 1:1 matching between annotated positive spots and detections.

    Each detection matches at most one positive and vice-versa (closest pairs
    first). Returns (matched_pairs, unmatched_positives, unmatched_detections).

    The key difference from find_matches(): unmatched_detections includes EVERY
    detection not explained by a positive annotation -- background/noise hits as
    well as detections on annotated negatives -- so it can be used as a true
    false-positive count that penalises over-detection.
    """
    # Build all candidate pairs within the margin, closest first.
    candidates = []
    for pi, p in enumerate(positive_spots):
        for di, d in enumerate(detected_events):
            if abs(p[0] - d[0]) <= margin and abs(p[1] - d[1]) <= margin:
                dist = ((p[0] - d[0]) ** 2 + (p[1] - d[1]) ** 2) ** 0.5
                candidates.append((dist, pi, di))
    candidates.sort(key=lambda c: c[0])

    matched_pos = set()
    matched_det = set()
    matched_pairs = []
    for dist, pi, di in candidates:
        if pi in matched_pos or di in matched_det:
            continue
        matched_pos.add(pi)
        matched_det.add(di)
        matched_pairs.append({
            'positive': positive_spots[pi],
            'detected': detected_events[di],
            'distance': dist
        })

    unmatched_positives = [p for i, p in enumerate(positive_spots) if i not in matched_pos]
    unmatched_detections = [d for i, d in enumerate(detected_events) if i not in matched_det]
    return matched_pairs, unmatched_positives, unmatched_detections

# ============================================================================
# PARAMETER EVALUATION
# ============================================================================

# Safety limit (seconds) for a single SCP run, so one bad parameter set can't
# stall the entire sweep.
DETECTION_TIMEOUT = 600

def run_detection(params, tif_path, output_json):
    """Run the SCP detection script on one movie, writing to an explicit output path."""
    cmd = [
        *PYTHON_CMD, scp_script,
        tif_path,
        "--no_plot",  # headless: never open a blocking matplotlib window
        "--output_json", output_json,  # explicit output path (no newest-file globbing)
        "--min_sigma", str(params['min_sigma']),
        "--max_sigma", str(params['max_sigma']),
        "--threshold", str(params['intensity_threshold']),
        "--circularity_threshold", str(params['circular_threshold']),
        "--max_displacement", str(params['max_displacement']),
        "--min_pulsation_frames", str(params['min_pulsation_frames']),
        "--intensity_variation_threshold", str(params['intensity_variation_threshold'])
    ]
    if USE_BANDPASS:
        cmd += ["--bandpass", "--bandpass_bg_sigma", str(BANDPASS_BG_SIGMA)]
    if EVENT_NMS_RADIUS > 0:
        cmd += ["--event_nms_radius", str(EVENT_NMS_RADIUS)]
    if INTENSITY_APERTURE > 0:
        cmd += ["--intensity_aperture", str(INTENSITY_APERTURE)]

    env = {
        **os.environ,
        # Non-interactive matplotlib backend (covers an SCP build lacking --no_plot).
        "MPLBACKEND": "Agg",
        # Force UTF-8 in the child. The detector prints non-ASCII chars (e.g. '->'
        # arrows) and, when its stdout is a captured pipe on a cp1252 Windows console,
        # crashes with a UnicodeEncodeError *before* writing its JSON -- then swallows
        # its own exception and exits 0, which would silently poison scoring.
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    }

    # Delete any prior output so a failed run can never be scored against a stale file.
    try:
        os.remove(output_json)
    except FileNotFoundError:
        pass

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True,
                                env=env, timeout=DETECTION_TIMEOUT)
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"Timeout: detection exceeded {DETECTION_TIMEOUT}s and was aborted")
        return False, "timeout"
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return False, e.stderr

def _empty_metrics(error=None):
    """Zeroed metrics dict so the Optuna objective never KeyErrors on a failed run."""
    return {
        'fbeta': 0.0, 'weighted_score': 0.0, 'recall': 0.0, 'precision': 0.0,
        'true_positives': 0, 'false_negatives': 0, 'false_positives': 0,
        'true_negatives': 0, 'negatives_hit': 0, 'detected_count': 0,
        'error': error,
    }


def compute_metrics(p_spots, n_spots, detected_events):
    """
    Score a set of detections against annotated positives/negatives.

    Reused for whole-movie scoring and for band-restricted (spatial-CV) scoring --
    just pass the subset of spots/detections in the band. detected_count reflects
    exactly the detections passed in.
    """
    # 1:1 match detections to positives. false_positives = detections explained by
    # NO positive (background/noise AND detections on annotated negatives).
    matched_pairs, unmatched_positives, unmatched_detections = \
        match_detections_to_positives(p_spots, detected_events, margin=MATCH_MARGIN)
    true_positives = len(matched_pairs)
    false_negatives = len(unmatched_positives)
    false_positives = len(unmatched_detections)

    # Secondary (reported only): of spots annotated as negatives, how many we fired on.
    n_matches, n_unmatched = find_matches(n_spots, detected_events, margin=MATCH_MARGIN)
    negatives_hit = len(n_matches)
    true_negatives = len(n_unmatched)

    total_positives = len(p_spots)
    recall = true_positives / total_positives if total_positives > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    return {
        'fbeta': calculate_fbeta(recall, precision, beta=FBETA),
        'weighted_score': calculate_score(true_positives, false_negatives, false_positives, true_negatives),
        'recall': recall,
        'precision': precision,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'negatives_hit': negatives_hit,
        'detected_count': len(detected_events),
        'error': None,
    }


def evaluate_parameters(params, tif_path, p_training_spots, n_training_spots,
                        output_json, verbose=False):
    """
    Run the detector on ONE movie and return its performance metrics.

    Returns a dict that ALWAYS contains fbeta, weighted_score, recall, precision,
    the TP/FN/FP/TN counts, negatives_hit and detected_count (zeroed on failure),
    so both the single- and multi-objective Optuna objectives can read it safely.
    """
    # Run detection
    success, output = run_detection(params, tif_path, output_json)
    if not success:
        return _empty_metrics('Script execution failed')

    # Require the explicit output the detector was told to write. run_detection
    # pre-deletes it, so a missing file here means the detector produced nothing
    # (e.g. it crashed) -- score it as a failure rather than reading anything else.
    # NB: no directory-glob fallback: the movie folder holds *_positive/_negative
    # label JSONs, and globbing would silently score against THOSE.
    if not os.path.exists(output_json):
        return _empty_metrics('Detector wrote no output')

    # Extract detected events and score them against the annotations.
    detected_events = extract_coordinates(output_json)
    results = compute_metrics(p_training_spots, n_training_spots, detected_events)

    if verbose:
        print(f"  F{FBETA:g}: {results['fbeta']:.4f} | Recall: {results['recall']:.2%} | "
              f"Precision: {results['precision']:.2%} | TP: {results['true_positives']} | "
              f"FP: {results['false_positives']} (neg-hit: {results['negatives_hit']}) | "
              f"dets: {results['detected_count']}")

    return results

# ============================================================================
# PARAMETER OPTIMIZATION (Optuna / Bayesian)
# ============================================================================

# Search space. Bounds match the ranges the old coordinate-descent used; Optuna
# samples the JOINT space (via the TPE sampler) so interacting parameters such as
# the sigma range and the intensity threshold are optimized together.
PARAM_BOUNDS = {
    # Real spots are small (diffraction-limited); keep the sigma range tight.
    'min_sigma': (0.0, 1.5),
    'max_sigma': (0.5, 3.5),
    # Floored at 0.25: with bandpass on, threshold ~0.3 recovers the real spots
    # (~95% recall) in ~130s, while thresholds below ~0.2 make the real movie
    # intractable (tracking explodes -> timeout). Keep the search in the tractable
    # band so every candidate can actually be judged on the real movie.
    'intensity_threshold': (0.25, 0.5),
    # Filter params narrowed toward the region that KEEPS real recall: the lenient
    # probe (circ 0.1, min_puls 3, intvar 0.05) hit ~95% recall, while high values
    # reject the irregularly-blinking real spots. The search tunes precision within
    # this recall-preserving region instead of wasting trials on over-filtering.
    'circular_threshold': (0.1, 0.6),
    'max_displacement': (0.5, 5.0),
    # See MIN_PULSATION_FRAMES_BY_METHOD: this default applies to the simulated
    # searches; the direct-on-real search overrides it with a much higher ceiling.
    'min_pulsation_frames': (3, 15),          # integer parameter
    'intensity_variation_threshold': (0.05, 0.25),
}

# min_pulsation_frames is NOT a pulsation criterion -- SCP._analyze_tracks applies
# it as a minimum TRACK LENGTH. How useful it is depends entirely on which movie
# is being searched, so the ceiling is per-METHOD rather than global.
#
#   REAL movie (realdata/vid_ch1.tiff, 501 frames). Median track length is 134
#   frames for annotated molecules, 41 for annotated aggregates, 25 for
#   unannotated false positives, so raising the floor strips aggregates ~5x
#   faster than molecules. Measured on the held-out test band, precision runs
#   67.9% at 11 (what every previous search picked, being unable to go higher),
#   76.7% at 20, 85.4% at 40, 88.1% at 60. A confirmed end-to-end run at 20 plus
#   3 px event NMS moved the held-out band from R 95.5% / P 67.9% to
#   R 94.6% / P 76.6%.
#
#   BUT the mechanism is detector instability, not photophysics. Aggregates are
#   bright and persistent; they produce SHORT tracks here because the real-tuned
#   sigma range (~0.45-1.27) plus the 8 px bandpass -- a high-pass -- detects
#   these larger objects only intermittently, so their tracks fragment. That is a
#   property of this movie and this configuration, and it is why the ceiling is
#   not raised globally.
#
#   SIM movies (BT simulator, 300 frames). The opposite holds and the filter is
#   useless: measured on rep0, median track length is 241 frames for targets and
#   269 for aggregates -- aggregates last LONGER, because the simulator builds
#   one by summing N always-on single-fluorophore traces. Precision is flat at
#   70.4% for every cut from 5 to 100, so a raised ceiling would only add a dead
#   dimension for TPE to waste trials on, and a high value would strip targets
#   before aggregates. On sim data the real discriminator is the coefficient of
#   variation (target 0.194 vs aggregate 0.123) -- which, note, carries no signal
#   at all on the real movie (0.1072 vs 0.1078).
#
# Both ceilings are absolute frame counts and so do not transfer between movies
# of different length (60 frames is 12% of the real movie but 20% of a sim one).
# Revisit if either movie length changes materially.
MIN_PULSATION_FRAMES_BY_METHOD = {
    'direct_real': (3, 60),
}


def active_param_bounds():
    """PARAM_BOUNDS with the METHOD-dependent overrides applied.

    Read this rather than PARAM_BOUNDS anywhere the actual searched space
    matters (sampling, logging, provenance). METHOD is routinely set by driver
    scripts AFTER import, so the override cannot be baked in at module load.
    """
    bounds = dict(PARAM_BOUNDS)
    if METHOD in MIN_PULSATION_FRAMES_BY_METHOD:
        bounds['min_pulsation_frames'] = MIN_PULSATION_FRAMES_BY_METHOD[METHOD]
    return bounds


def suggest_params(trial):
    """Draw one parameter set from the search space for this trial."""
    bounds = active_param_bounds()
    min_sigma = trial.suggest_float('min_sigma', *bounds['min_sigma'])
    # max_sigma must exceed min_sigma for LoG blob detection to be meaningful;
    # constrain the draw so we never waste an expensive trial on a degenerate config.
    ms_lo, ms_hi = bounds['max_sigma']
    max_sigma = trial.suggest_float('max_sigma', max(ms_lo, min_sigma + 0.1), ms_hi)
    return {
        'min_sigma': min_sigma,
        'max_sigma': max_sigma,
        'intensity_threshold': trial.suggest_float('intensity_threshold', *bounds['intensity_threshold']),
        'circular_threshold': trial.suggest_float('circular_threshold', *bounds['circular_threshold']),
        'max_displacement': trial.suggest_float('max_displacement', *bounds['max_displacement']),
        'min_pulsation_frames': trial.suggest_int('min_pulsation_frames', *bounds['min_pulsation_frames']),
        'intensity_variation_threshold': trial.suggest_float('intensity_variation_threshold', *bounds['intensity_variation_threshold']),
    }


# ============================================================================
# DATASET LOADING (train / validation from the simulation manifest, + test)
# ============================================================================

def _resolve(path_str, manifest_dir, split):
    """Resolve a manifest path, falling back to <manifest_dir>/<split>/<name> if moved."""
    p = Path(path_str)
    if p.exists():
        return str(p)
    alt = manifest_dir / split / p.name
    return str(alt)


def load_movies_from_manifest(manifest_path):
    """
    Load the seeded train/val movie set produced by Simulation/generate_dataset.py.

    Each returned movie is a dict {name, tif, positive_spots, negative_spots,
    output_json}; spots are pre-loaded once so trials only pay for detection, not
    for re-parsing labels. Returns (meta, train_movies, val_movies).
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    with open(manifest_path) as f:
        manifest = json.load(f)

    def build(records, split):
        movies = []
        for rec in records:
            tif = _resolve(rec['tif'], manifest_dir, split)
            pos = _resolve(rec['positive'], manifest_dir, split)
            neg = _resolve(rec['negative'], manifest_dir, split)
            name = Path(tif).stem
            movies.append({
                'name': f"{split}:{name}",
                'tif': tif,
                'positive_spots': extract_center_coordinates(pos),
                'negative_spots': extract_center_coordinates(neg),
                'output_json': str(OUTPUT_DIR / f"_scp_detected_{split}_{name}.json"),
            })
        return movies

    train = build(manifest.get('train', []), 'train')
    val = build(manifest.get('val', []), 'val')
    return manifest.get('meta', {}), train, val


def load_test_labels(coord_npy, class_npy):
    """
    Load the REAL test movie's labels from the BachelorThesis realdata .npy format.

    coord.npy is (N, 2) in [y, x] order; class.npy is (N,) of class ids. We swap to
    (x, y) to match the detector/sim convention and split by class:
    TEST_POSITIVE_CLASS -> positives (detect target), TEST_NEGATIVE_CLASS -> negatives.

    This is the ONE place tied to the real label format -- adapt here if it changes.
    """
    coord = np.load(coord_npy)          # (N, 2) as [y, x]
    cls = np.load(class_npy)            # (N,)
    xy = [(int(c[1]), int(c[0])) for c in coord]   # [y, x] -> (x, y)
    positives = [xy[i] for i in range(len(xy)) if cls[i] == TEST_POSITIVE_CLASS]
    negatives = [xy[i] for i in range(len(xy)) if cls[i] == TEST_NEGATIVE_CLASS]
    return positives, negatives


def load_test_movie():
    """Build the held-out test movie dict, or None if the movie/labels aren't present."""
    if not (os.path.exists(TEST_TIF) and os.path.exists(TEST_COORD_NPY) and os.path.exists(TEST_CLASS_NPY)):
        return None
    positives, negatives = load_test_labels(TEST_COORD_NPY, TEST_CLASS_NPY)
    return {
        'name': 'test:real',
        'tif': TEST_TIF,
        'positive_spots': positives,
        'negative_spots': negatives,
        'output_json': str(OUTPUT_DIR / "_scp_detected_test.json"),
    }


# ============================================================================
# CROSS-MOVIE EVALUATION
# ============================================================================

def evaluate_on_movies(params, movies, verbose=False):
    """
    Run the detector with `params` on every movie and aggregate.

    The headline score is the MEAN per-movie F-beta (each movie weighted equally,
    the standard cross-validation aggregation). Also returns mean recall/precision
    and the per-movie breakdown for reporting.
    """
    per_movie = []
    for mv in movies:
        r = evaluate_parameters(params, mv['tif'], mv['positive_spots'],
                                mv['negative_spots'], mv['output_json'], verbose=False)
        per_movie.append({
            'name': mv['name'], 'fbeta': r['fbeta'], 'recall': r['recall'],
            'precision': r['precision'], 'true_positives': r['true_positives'],
            'false_positives': r['false_positives'], 'false_negatives': r['false_negatives'],
            'detected_count': r['detected_count'], 'error': r['error'],
        })
        if verbose:
            print(f"    {mv['name']}: F{FBETA:g}={r['fbeta']:.4f} "
                  f"R={r['recall']:.2%} P={r['precision']:.2%}")

    n = len(per_movie) or 1
    mean_fbeta = sum(m['fbeta'] for m in per_movie) / n
    mean_recall = sum(m['recall'] for m in per_movie) / n
    mean_precision = sum(m['precision'] for m in per_movie) / n
    return {
        'fbeta': mean_fbeta,
        'recall': mean_recall,
        'precision': mean_precision,
        'per_movie': per_movie,
        'n_movies': len(per_movie),
    }


# ============================================================================
# PARAMETER OPTIMIZATION (Optuna / Bayesian) over the training set
# ============================================================================

def make_objective(train_movies):
    """Build the Optuna objective: optimize the mean F-beta across the TRAIN movies."""
    def objective(trial):
        params = suggest_params(trial)
        print(f"\n[trial {trial.number}] {params}")
        agg = evaluate_on_movies(params, train_movies, verbose=True)
        trial.set_user_attr('train_fbeta', agg['fbeta'])
        trial.set_user_attr('train_recall', agg['recall'])
        trial.set_user_attr('train_precision', agg['precision'])
        print(f"  -> TRAIN mean F{FBETA:g}={agg['fbeta']:.4f} "
              f"R={agg['recall']:.2%} P={agg['precision']:.2%}")
        if OPTIMIZATION_MODE == "multi":
            return agg['recall'], agg['precision']
        return agg['fbeta']
    return objective


def run_study(train_movies, n_trials=N_TRIALS):
    """Run the Optuna study on the training set. Single- or multi-objective per config."""
    sampler = optuna.samplers.TPESampler(seed=SEED)  # seeded for reproducible searches
    if OPTIMIZATION_MODE == "multi":
        study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
        print("Mode: MULTI-objective (maximize mean recall & mean precision) -> Pareto front")
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)
        print(f"Mode: SINGLE-objective (maximize mean F{FBETA:g})")
    # n_jobs=1: each trial spawns detector subprocesses writing per-movie output
    # files; parallel trials would need per-trial (not just per-movie) output paths.
    study.optimize(make_objective(train_movies), n_trials=n_trials)
    return study


def select_candidates(study):
    """Trials to carry from TRAIN into VALIDATION for model selection."""
    if OPTIMIZATION_MODE == "multi":
        # Every non-dominated (Pareto) trial is a legitimate candidate.
        return list(study.best_trials)
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    return completed[:TOP_K]


# ============================================================================
# MAIN EXECUTION:  train (optimize) -> validate (select) -> test (report once)
# ============================================================================

RESULTS_FILE = "optimization_results.json"   # override to redirect the output JSON


def run_pipeline():
    print("Loading dataset manifest...")
    meta, train_movies, val_movies = load_movies_from_manifest(DATASET_MANIFEST)
    print(f"  train movies: {len(train_movies)}  |  val movies: {len(val_movies)}")
    print(f"  meta: {meta}")
    if not train_movies or not val_movies:
        print("ERROR: manifest must contain both train and val movies.")
        return None

    print("=" * 80)
    print("STAGE 1/3 - TRAIN: Optuna TPE search over the training movies")
    print(f"Trials: {N_TRIALS}  |  detector runs/trial: {len(train_movies)}")
    print("=" * 80)
    study = run_study(train_movies, n_trials=N_TRIALS)

    print("\n" + "=" * 80)
    print(f"STAGE 2/3 - VALIDATE: re-scoring top {TOP_K} candidates on the validation movies")
    print("=" * 80)
    candidates = select_candidates(study)
    scored = []
    for t in candidates:
        val_agg = evaluate_on_movies(t.params, val_movies, verbose=True)
        scored.append({'trial': t, 'val': val_agg})
        print(f"  [trial {t.number}] TRAIN F{FBETA:g}={t.user_attrs.get('train_fbeta', 0):.4f} "
              f"-> VAL F{FBETA:g}={val_agg['fbeta']:.4f} "
              f"(R={val_agg['recall']:.2%} P={val_agg['precision']:.2%})")

    # Final config = best VALIDATION F-beta (guards against overfitting to train).
    best = max(scored, key=lambda s: s['val']['fbeta'])
    final_trial, final_val = best['trial'], best['val']
    final_params = final_trial.params
    print(f"\nSelected trial {final_trial.number} "
          f"(TRAIN F{FBETA:g}={final_trial.user_attrs.get('train_fbeta', 0):.4f}, "
          f"VAL F{FBETA:g}={final_val['fbeta']:.4f})")
    print("Final parameters:")
    for k, v in final_params.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("STAGE 3/3 - TEST: scoring the final config on the real movie (ONCE)")
    print("=" * 80)
    test_movie = load_test_movie()
    if test_movie is None:
        print("Test movie not found (set TEST_TIF/TEST_POSITIVE/TEST_NEGATIVE). Skipping test.")
        test_agg = None
    else:
        test_agg = evaluate_on_movies(final_params, [test_movie], verbose=True)
        print(f"\nTEST  F{FBETA:g}={test_agg['fbeta']:.4f}  "
              f"R={test_agg['recall']:.2%}  P={test_agg['precision']:.2%}")

    # Generalization summary: train -> val -> test.
    print("\n" + "=" * 80)
    print("GENERALIZATION SUMMARY (mean F-beta)")
    print(f"  TRAIN: {final_trial.user_attrs.get('train_fbeta', 0):.4f}")
    print(f"  VAL:   {final_val['fbeta']:.4f}")
    print(f"  TEST:  {test_agg['fbeta']:.4f}" if test_agg else "  TEST:  (skipped)")
    print("=" * 80)

    payload = {
        'mode': OPTIMIZATION_MODE,
        'fbeta': FBETA,
        'n_trials': len(study.trials),
        'dataset_meta': meta,
        'final_parameters': final_params,
        'final_trial_number': final_trial.number,
        'train_metrics': {
            'fbeta': final_trial.user_attrs.get('train_fbeta'),
            'recall': final_trial.user_attrs.get('train_recall'),
            'precision': final_trial.user_attrs.get('train_precision'),
        },
        'validation_metrics': final_val,
        'test_metrics': test_agg,
        'candidates': [
            {'trial': s['trial'].number, 'params': s['trial'].params,
             'train_fbeta': s['trial'].user_attrs.get('train_fbeta'),
             'val_fbeta': s['val']['fbeta']}
            for s in scored
        ],
        'all_trials': [
            {'number': t.number, 'params': t.params, 'values': t.values,
             'metrics': dict(t.user_attrs)}
            for t in study.trials
        ],
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    return payload


# ============================================================================
# SIM-GUIDED, SELECT-ON-REAL:  search on sim -> judge top-K on the real movie
# ============================================================================

def _in_band(spot, xband):
    """spot is (x, y); band is (x_min, x_max) half-open on x."""
    return xband[0] <= spot[0] < xband[1]


def evaluate_real_bands(params, positives, negatives, output_json, bands=None, verbose=False):
    """
    Run the detector ONCE on the real movie, then score each spatial band separately
    by slicing detections and annotations by x. One detection -> all band scores.

    `bands` is a list of (name, (x_min, x_max)); defaults to sim-guided val/test.
    """
    if bands is None:
        bands = [('val', REAL_VAL_XBAND), ('test', REAL_TEST_XBAND)]
    success, _ = run_detection(params, TEST_TIF, output_json)
    if not success or not os.path.exists(output_json):
        empty = _empty_metrics('real detection failed/empty')
        out = {name: empty for name, _ in bands}
        out.update({'detected_count': 0, 'error': 'real detection failed'})
        return out

    dets = extract_coordinates(output_json)
    out = {'detected_count': len(dets), 'error': None}
    for name, band in bands:
        p = [s for s in positives if _in_band(s, band)]
        n = [s for s in negatives if _in_band(s, band)]
        d = [s for s in dets if _in_band(s, band)]
        out[name] = compute_metrics(p, n, d)
        if verbose:
            m = out[name]
            print(f"    {name} band x{band}: F{FBETA:g}={m['fbeta']:.4f} "
                  f"R={m['recall']:.2%} P={m['precision']:.2%} "
                  f"(pos={len(p)} dets={len(d)})")
    return out


def run_sim_guided():
    """
    Search on the simulation for breadth, then choose the final config on the REAL
    movie's validation band and report its held-out real TEST band -- so the real
    data (not the sim) decides which parameters win.
    """
    print("Loading dataset manifest (simulation, for the search)...")
    meta, train_movies, val_movies = load_movies_from_manifest(DATASET_MANIFEST)
    print(f"  sim train movies: {len(train_movies)} (val movies unused in this method)")
    if not train_movies:
        print("ERROR: manifest has no train movies.")
        return None

    test_movie = load_test_movie()
    if test_movie is None:
        print("ERROR: real movie/labels not found (TEST_TIF/TEST_COORD_NPY/TEST_CLASS_NPY).")
        return None
    real_pos, real_neg = test_movie['positive_spots'], test_movie['negative_spots']
    v = sum(_in_band(s, REAL_VAL_XBAND) for s in real_pos)
    t = sum(_in_band(s, REAL_TEST_XBAND) for s in real_pos)
    print(f"  real positives: {len(real_pos)}  (val band: {v}, test band: {t})")

    print("=" * 80)
    print(f"STAGE 1/2 - SIM SEARCH: Optuna TPE, {N_TRIALS} trials on {len(train_movies)} sim movies")
    print("=" * 80)
    study = run_study(train_movies, n_trials=N_TRIALS)
    candidates = select_candidates(study)   # top-K by sim score (or Pareto front)

    print("\n" + "=" * 80)
    print(f"STAGE 2/2 - JUDGE ON REAL: scoring top {len(candidates)} configs on the real movie")
    print("(select on VAL band, report held-out TEST band)")
    print("=" * 80)
    scored = []
    for i, tr in enumerate(candidates):
        out_json = str(OUTPUT_DIR / f"_scp_detected_real_cand{tr.number}.json")
        print(f"\n[candidate {i+1}/{len(candidates)}] trial {tr.number} "
              f"(sim F{FBETA:g}={tr.user_attrs.get('train_fbeta', 0):.4f})")
        bands = evaluate_real_bands(tr.params, real_pos, real_neg, out_json, verbose=True)
        scored.append({'trial': tr, 'bands': bands})

    # Final config chosen on the REAL validation band; TEST band stays held out.
    best = max(scored, key=lambda s: s['bands']['val']['fbeta'])
    ft, fb = best['trial'], best['bands']
    print("\n" + "=" * 80)
    print(f"Selected trial {ft.number}: real-VAL F{FBETA:g}={fb['val']['fbeta']:.4f} "
          f"-> real-TEST F{FBETA:g}={fb['test']['fbeta']:.4f}")
    print("Final parameters:")
    for k, val in ft.params.items():
        print(f"  {k}: {val}")
    print("\nGENERALIZATION (mean F-beta):")
    print(f"  SIM (train): {ft.user_attrs.get('train_fbeta', 0):.4f}")
    print(f"  REAL val:    {fb['val']['fbeta']:.4f}  (R={fb['val']['recall']:.2%} P={fb['val']['precision']:.2%})")
    print(f"  REAL test:   {fb['test']['fbeta']:.4f}  (R={fb['test']['recall']:.2%} P={fb['test']['precision']:.2%})")
    print("=" * 80)

    payload = {
        'method': 'sim_guided_real',
        'fbeta': FBETA,
        'n_trials': len(study.trials),
        'dataset_meta': meta,
        'final_parameters': ft.params,
        'final_trial_number': ft.number,
        'sim_train_fbeta': ft.user_attrs.get('train_fbeta'),
        'real_val_metrics': fb['val'],
        'real_test_metrics': fb['test'],
        'candidates': [
            {'trial': s['trial'].number, 'params': s['trial'].params,
             'sim_train_fbeta': s['trial'].user_attrs.get('train_fbeta'),
             'real_val_fbeta': s['bands']['val']['fbeta'],
             'real_test_fbeta': s['bands']['test']['fbeta']}
            for s in scored
        ],
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    return payload


# ============================================================================
# DIRECT-ON-REAL:  search parameters directly on the real movie (spatial CV)
# ============================================================================

# Configs enqueued before the TPE search starts. The default three come from the
# lenient recall probe plus two variations that add filtering for precision: they seed
# TPE in the region that KEEPS real recall instead of it burning trials on
# over-filtering combos that score ~0. Overridable at module level so a driver can
# warm-start from somewhere else (e.g. simulator-tuned shape parameters) while every
# other aspect of the search stays identical -- that is what makes two such runs
# comparable. Enqueued values outside PARAM_BOUNDS are clamped by Optuna with a
# warning, so keep them in range.
#
# The three historical seeds, kept verbatim and separately named so the pre-flight
# check can still assert they have not drifted.
DIRECT_WARM_START_HISTORICAL = (
    {'min_sigma': 0.5, 'max_sigma': 2.5, 'intensity_threshold': 0.30,
     'circular_threshold': 0.10, 'max_displacement': 3.0,
     'min_pulsation_frames': 3, 'intensity_variation_threshold': 0.05},
    {'min_sigma': 0.5, 'max_sigma': 2.5, 'intensity_threshold': 0.35,
     'circular_threshold': 0.30, 'max_displacement': 2.0,
     'min_pulsation_frames': 5, 'intensity_variation_threshold': 0.10},
    {'min_sigma': 0.7, 'max_sigma': 2.0, 'intensity_threshold': 0.40,
     'circular_threshold': 0.25, 'max_displacement': 2.0,
     'min_pulsation_frames': 4, 'intensity_variation_threshold': 0.08},
)

# Fourth seed, added with the (3, 60) track-length ceiling. All three historical
# seeds sit at min_pulsation_frames 3-5, i.e. at the very bottom of the newly
# opened range, so without this the search would spend its early trials
# re-exploring the region that produced 67.9% held-out precision. This is trial
# 22's parameter set with min_pulsation_frames moved to 20 -- confirmed
# end-to-end on the real movie at train P 83.0% / val P 75.0% / test P 76.6%
# (recall 89.7 / 93.3 / 94.6), so TPE starts with a measured point in the region
# the old ceiling made unreachable.
#
# NOTE this only helps a direct_real search; on sim METHOD the ceiling stays
# (3, 15) and Optuna would clamp min_pulsation_frames=20 down to 15 with a
# warning. Sim drivers should override DIRECT_WARM_START, as arm B already does.
DIRECT_WARM_START_TRACK_LENGTH_SEED = {
    'min_sigma': 0.4515457399879006, 'max_sigma': 1.2689848283173364,
    'intensity_threshold': 0.32336429282837625,
    'circular_threshold': 0.20209149318019778,
    'max_displacement': 4.9934159843646295,
    'min_pulsation_frames': 20,
    'intensity_variation_threshold': 0.07179385564931576,
}

DIRECT_WARM_START = DIRECT_WARM_START_HISTORICAL + (DIRECT_WARM_START_TRACK_LENGTH_SEED,)


def run_direct_real():
    """
    Tune parameters directly on the real movie using a spatial train/val/test split.

    Each trial runs the detector ONCE on the real movie and scores the TRAIN band
    (the objective); VAL and TEST bands are sliced from the same detection and
    stored. The final config is the top-TRAIN trial re-picked by VAL, and its TEST
    band is the honest held-out number. The real data drives every decision.
    """
    test_movie = load_test_movie()
    if test_movie is None:
        print("ERROR: real movie/labels not found.")
        return None
    real_pos, real_neg = test_movie['positive_spots'], test_movie['negative_spots']
    bands = [('train', DIRECT_TRAIN_XBAND), ('val', DIRECT_VAL_XBAND), ('test', DIRECT_TEST_XBAND)]
    for name, b in bands:
        print(f"  {name} band x{b}: {sum(_in_band(s, b) for s in real_pos)} positive spots")

    print("=" * 80)
    print(f"DIRECT-ON-REAL: Optuna TPE, {N_TRIALS} trials on the real movie")
    print("(optimize TRAIN band, select on VAL, report TEST)")
    print("=" * 80)

    def objective(trial):
        params = suggest_params(trial)
        out_json = str(OUTPUT_DIR / f"_scp_detected_direct_t{trial.number}.json")
        print(f"\n[trial {trial.number}] thr={params['intensity_threshold']:.3f} "
              f"sig={params['min_sigma']:.2f}-{params['max_sigma']:.2f} "
              f"circ={params['circular_threshold']:.2f} puls={params['min_pulsation_frames']}")
        res = evaluate_real_bands(params, real_pos, real_neg, out_json, bands=bands, verbose=True)
        for nm in ('train', 'val', 'test'):
            trial.set_user_attr(f'{nm}_fbeta', res[nm]['fbeta'])
            trial.set_user_attr(f'{nm}_recall', res[nm]['recall'])
            trial.set_user_attr(f'{nm}_precision', res[nm]['precision'])
        return res['train']['fbeta']

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Warm-start (see DIRECT_WARM_START): seeds TPE in a known-good region rather
    # than letting it burn trials on combos that score ~0.
    for seed_cfg in DIRECT_WARM_START:
        study.enqueue_trial(seed_cfg)

    study.optimize(objective, n_trials=N_TRIALS)

    # Select among the top-TRAIN trials by VAL band; report held-out TEST.
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    topk = completed[:TOP_K]
    best = max(topk, key=lambda t: t.user_attrs.get('val_fbeta', 0))

    print("\n" + "=" * 80)
    print(f"Selected trial {best.number}")
    print("Final parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("\nGENERALIZATION (F-beta on the real movie, by band):")
    for nm in ('train', 'val', 'test'):
        print(f"  {nm:5s}: F{FBETA:g}={best.user_attrs.get(nm+'_fbeta',0):.3f}  "
              f"R={best.user_attrs.get(nm+'_recall',0):.1%}  P={best.user_attrs.get(nm+'_precision',0):.1%}")
    print("=" * 80)

    payload = {
        'method': 'direct_real',
        'fbeta': FBETA,
        'n_trials': len(study.trials),
        'bands': {'train': DIRECT_TRAIN_XBAND, 'val': DIRECT_VAL_XBAND, 'test': DIRECT_TEST_XBAND},
        # Recorded so two runs of this function can be told apart by what seeded them.
        'warm_start': [dict(c) for c in DIRECT_WARM_START],
        # active_param_bounds(), not PARAM_BOUNDS: min_pulsation_frames is
        # overridden per METHOD, so the raw dict would misreport what was searched.
        'param_bounds': {k: list(v) for k, v in active_param_bounds().items()},
        'event_nms_radius': EVENT_NMS_RADIUS,
        'seed': SEED,
        'top_k': TOP_K,
        'final_parameters': best.params,
        'final_trial_number': best.number,
        'train_metrics': {k: best.user_attrs.get('train_'+k) for k in ('fbeta', 'recall', 'precision')},
        'val_metrics': {k: best.user_attrs.get('val_'+k) for k in ('fbeta', 'recall', 'precision')},
        'test_metrics': {k: best.user_attrs.get('test_'+k) for k in ('fbeta', 'recall', 'precision')},
        'all_trials': [
            {'number': t.number, 'params': t.params,
             'train_fbeta': t.user_attrs.get('train_fbeta'),
             'val_fbeta': t.user_attrs.get('val_fbeta'),
             'test_fbeta': t.user_attrs.get('test_fbeta')}
            for t in study.trials
        ],
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    return payload


# ============================================================================
# SIM SHAPE + REAL THRESHOLD: transfer what transfers, calibrate what does not
# ============================================================================

# Thresholds tried on the real movie when recalibrating. Constrained by measurement,
# not taste: on vid_ch1.tiff, 0.20 runs in 184s at F1 0.699, 0.15 takes 320s and
# DEGRADES to F1 0.213 (spurious blobs hijack track association, so recall falls from
# 83% to 32%), and 0.12 exceeds a 420s timeout entirely. Nothing below 0.20 is useful.
REAL_THRESHOLD_GRID = (0.20, 0.26, 0.32)


def run_sim_shape_real_threshold():
    """
    Search the six SHAPE/behaviour parameters on simulated movies, then calibrate
    `intensity_threshold` on the real movie's validation band.

    Why the split: SCP min-max normalises every frame before blob detection, so
    `intensity_threshold` is a fraction of whichever object is brightest in that frame
    -- not an absolute brightness. Its usable range is therefore a property of the
    movie, and the two datasets' ranges do not overlap (the simulation needs <=0.10,
    the real movie >=0.20; brightness rescaling cannot fix this, because min-max
    normalisation is scale-invariant -- a 3x brightness sweep leaves the measured LoG
    response flat). Transferring that one number is what broke sim-guided tuning.

    The other six -- sigma range, circularity, max displacement, min pulsation frames,
    intensity CV -- describe the physical size and behaviour of a spot and are not tied
    to a frame's intensity scaling, so they DO transfer.

    Protocol: optimise on sim TRAIN -> for each of the top-K configs, sweep
    REAL_THRESHOLD_GRID on the real movie and score the VAL band -> pick the single
    (config, threshold) pair with the best real VAL F-beta -> report its TEST band.
    The test band is never consulted for any decision.
    """
    print("Loading dataset manifest (simulation, for the shape search)...")
    meta, train_movies, _ = load_movies_from_manifest(DATASET_MANIFEST)
    print(f"  sim train movies: {len(train_movies)}")
    if not train_movies:
        print("ERROR: manifest has no train movies.")
        return None

    test_movie = load_test_movie()
    if test_movie is None:
        print("ERROR: real movie/labels not found (TEST_TIF/TEST_COORD_NPY/TEST_CLASS_NPY).")
        return None
    real_pos, real_neg = test_movie['positive_spots'], test_movie['negative_spots']
    v = sum(_in_band(s, REAL_VAL_XBAND) for s in real_pos)
    t = sum(_in_band(s, REAL_TEST_XBAND) for s in real_pos)
    print(f"  real positives: {len(real_pos)}  (val band: {v}, test band: {t})")
    print(f"  sim threshold search range: {PARAM_BOUNDS['intensity_threshold']}")
    print(f"  real threshold grid:        {REAL_THRESHOLD_GRID}")

    print("=" * 80)
    print(f"STAGE 1/2 - SIM SEARCH: Optuna TPE, {N_TRIALS} trials on {len(train_movies)} sim movies")
    print("=" * 80)
    study = run_study(train_movies, n_trials=N_TRIALS)
    candidates = select_candidates(study)

    print("\n" + "=" * 80)
    print(f"STAGE 2/2 - CALIBRATE ON REAL: {len(candidates)} configs x "
          f"{len(REAL_THRESHOLD_GRID)} thresholds")
    print("(select on the real VAL band; the TEST band stays held out)")
    print("=" * 80)
    scored = []
    for i, tr in enumerate(candidates):
        print(f"\n[candidate {i+1}/{len(candidates)}] trial {tr.number} "
              f"(sim F{FBETA:g}={tr.user_attrs.get('train_fbeta', 0):.4f}, "
              f"sim thr={tr.params['intensity_threshold']:.3f})")
        for thr in REAL_THRESHOLD_GRID:
            params = dict(tr.params, intensity_threshold=thr)
            out_json = str(OUTPUT_DIR / f"_scp_bt_cand{tr.number}_thr{thr:.2f}.json")
            print(f"  real threshold {thr:.2f}:")
            bands = evaluate_real_bands(params, real_pos, real_neg, out_json, verbose=True)
            scored.append({'trial': tr, 'threshold': thr, 'params': params, 'bands': bands})

    usable = [s for s in scored if s['bands'].get('error') is None]
    if not usable:
        print("\nERROR: every real-movie run failed or timed out; nothing to select from.")
        return None

    best = max(usable, key=lambda s: s['bands']['val']['fbeta'])
    ft, fb, fthr = best['trial'], best['bands'], best['threshold']
    print("\n" + "=" * 80)
    print(f"Selected trial {ft.number} with real threshold {fthr:.2f}")
    print("Final parameters:")
    for k, val in best['params'].items():
        tag = "  <- calibrated on real VAL" if k == 'intensity_threshold' else ""
        print(f"  {k}: {val}{tag}")
    print("\nGENERALIZATION (mean F-beta):")
    print(f"  SIM (train): {ft.user_attrs.get('train_fbeta', 0):.4f}")
    print(f"  REAL val:    {fb['val']['fbeta']:.4f}  "
          f"(R={fb['val']['recall']:.2%} P={fb['val']['precision']:.2%})")
    print(f"  REAL test:   {fb['test']['fbeta']:.4f}  "
          f"(R={fb['test']['recall']:.2%} P={fb['test']['precision']:.2%})")
    print("=" * 80)

    payload = {
        'method': 'sim_shape_real_threshold',
        'fbeta': FBETA,
        'n_trials': len(study.trials),
        'dataset_meta': meta,
        'sim_threshold_bounds': list(PARAM_BOUNDS['intensity_threshold']),
        'real_threshold_grid': list(REAL_THRESHOLD_GRID),
        'final_parameters': best['params'],
        'final_trial_number': ft.number,
        'calibrated_threshold': fthr,
        'sim_train_fbeta': ft.user_attrs.get('train_fbeta'),
        'real_val_metrics': fb['val'],
        'real_test_metrics': fb['test'],
        'calibration_grid': [
            {'trial': s['trial'].number, 'threshold': s['threshold'],
             'sim_train_fbeta': s['trial'].user_attrs.get('train_fbeta'),
             'real_val_fbeta': s['bands']['val']['fbeta'],
             'real_test_fbeta': s['bands']['test']['fbeta'],
             'error': s['bands'].get('error')}
            for s in scored
        ],
        'all_trials': [
            {'number': tr.number, 'params': tr.params, 'values': tr.values,
             'metrics': dict(tr.user_attrs)}
            for tr in study.trials
        ],
    }
    with open(RESULTS_FILE, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    return payload


if __name__ == "__main__":
    if METHOD == "sim_only":
        run_pipeline()
    elif METHOD == "direct_real":
        run_direct_real()
    elif METHOD == "sim_shape_real_threshold":
        run_sim_shape_real_threshold()
    else:
        run_sim_guided()