"""
Measurement primitives shared by measure_real_movie.py and validate_against_real.py.

Every number in SIMULATOR_SPEC.md section 4 and every acceptance gate in section 7
is computed by a function in this module, so the real movie and a simulated movie
are always measured by *identical* code. That is the whole point of the file: a
side-by-side comparison is only meaningful if both sides went through the same
estimator, including its arbitrary choices (aperture radii, frame sampling,
preprocessing order).

Two deliberate reuses, per SIMULATOR_SPEC.md section 9:

  * trap 4 ("do not write a second scorer") -- detection scoring comes from
    Parameter_opt/SCP_parametrizer_002.py (compute_metrics / the greedy 1:1
    matcher at MATCH_MARGIN = 3). Nothing here re-implements it.
  * the detector is invoked through that module's run_detection(), so acceptance
    runs use the same subprocess, the same CLI flags and the same UTF-8
    environment fix (trap 2) as the tuning experiments on record.

Axis-order convention (trap 1). Everything in this module speaks (x, y) with x =
column, matching the detector and the label JSON. The BachelorThesis .npy format
is [y, x] and is converted at exactly one place, load_labels_npy().
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
from skimage.feature import blob_log

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent                       # smTIRF_GUI_SCP_Sim_Param
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import SCP_parametrizer_002 as pm  # noqa: E402  (path set up above)

# Point the parametrizer at this checkout. Its module-level defaults are absolute
# paths on a different machine; run_detection() would otherwise launch nothing.
pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]


# ============================================================================
# REFERENCE DETECTOR CONFIGURATION  (SIMULATOR_SPEC.md section 4f)
# ============================================================================
# The tuned real-movie configuration: held-out R 94.6% / P 76.6% / F1 0.847.
# Used wherever "the detector" is invoked for calibration or acceptance, and NOT
# tuned as part of the simulator work. Keys match run_detection()'s expectations.
REFERENCE_DETECTOR = {
    "min_sigma": 0.4515457399879006,
    "max_sigma": 1.2689848283173364,
    "intensity_threshold": 0.32336429282837625,
    "circular_threshold": 0.20209149318019778,
    "max_displacement": 4.9934159843646295,
    "min_pulsation_frames": 20,
    "intensity_variation_threshold": 0.07179385564931576,
}
# Module-level detector switches that run_detection() reads rather than taking as
# params. bandpass on with bg sigma 8.0 and 3 px event NMS are part of section 4f.
REFERENCE_BANDPASS = True
REFERENCE_BANDPASS_BG_SIGMA = 8.0
REFERENCE_EVENT_NMS_RADIUS = 3.0
REFERENCE_INTENSITY_APERTURE = 0.0      # legacy photometry, deliberately (trap 5)

# Photometry recipe of section 4d. Aperture radius 2 px, local background from the
# 4-7 px annulus, every 5th frame (101 frames of the real movie's 501).
PHOTOMETRY = {"r_signal": 2.0, "r_in": 4.0, "r_out": 7.0, "frame_step": 5}

# Class ids in the BachelorThesis .npy ground-truth format.
CLASS_TARGET = 0        # stationary / detection target -> positives
CLASS_OFFTARGET = 1     # aggregate / off-target        -> negatives
CLASS_MOVING = 2        # absent from the real movie


def apply_reference_detector_globals():
    """Force the parametrizer's detector switches to the section 4f values.

    run_detection() reads USE_BANDPASS / BANDPASS_BG_SIGMA / EVENT_NMS_RADIUS /
    INTENSITY_APERTURE from module state, not from its params argument, so they
    have to be set explicitly before any acceptance run. Call this rather than
    trusting the parametrizer's own defaults, which a driver script may have
    changed.
    """
    pm.USE_BANDPASS = REFERENCE_BANDPASS
    pm.BANDPASS_BG_SIGMA = REFERENCE_BANDPASS_BG_SIGMA
    pm.EVENT_NMS_RADIUS = REFERENCE_EVENT_NMS_RADIUS
    pm.INTENSITY_APERTURE = REFERENCE_INTENSITY_APERTURE


# ============================================================================
# I/O
# ============================================================================

def load_movie(path):
    """Read a movie to (T, H, W). Rejects anything else loudly.

    Deliberately stricter than SCP.load_tif_video: a (T, C, H, W) stack is the
    known trap of section 6d, and silently taking a channel here would hide a
    simulator that wrote the wrong shape.
    """
    arr = tifffile.imread(str(path))
    if arr.ndim != 3:
        raise ValueError(
            f"{path}: expected a single-channel (T, H, W) stack, got {arr.shape}. "
            "See SIMULATOR_SPEC.md section 6d -- (T, C, H, W) is not acceptable."
        )
    return arr


def movie_shape(path):
    """(T, H, W) without reading the pixels."""
    with tifffile.TiffFile(str(path)) as tf:
        shape = tuple(tf.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"{path}: expected 3 dims, got {shape}")
    return shape


def load_labels_npy(coord_npy, class_npy):
    """BachelorThesis ground truth -> ((target_xy, offtarget_xy), classes).

    coord.npy is (N, 2) in [y, x]; this is the ONE place that flip happens
    (trap 1). Returned coordinates are (x, y) float tuples.
    """
    coord = np.load(str(coord_npy))
    cls = np.load(str(class_npy))
    xy = [(float(c[1]), float(c[0])) for c in coord]
    targets = [xy[i] for i in range(len(xy)) if cls[i] == CLASS_TARGET]
    offtargets = [xy[i] for i in range(len(xy)) if cls[i] == CLASS_OFFTARGET]
    return (targets, offtargets), np.asarray(cls)


def load_labels_json(positive_json, negative_json):
    """Label-JSON ground truth -> (target_xy, offtarget_xy).

    Goes through the parametrizer's own reader, so a label file that this
    project's consumer cannot parse fails here too instead of being quietly
    accepted by a private parser (section 7-T8).
    """
    return (pm.extract_center_coordinates(str(positive_json)),
            pm.extract_center_coordinates(str(negative_json)))


# ============================================================================
# SPATIAL STATISTICS  (section 4c / gate T5)
# ============================================================================

def nn_distances(xy):
    """Nearest-neighbour distance for each point, within the given set.

    Recipe of section 4c: cKDTree, query(k=2), take column 1. Fewer than two
    points gives an empty array rather than raising -- a movie with one spot is
    degenerate but not an error here.
    """
    pts = np.asarray(xy, dtype=float)
    if len(pts) < 2:
        return np.empty(0)
    d, _ = cKDTree(pts).query(pts, k=2)
    return np.asarray(d[:, 1], dtype=float)


def nn_summary(xy):
    """Percentile summary of the NN distribution plus the close-pair counts.

    close_pairs_under_N counts POINTS whose nearest neighbour is under N px, which
    is how the numbers in section 4c were produced (2 under 4 px, 12 under 6 px,
    33 under 8 px for targets); it is not a count of unordered pairs.
    """
    return nn_summary_from_distances(nn_distances(xy), n_points=len(xy))


def nn_summary_from_distances(d, n_points=None):
    """nn_summary() over a distance array that has already been computed.

    The validation harness needs this to pool distances across several simulated
    movies: nearest-neighbour distance is an intra-movie quantity, so it must be
    computed per movie and only then concatenated -- never by pooling coordinates
    from different movies into one KD-tree.
    """
    d = np.asarray(d, dtype=float)
    n = int(len(d) if n_points is None else n_points)
    if d.size == 0:
        return {"n": n, "distances": []}
    return {
        "n": n,
        "min": float(d.min()),
        "p1": float(np.percentile(d, 1)),
        "p2": float(np.percentile(d, 2)),
        "p5": float(np.percentile(d, 5)),
        "p10": float(np.percentile(d, 10)),
        "p25": float(np.percentile(d, 25)),
        "median": float(np.median(d)),
        "p75": float(np.percentile(d, 75)),
        "p90": float(np.percentile(d, 90)),
        "mean": float(d.mean()),
        "max": float(d.max()),
        "close_pairs_under_4": int(np.sum(d < 4)),
        "close_pairs_under_6": int(np.sum(d < 6)),
        "close_pairs_under_8": int(np.sum(d < 8)),
        # Kept in full so gate T5 can run a two-sample KS test against it.
        "distances": [round(float(v), 4) for v in np.sort(d)],
    }


# ============================================================================
# PHOTOMETRY  (section 4d / gate T3)
# ============================================================================

def _spot_window(shape, x, y, r_out):
    """Local window around (x, y), CLIPPED to the frame, plus the squared-distance
    grid over it.

    Clipped, not skipped. Spots do sit on the border of the real movie (annotated
    coordinates reach x=511 and y=1 of a 256x512 frame), and dropping them changes
    the population being described: reproducing section 4d requires all 243 + 91
    spots, with partial discs and partial annuli measured over whatever pixels
    exist. Verified -- dropping the 13 + 9 edge spots instead moves the target CV
    from 0.791 to 0.813 and breaks the match with the recorded numbers.

    Returns (slice_y, slice_x, d2).
    """
    h, w = shape
    half = int(np.ceil(r_out)) + 1
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    return slice(y0, y1), slice(x0, x1), (xx - x) ** 2 + (yy - y) ** 2


def aperture_photometry(movie, xy, r_signal=None, r_in=None, r_out=None,
                        frame_step=None):
    """Per-spot brightness traces measured directly at the given coordinates.

    No detector involved -- section 4d's recipe. For every spot, on every sampled
    frame, two quantities:

        bgsub      = mean(disc r <= r_signal) - median(annulus r_in..r_out)
        aperture1px = mean(disc r <= 1), with NO background subtraction

    Each trace is then reduced per spot by its MEAN (and its CV = std/mean).

    Both of those choices were recovered by reproducing section 4d rather than
    assumed, and both matter:

      * the mean, not the median, of the trace. A blinking molecule is dark in
        most sampled frames, so its median sits near the dark level while its mean
        carries the emission. Using the median gives target 51.1 / off-target 7.2
        and AUC 0.786; the mean gives 60.6 / 18.9 and AUC 0.909, matching the
        recorded values to three digits.
      * "raw 1-px" in section 4d's table is a radius-1 aperture, not literally one
        pixel. The single pixel at round(x), round(y) gives 232.7 / 167.4 with CV
        0.379 / 0.329; the r <= 1 disc gives 220.5 / 164.9 with CV 0.305 / 0.258
        and AUC 0.899 / 0.580 -- all four exact.

    Returns arrays aligned with the input order.
    """
    p = PHOTOMETRY
    r_signal = p["r_signal"] if r_signal is None else r_signal
    r_in = p["r_in"] if r_in is None else r_in
    r_out = p["r_out"] if r_out is None else r_out
    frame_step = p["frame_step"] if frame_step is None else frame_step

    movie = np.asarray(movie)
    n_frames, h, w = movie.shape
    frames = np.arange(0, n_frames, frame_step)
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(pts)

    stats = {k: np.full(n, np.nan) for k in
             ("bgsub_intensity", "bgsub_cv", "bgsub_trace_median",
              "aperture1px_intensity", "aperture1px_cv")}
    n_clipped = 0

    for si, (x, y) in enumerate(pts):
        sy, sx, d2 = _spot_window((h, w), x, y, r_out)
        disc = d2 <= r_signal ** 2
        ann = (d2 >= r_in ** 2) & (d2 <= r_out ** 2)
        disc1 = d2 <= 1.0
        if not disc.any() or not ann.any() or not disc1.any():
            continue
        if (sy.stop - sy.start) * (sx.stop - sx.start) < (2 * (int(np.ceil(r_out)) + 1) + 1) ** 2:
            n_clipped += 1
        # Slice all sampled frames at once: 334 windows instead of 334 x 101.
        sub = movie[frames, sy, sx].astype(np.float64)
        bgsub = sub[:, disc].mean(axis=1) - np.median(sub[:, ann], axis=1)
        ap1 = sub[:, disc1].mean(axis=1)
        for key, trace in (("bgsub", bgsub), ("aperture1px", ap1)):
            m = float(trace.mean())
            stats[f"{key}_intensity"][si] = m
            stats[f"{key}_cv"][si] = float(trace.std() / abs(m)) if abs(m) > 1e-12 else np.nan
        stats["bgsub_trace_median"][si] = float(np.median(bgsub))

    return {
        "n_spots": int(n),
        "n_edge_clipped": int(n_clipped),
        "n_frames_sampled": int(len(frames)),
        "recipe": {"r_signal": r_signal, "r_in": r_in, "r_out": r_out,
                   "frame_step": int(frame_step),
                   "trace_reduction": "mean over sampled frames",
                   "edge_handling": "window clipped to the frame, spot kept"},
        **stats,
    }


def auc(a, b):
    """Mann-Whitney AUC reported as max(auc, 1 - auc), per section 4d's recipe.

    Direction-free by construction, so it measures separability only. Report the
    medians alongside it whenever the direction matters (gate T3 requires the
    off-target population to be the FAINTER one).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    value = u / (a.size * b.size)
    return float(max(value, 1.0 - value))


def quartiles(values):
    """{median, q1, q3, n} over the finite entries. Empty input -> NaNs."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan"), "n": 0}
    return {"median": float(np.median(v)),
            "q1": float(np.percentile(v, 25)),
            "q3": float(np.percentile(v, 75)),
            "n": int(v.size)}


# The four rows of section 4d's table, in order, with the values they must match
# on the real movie. Kept here so the measurement code, the reference file and the
# acceptance table all name them identically.
PHOTOMETRY_KEYS = ("bgsub_intensity", "bgsub_cv", "aperture1px_intensity", "aperture1px_cv")
PHOTOMETRY_LABELS = {
    "bgsub_intensity": "median bg-subtracted intensity",
    "bgsub_cv": "CV of that trace",
    "aperture1px_intensity": "raw 1-px value (no bg subtraction)",
    "aperture1px_cv": "CV of raw 1-px trace",
}


def photometry_report(movie, targets_xy, offtargets_xy):
    """Section 4d's whole table: per-class quartiles plus the four AUCs."""
    tgt = aperture_photometry(movie, targets_xy) if len(targets_xy) else None
    off = aperture_photometry(movie, offtargets_xy) if len(offtargets_xy) else None

    out = {"recipe": (tgt or off)["recipe"] if (tgt or off) else PHOTOMETRY,
           "target": {}, "off_target": {}, "auc": {}, "per_spot_values": {}}
    for key in PHOTOMETRY_KEYS:
        out["target"][key] = quartiles(tgt[key]) if tgt else quartiles([])
        out["off_target"][key] = quartiles(off[key]) if off else quartiles([])
        out["auc"][key] = auc(tgt[key], off[key]) if (tgt and off) else float("nan")
        # Kept so the harness can plot distributions rather than only medians.
        out["per_spot_values"][key] = {
            "target": _round_list(tgt[key]) if tgt else [],
            "off_target": _round_list(off[key]) if off else [],
        }
    for label, res in (("target", tgt), ("off_target", off)):
        if res:
            out[label]["n_spots"] = res["n_spots"]
            out[label]["n_edge_clipped"] = res["n_edge_clipped"]
            out[label]["n_frames_sampled"] = res["n_frames_sampled"]
    return out


def _round_list(arr, nd=4):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(arr)]


# ============================================================================
# PSF WIDTH  (requirement R1 / gate T1)
# ============================================================================

PSF_WINDOW_HALF = 7        # 15x15 px window: 4-5 sigma out for a diffraction spot
PSF_BG_ANNULUS = (5.0, 7.0)
PSF_CORE_RADIUS = 4.0      # radius the second moment is taken over


def psf_width(movie, xy, half=PSF_WINDOW_HALF, frame_step=None,
              bg_annulus=PSF_BG_ANNULUS, r_core=PSF_CORE_RADIUS):
    """Measure the physical width of the spots at the given coordinates.

    THIS, not a detector sigma sweep, is what gate T1 should compare. Measured on
    the real movie, blob_log's response is identical for every max_sigma from 0.8
    to 3.0 once min_sigma is small, so the detector's "preferred window" is not
    identifiable from a sweep -- and a sweep-based T1 passes the BachelorThesis
    simulator, whose PSF section 3b calls a mismatch. A direct width measurement is
    identifiable, physical, and independent of the detector's parameterisation.

    Method. For each spot, a (2*half+1)^2 window is averaged over the sampled
    frames, the median of the background annulus is subtracted, and the width is
    taken three ways:

      per_spot_sigma          intensity-weighted second moment inside r_core,
                              using <r^2> = 2*sigma^2 for a 2-D Gaussian
      stacked_second_moment   the same estimator applied to the mean of all spot
                              windows -- far less noise-sensitive
      stacked_gaussfit        log-linear least squares of ln(v) against r^2 on the
                              stacked image, i.e. a Gaussian fit

    Spots whose window would fall off the frame are skipped, not clipped: a clipped
    window biases a second moment toward zero. (Section 4d's photometry does the
    opposite -- it clips -- because that is what reproduces the recorded numbers
    there. The two are different measurements with different failure modes.)

    Measured values, for orientation: the real movie's targets give 1.478 (stacked
    second moment) / 1.718 (Gaussian fit) / 1.473 (per-spot median), and an
    independent fit-free FWHM of the radial profile gives 3.65 px, i.e. sigma 1.55.
    The BachelorThesis simulator, configured with gauss_sigma = 1.5, measures
    1.424 / 1.517 / 1.423 and FWHM 3.54 px. The real PSF is therefore about 1.5 px
    -- see CALIBRATION.md, because this contradicts spec section 3b's first row.
    """
    frame_step = PHOTOMETRY["frame_step"] if frame_step is None else frame_step
    movie = np.asarray(movie)
    n_frames, h, w = movie.shape
    frames = np.arange(0, n_frames, frame_step)
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)

    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    d2 = (xx ** 2 + yy ** 2).astype(float)
    ann = (d2 >= bg_annulus[0] ** 2) & (d2 <= bg_annulus[1] ** 2)
    core = d2 <= r_core ** 2

    stack = np.zeros_like(d2)
    per_spot, n_used, n_skipped = [], 0, 0
    for x, y in pts:
        cx, cy = int(round(x)), int(round(y))
        if cx - half < 0 or cy - half < 0 or cx + half + 1 > w or cy + half + 1 > h:
            n_skipped += 1
            continue
        win = movie[frames, cy - half:cy + half + 1,
                    cx - half:cx + half + 1].astype(np.float64).mean(axis=0)
        win = win - np.median(win[ann])
        stack += win
        n_used += 1
        s = _second_moment_sigma(win, d2, core)
        if np.isfinite(s):
            per_spot.append(s)

    if n_used == 0:
        return {"n_spots_used": 0, "n_spots_skipped": int(n_skipped),
                "stacked_second_moment_sigma": float("nan"),
                "stacked_gaussfit_sigma": float("nan"),
                "per_spot_sigma": {"median": float("nan"), "q1": float("nan"),
                                   "q3": float("nan"), "n": 0},
                "per_spot_values": [], "radial_profile": []}

    stack /= n_used
    return {
        "n_spots_used": int(n_used),
        "n_spots_skipped": int(n_skipped),
        "n_frames_sampled": int(len(frames)),
        "window_half_px": int(half),
        "stacked_second_moment_sigma": float(_second_moment_sigma(stack, d2, core)),
        "stacked_gaussfit_sigma": float(_gaussfit_sigma(stack, d2, r_core)),
        "per_spot_sigma": quartiles(per_spot),
        "per_spot_values": _round_list(per_spot),
        # Normalised radial profile, for the side-by-side plot and for an
        # independent fit-free FWHM check.
        "radial_profile": _spot_radial_profile(stack, d2),
    }


def _second_moment_sigma(img, d2, core):
    wgt = np.clip(img, 0, None) * core
    s = wgt.sum()
    if s <= 0:
        return float("nan")
    return float(np.sqrt(max((wgt * d2).sum() / s / 2.0, 1e-9)))


def _gaussfit_sigma(img, d2, r_core):
    """sigma from a log-linear least-squares fit of ln(v) against r^2."""
    peak = img[img.shape[0] // 2, img.shape[1] // 2]
    m = (d2 <= r_core ** 2) & (img > 0.05 * peak) & (img > 0)
    if m.sum() < 6:
        return float("nan")
    A = np.column_stack([np.ones(int(m.sum())), d2[m]])
    coef, *_ = np.linalg.lstsq(A, np.log(img[m]), rcond=None)
    return float(np.sqrt(-1.0 / (2 * coef[1]))) if coef[1] < 0 else float("nan")


def _spot_radial_profile(img, d2, rmax=5.0, step=0.5):
    """Peak-normalised radial profile, as [[r, value], ...]."""
    r = np.sqrt(d2).ravel()
    peak = img[img.shape[0] // 2, img.shape[1] // 2]
    v = (img / peak).ravel() if peak else img.ravel()
    out = []
    for rr in np.arange(0.0, rmax + 1e-9, step):
        m = (r >= rr - step / 2) & (r < rr + step / 2)
        if m.any():
            out.append([round(float(rr), 2), round(float(v[m].mean()), 5)])
    return out


def fwhm_sigma_from_profile(profile):
    """sigma implied by the half-maximum crossing of a radial profile.

    Fit-free cross-check on psf_width()'s two model-based estimators: read the
    radius where the profile crosses 0.5, double it for the FWHM, and divide by
    2*sqrt(2*ln 2). Returns NaN if the profile never crosses.
    """
    prof = [(float(r), float(v)) for r, v in profile if np.isfinite(v)]
    if len(prof) < 2:
        return float("nan")
    rs = np.array([p[0] for p in prof])
    vs = np.array([p[1] for p in prof])
    below = np.where(vs < 0.5)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = int(below[0])
    hwhm = rs[i - 1] + (vs[i - 1] - 0.5) / (vs[i - 1] - vs[i]) * (rs[i] - rs[i - 1])
    return float(2 * hwhm / 2.354820045)


def psf_report(movie, targets_xy, offtargets_xy):
    """Requirement R1 / gate T1: measured spot width, per class."""
    out = {}
    for label, pts in (("target", targets_xy), ("off_target", offtargets_xy)):
        r = psf_width(movie, pts) if len(pts) else psf_width(movie, [])
        r["fwhm_sigma"] = fwhm_sigma_from_profile(r["radial_profile"])
        out[label] = r
    return out


# ============================================================================
# SIGNAL-TO-NOISE  (section 4b)
# ============================================================================

# SNR estimator variants. All are peak-signal-over-noise in the spirit of
# smTIRF_simulation.py:calculate_snr; they differ in what plays the role of the
# noise term. See spot_snr() for why more than one is reported.
SNR_VARIANTS = ("peak_over_annulus_std", "peak_over_sqrt_peak",
                "peak_raw_over_annulus_std", "mean_bgsub_over_annulus_std")
SNR_PRIMARY = "peak_over_annulus_std"


def spot_snr(movie, xy, frame_step=None):
    """Per-spot SNR under several peak-signal-over-noise definitions.

    The analytic definition documented in smTIRF_simulation.py:calculate_snr is

        SNR = S_peak / sqrt(S_peak + B + sigma_read^2)

    which is peak signal divided by the standard deviation of that pixel in PHOTON
    units. It is a function of simulator parameters (photon_scale, background_level,
    read_noise) and cannot be evaluated on a real movie, whose gain and read noise
    are unknown -- so section 4b's instruction to re-derive "with that definition"
    has no literal image-domain form. Rather than silently substitute one, this
    reports four variants and marks the primary:

      peak_over_annulus_std        (peak(disc) - median(annulus)) / std(annulus)
      peak_over_sqrt_peak          (peak(disc) - median(annulus)) / sqrt(peak)
                                   -- the closest analogue of the analytic form,
                                      taking ADU as photons
      peak_raw_over_annulus_std    peak(disc) / std(annulus), no background removed
      mean_bgsub_over_annulus_std  mean(disc) - median(annulus), over std(annulus)

    On the real movie these give roughly 4.2, 8.0, 9.0 and 2.0 respectively; none
    reproduces the "median SNR ~ 11" quoted in section 4b, which traces to a
    bracketed placeholder in METHODS_parameter_optimization.md whose definition was
    never written down. No section 7 gate depends on SNR, so the discrepancy is
    recorded rather than resolved -- see CALIBRATION.md. What matters for the
    simulator is that the SAME estimator is applied to simulated and real movies,
    which it is.

    Returns {variant: array aligned with xy}, each per-spot value the median over
    the sampled frames.
    """
    p = PHOTOMETRY
    frame_step = p["frame_step"] if frame_step is None else frame_step
    movie = np.asarray(movie)
    n_frames, h, w = movie.shape
    frames = np.arange(0, n_frames, frame_step)
    pts = np.asarray(xy, dtype=float).reshape(-1, 2)

    out = {k: np.full(len(pts), np.nan) for k in SNR_VARIANTS}
    for si, (x, y) in enumerate(pts):
        sy, sx, d2 = _spot_window((h, w), x, y, p["r_out"])
        disc = d2 <= p["r_signal"] ** 2
        ann = (d2 >= p["r_in"] ** 2) & (d2 <= p["r_out"] ** 2)
        if not disc.any() or not ann.any():
            continue
        sub = movie[frames, sy, sx].astype(np.float64)
        peak = sub[:, disc].max(axis=1)
        dmean = sub[:, disc].mean(axis=1)
        bgm = np.median(sub[:, ann], axis=1)
        bgs = sub[:, ann].std(axis=1)
        ok = bgs > 0
        if not ok.any():
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            out["peak_over_annulus_std"][si] = np.median(((peak - bgm) / bgs)[ok])
            out["peak_over_sqrt_peak"][si] = np.median(
                (peak - bgm) / np.sqrt(np.maximum(peak, 1.0)))
            out["peak_raw_over_annulus_std"][si] = np.median((peak / bgs)[ok])
            out["mean_bgsub_over_annulus_std"][si] = np.median(((dmean - bgm) / bgs)[ok])
    return out


def snr_report(movie, targets_xy, offtargets_xy):
    """Section 4b: every SNR variant, per class and pooled."""
    t = spot_snr(movie, targets_xy) if len(targets_xy) else None
    o = spot_snr(movie, offtargets_xy) if len(offtargets_xy) else None
    out = {"primary_variant": SNR_PRIMARY,
           "definitions": spot_snr.__doc__.strip().splitlines()[0],
           "target": {}, "off_target": {}, "all": {}}
    for k in SNR_VARIANTS:
        out["target"][k] = quartiles(t[k]) if t else quartiles([])
        out["off_target"][k] = quartiles(o[k]) if o else quartiles([])
        pooled = np.concatenate([a[k] for a in (t, o) if a is not None]) if (t or o) else []
        out["all"][k] = quartiles(pooled)
    return out


# ============================================================================
# BACKGROUND STRUCTURE  (section 4g / requirement R4)
# ============================================================================

def background_stats(movie, sigma=None, n_frames=10):
    """Statistics of the sigma=8 Gaussian low-pass field the detector removes.

    Section 4g asks for four things and this returns all of them:
      1. the field itself, per sampled frame (not returned, but summarised);
      2. mean / std / min / max and the relative swing (max - min) / mean;
      3. the spatial autocorrelation length, and whether the field drifts in time;
      4. comparable numbers for any other movie -- same function, so a BT-sim
         frame is measured identically.

    autocorr_length_px is the radius at which the radially averaged, mean-removed
    autocorrelation of the field first falls below 1/e. It is a length scale for
    "how broad is the unevenness", which is what the simulator has to reproduce.
    """
    sigma = REFERENCE_BANDPASS_BG_SIGMA if sigma is None else sigma
    movie = np.asarray(movie)
    n = movie.shape[0]
    idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)

    per_frame, fields = [], []
    for t in idx:
        bg = ndimage.gaussian_filter(movie[t].astype(np.float64), sigma)
        fields.append(bg)
        per_frame.append({
            "frame": int(t),
            "mean": float(bg.mean()), "std": float(bg.std()),
            "min": float(bg.min()), "max": float(bg.max()),
            "relative_swing": float((bg.max() - bg.min()) / bg.mean()) if bg.mean() else float("nan"),
        })

    acl = [_autocorr_length(f) for f in fields]

    # Temporal behaviour: how much the per-frame mean moves, and how similar the
    # spatial pattern stays. A static field correlates ~1.0 with frame 0; a
    # drifting one decays.
    means = np.array([d["mean"] for d in per_frame])
    flat0 = (fields[0] - fields[0].mean()).ravel()
    corr = []
    for f in fields[1:]:
        flat = (f - f.mean()).ravel()
        denom = np.linalg.norm(flat0) * np.linalg.norm(flat)
        corr.append(float(flat0 @ flat / denom) if denom else float("nan"))

    return {
        "sigma": float(sigma),
        "frames_sampled": [int(i) for i in idx],
        "per_frame": per_frame,
        "mean": float(np.mean([d["mean"] for d in per_frame])),
        "std": float(np.mean([d["std"] for d in per_frame])),
        "min": float(np.min([d["min"] for d in per_frame])),
        "max": float(np.max([d["max"] for d in per_frame])),
        "relative_swing": float(np.mean([d["relative_swing"] for d in per_frame])),
        "autocorr_length_px": float(np.mean(acl)),
        "autocorr_length_px_per_frame": [round(float(v), 3) for v in acl],
        "temporal_mean_drift_std": float(means.std()),
        "temporal_mean_range": [float(means.min()), float(means.max())],
        "spatial_pattern_corr_with_frame0": {
            "min": float(np.min(corr)) if corr else float("nan"),
            "mean": float(np.mean(corr)) if corr else float("nan"),
        },
        "radial_autocorr": _radial_autocorr_curve(fields[0]),
    }


def _radial_profile(field2d):
    """Radially averaged profile of a 2-D field about its centre pixel."""
    h, w = field2d.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    nbins = min(cy, cx)
    total = np.bincount(r.ravel(), field2d.ravel(), minlength=nbins + 1)[: nbins + 1]
    count = np.bincount(r.ravel(), minlength=nbins + 1)[: nbins + 1]
    return total / np.maximum(count, 1)


def _autocorr2d(field):
    f = field - field.mean()
    spec = np.fft.rfft2(f)
    ac = np.fft.irfft2(spec * np.conj(spec), s=f.shape).real
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    return ac / peak if peak > 0 else ac


def _autocorr_length(field):
    """Radius (px) where the radial autocorrelation first drops below 1/e."""
    prof = _radial_profile(_autocorr2d(field))
    below = np.where(prof < 1.0 / np.e)[0]
    if below.size == 0:
        return float(len(prof) - 1)
    i = int(below[0])
    if i == 0:
        return 0.0
    # Linear interpolation between the last point above 1/e and the first below.
    y0, y1 = prof[i - 1], prof[i]
    t = (y0 - 1.0 / np.e) / (y0 - y1) if y0 != y1 else 0.0
    return float((i - 1) + t)


def _radial_autocorr_curve(field, max_r=40):
    prof = _radial_profile(_autocorr2d(field))
    return [round(float(v), 5) for v in prof[: max_r + 1]]


# ============================================================================
# BLOB DENSITY VS THRESHOLD  (gate T2)
# ============================================================================

def blob_count(frame, min_sigma, max_sigma, threshold, use_bandpass=True,
               bg_sigma=None):
    """One frame through SCP's exact blob-detection front end. No tracking.

    Ported from experiments/sim_vs_real_transfer_comparison/run_transfer.py's
    _blob_count so the density numbers already on record stay comparable: same
    sigma=0.5 pre-smooth, same bandpass, same per-frame min-max normalisation
    (trap 3), same overlap and log_scale.
    """
    bg_sigma = REFERENCE_BANDPASS_BG_SIGMA if bg_sigma is None else bg_sigma
    f = ndimage.gaussian_filter(frame.astype(np.float64), sigma=0.5)
    if use_bandpass:
        f = np.clip(f - ndimage.gaussian_filter(f, bg_sigma), 0, None)
    f = (f - f.min()) / (f.max() - f.min() + 1e-10)
    return len(blob_log(f, min_sigma=min_sigma, max_sigma=max_sigma,
                        threshold=threshold, overlap=0.5, log_scale=False))


def mean_density(movie_or_path, min_sigma, max_sigma, threshold,
                 use_bandpass=True, bg_sigma=None, n_frames=10):
    """Mean blobs per 1000 px over evenly sampled frames.

    Density, never raw count (trap 3): a 256x256 sim frame and a 256x512 real
    frame with equal blobs-per-frame differ 2x in areal density.
    """
    if isinstance(movie_or_path, (str, os.PathLike, Path)):
        with tifffile.TiffFile(str(movie_or_path)) as tf:
            n = tf.series[0].shape[0]
            idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)
            counts, px = [], None
            for i in idx:
                frame = np.asarray(tf.asarray(key=int(i)))
                px = frame.size
                counts.append(blob_count(frame, min_sigma, max_sigma, threshold,
                                         use_bandpass, bg_sigma))
    else:
        movie = np.asarray(movie_or_path)
        n = movie.shape[0]
        idx = np.linspace(0, n - 1, min(n_frames, n)).astype(int)
        counts, px = [], movie[0].size
        for i in idx:
            counts.append(blob_count(movie[i], min_sigma, max_sigma, threshold,
                                     use_bandpass, bg_sigma))
    mean_count = float(np.mean(counts))
    return mean_count / (px / 1000.0), mean_count, int(px)


# Thresholds gate T2 compares at, plus 0.05 for continuity with the historical
# numbers quoted in section 7-T2.
T2_THRESHOLDS = (0.05, 0.25, 0.32, 0.40, 0.50)


def density_vs_threshold(movie_or_path, min_sigma, max_sigma,
                         thresholds=T2_THRESHOLDS, n_frames=10, bg_sigma=None):
    """Blob-count-vs-threshold curve, as densities per 1000 px."""
    out = []
    for thr in thresholds:
        d, c, px = mean_density(movie_or_path, min_sigma, max_sigma, thr,
                                use_bandpass=True, bg_sigma=bg_sigma,
                                n_frames=n_frames)
        out.append({"threshold": float(thr), "density_per_1000px": round(d, 4),
                    "blobs_per_frame": round(c, 3), "pixels": px})
    return out


# ============================================================================
# PREFERRED SIGMA WINDOW  (gate T1)
# ============================================================================

# Scales swept by sigma_response(). Fine below 0.6 because that is where the real
# movie's spots live, and out to 2.6 so a BT-sim movie (PSF sigma 1.5, tuned window
# 1.29-2.29) lands well inside the grid rather than off its edge.
SIGMA_SCALES = tuple(np.round(np.concatenate([
    np.arange(0.15, 0.60, 0.05),
    np.arange(0.60, 2.65, 0.10),
]), 2).tolist())


def sigma_response(movie_or_path, targets_xy, threshold=None, n_frames=8,
                   scales=SIGMA_SCALES, bg_sigma=None):
    """At which LoG scales does this movie's spot population actually respond?

    Gate T1 asks whether the detector's preferred sigma window agrees between sim
    and real. Sweeping WINDOWS cannot answer that: on the real movie, blob_log
    returns an identical 312 blobs/frame for every max_sigma from 0.8 to 3.0 once
    min_sigma is 0.2, because every response comes from the smallest scale in the
    window. Window width is therefore not identifiable, and a window sweep's argmax
    just reports wherever recall and precision happen to cross.

    Sweeping SINGLE scales is identifiable. For each sigma, blob_log runs with a
    hairline window [sigma, sigma + 0.02], its blobs are matched to the annotated
    TARGET coordinates with the project's greedy 1:1 matcher at MATCH_MARGIN = 3 px,
    and the mean per-frame F1 is recorded. The resulting curve is a direct measure
    of spot scale, and it is what T1 compares between movies.

    Reported: peak_sigma, the half-maximum support of the curve, and the curve
    itself. On the real movie the response is flat from 0.15 to 0.35 and collapses
    by 0.50 (F1 0.49 -> 0.20 -> 0.03), so its half-max support is 0.15-0.45 -- note
    that the tuned window's LOWER bound of 0.45 sits exactly at that collapse edge,
    while its upper bound of 1.27 is inert (F1 ~ 0.036 there). Tracking is excluded
    on purpose: this measures the front end, independent of the track-length filter.

    Per-frame recall is modest by construction -- a blinking molecule is dark in
    most frames, so most annotations are legitimately absent from any single frame
    -- but the criterion is applied identically to every movie.
    """
    threshold = REFERENCE_DETECTOR["intensity_threshold"] if threshold is None else threshold
    bg_sigma = REFERENCE_BANDPASS_BG_SIGMA if bg_sigma is None else bg_sigma

    if isinstance(movie_or_path, (str, os.PathLike, Path)):
        movie = load_movie(movie_or_path)
    else:
        movie = np.asarray(movie_or_path)
    idx = np.linspace(0, movie.shape[0] - 1, min(n_frames, movie.shape[0])).astype(int)

    # Preprocess each sampled frame once; the sweep only re-runs blob_log.
    prepped = []
    for t in idx:
        f = ndimage.gaussian_filter(movie[t].astype(np.float64), sigma=0.5)
        f = np.clip(f - ndimage.gaussian_filter(f, bg_sigma), 0, None)
        prepped.append((f - f.min()) / (f.max() - f.min() + 1e-10))

    targets = list(targets_xy)
    curve = []
    for s in scales:
        f1s, matched, counts = [], [], []
        for f in prepped:
            blobs = blob_log(f, min_sigma=s, max_sigma=s + 0.02, threshold=threshold,
                             overlap=0.5, log_scale=False)
            dets = [(float(b[1]), float(b[0])) for b in blobs]   # (y,x,r) -> (x,y)
            m = pm.compute_metrics(targets, [], dets)
            f1s.append(m["fbeta"])
            matched.append(m["true_positives"])
            counts.append(len(dets))
        curve.append({
            "sigma": float(s),
            "mean_f1": float(np.mean(f1s)),
            "mean_matched": float(np.mean(matched)),
            "mean_blobs_per_frame": float(np.mean(counts)),
        })

    f1 = np.array([c["mean_f1"] for c in curve])
    sig = np.array([c["sigma"] for c in curve])
    peak_i = int(f1.argmax())
    support = sig[f1 >= f1[peak_i] / 2.0] if f1[peak_i] > 0 else sig[:0]
    return {
        "threshold": float(threshold),
        "frames_sampled": [int(i) for i in idx],
        "criterion": "per-single-scale mean per-frame F1 of blobs vs annotated "
                     "targets, greedy 1:1 match at MATCH_MARGIN=3 px, no tracking",
        "peak_sigma": float(sig[peak_i]),
        "peak_f1": float(f1[peak_i]),
        "half_max_low": float(support.min()) if support.size else float("nan"),
        "half_max_high": float(support.max()) if support.size else float("nan"),
        "curve": curve,
    }


# ============================================================================
# DETECTOR-MEDIATED TRACK DURATION  (section 4e / gate T4)
# ============================================================================

def run_detector(tif_path, params=None, output_json=None, quiet=True):
    """Run SCP once through the parametrizer's run_detection(), return its events.

    Events keep every field SCP exports, so 'duration' is available without a
    second detection run. Returns (events, ok).
    """
    params = dict(REFERENCE_DETECTOR) if params is None else dict(params)
    apply_reference_detector_globals()
    if output_json is None:
        output_json = str(Path(tif_path).with_suffix("").as_posix() + "_scpevents.json")

    buf = io.StringIO()
    with redirect_stdout(buf) if quiet else _null_ctx():
        ok, _ = pm.run_detection(params, str(tif_path), str(output_json))
    if not ok or not os.path.exists(output_json):
        return [], False
    with open(output_json) as f:
        return json.load(f), True


class _null_ctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def track_durations_by_class(events, targets_xy, offtargets_xy, box=3):
    """Bin detected events by which annotation class they land on.

    Section 4e's recipe: a 3 px BOX (Chebyshev distance, matching MATCH_MARGIN's
    semantics in the project's matcher). A target claim wins over an off-target
    claim when both are in range, since the target is the population of interest.
    Events near neither are 'spurious'.

    The result is explicitly detector-mediated (trap 7): it is what SCP reports,
    not a ground-truth on-time.
    """
    tgt = np.asarray(targets_xy, dtype=float).reshape(-1, 2)
    off = np.asarray(offtargets_xy, dtype=float).reshape(-1, 2)
    bins = {"target": [], "off_target": [], "spurious": []}

    for ev in events:
        x, y = float(ev["x"]), float(ev["y"])
        dur = int(ev["duration"])
        if tgt.size and np.any((np.abs(tgt[:, 0] - x) <= box) & (np.abs(tgt[:, 1] - y) <= box)):
            bins["target"].append(dur)
        elif off.size and np.any((np.abs(off[:, 0] - x) <= box) & (np.abs(off[:, 1] - y) <= box)):
            bins["off_target"].append(dur)
        else:
            bins["spurious"].append(dur)

    out = {"box_px": box, "n_events": len(events)}
    for key, vals in bins.items():
        v = np.asarray(vals, dtype=float)
        out[key] = {
            "n": int(v.size),
            "median": float(np.median(v)) if v.size else float("nan"),
            "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v.size else [float("nan")] * 2,
            "durations": [int(d) for d in vals],
        }
    return out


# ============================================================================
# THE WHOLE MEASUREMENT SUITE FOR ONE MOVIE
# ============================================================================

def measure_movie(tif_path, targets_xy, offtargets_xy, movie=None,
                  run_detector_pass=True, n_frames_background=12,
                  n_frames_density=10, n_frames_sigma=8, events_json=None,
                  bands=(("val_x0_256", (0, 256)), ("test_x256_512", (256, 512))),
                  label_band=None):
    """Every section-4 and section-7 measurement for one movie, in one dict.

    The real movie and every simulated movie go through THIS function, so the two
    sides of a comparison can never drift apart by assembling their statistics
    differently -- only the inputs differ. measure_real_movie.py and
    validate_against_real.py are both thin wrappers around it.

    run_detector_pass=False skips the SCP run (the only slow step) and omits the
    track_duration block, which the harness then reports as a missing target
    rather than a pass.
    """
    movie = load_movie(tif_path) if movie is None else np.asarray(movie)
    n_frames, height, width = movie.shape
    targets, offtargets = list(targets_xy), list(offtargets_xy)
    area_1000px = (height * width) / 1000.0
    d = REFERENCE_DETECTOR

    out = {
        "geometry": {
            "n_frames": int(n_frames), "height": int(height), "width": int(width),
            "dtype": str(movie.dtype), "bit_depth": int(movie.dtype.itemsize * 8),
            "intensity_min": float(movie.min()), "intensity_max": float(movie.max()),
            "intensity_mean": float(movie.mean()),
            "n_annotated": len(targets) + len(offtargets),
            "n_target": len(targets), "n_off_target": len(offtargets),
            "density_per_1000px": round((len(targets) + len(offtargets)) / area_1000px, 4),
            "target_density_per_1000px": round(len(targets) / area_1000px, 4),
            "target_to_off_target_ratio": round(len(targets) / max(len(offtargets), 1), 4),
        },
        "psf": psf_report(movie, targets, offtargets),
        "snr": snr_report(movie, targets, offtargets),
        "nn_distance": {"target": nn_summary(targets),
                        "all": nn_summary(targets + offtargets)},
        "photometry": photometry_report(movie, targets, offtargets),
        "background": background_stats(movie, n_frames=n_frames_background),
        "blob_density": {
            "reference_sigmas": density_vs_threshold(
                movie, d["min_sigma"], d["max_sigma"], n_frames=n_frames_density),
            # The BT simulator's own tuned window, kept so the historical numbers
            # quoted in spec 7-T2 stay checkable on any movie.
            "bt_sim_sigmas": density_vs_threshold(
                movie, 1.29, 2.29, n_frames=n_frames_density),
            "n_frames_sampled": n_frames_density,
        },
        "sigma_response": sigma_response(movie, targets, n_frames=n_frames_sigma),
    }

    if run_detector_pass:
        events, ok = run_detector(tif_path, output_json=events_json)
        if ok:
            # When the labels are restricted to a band, the events must be too.
            # Otherwise every detection outside the band -- including all the real
            # ones -- is binned as 'spurious', which is not what that class means:
            # measured, it turned 14 spurious events of median 31 frames into 143 of
            # median 114, since it swept up the whole test band's true detections.
            if label_band is not None:
                lo, hi = label_band
                events = [e for e in events if lo <= float(e["x"]) < hi]
            out["track_duration"] = track_durations_by_class(events, targets, offtargets)
            dets = [(float(e["x"]), float(e["y"])) for e in events]
            out["reference_detector_performance"] = {
                "whole_movie": pm.compute_metrics(targets, offtargets, dets),
                "bands": {
                    name: pm.compute_metrics(
                        [s for s in targets if band[0] <= s[0] < band[1]],
                        [s for s in offtargets if band[0] <= s[0] < band[1]],
                        [s for s in dets if band[0] <= s[0] < band[1]])
                    for name, band in bands},
            }
        else:
            out["detector_error"] = "SCP run failed or produced no output"
    return out


# ============================================================================
# JSON HELPERS
# ============================================================================

def to_jsonable(obj):
    """Recursively convert numpy scalars/arrays so json.dump accepts the result."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    return obj


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    return path
