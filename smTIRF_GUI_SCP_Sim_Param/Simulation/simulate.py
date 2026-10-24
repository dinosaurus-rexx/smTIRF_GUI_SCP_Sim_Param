"""
Real-calibrated smTIRF movie simulator (SIMULATOR_SPEC.md sections 5, 6).

One movie per call, from a YAML config and a seed. Emitters get their own sub-pixel
volume-normalised Gaussian PSF (R1); their amplitude varies over time and their
size never does (R2); the background is a structured field, not a constant (R4);
placement is hard-core rejection sampling at a 3.5 px floor (R5); everything is
seeded through explicitly passed generators (R6).

    from simulate import load_config, simulate
    cfg = load_config("config/real_calibrated.yaml")
    movie, spots, meta = simulate(cfg, seed=0)     # movie is (T, H, W) uint16

What was calibrated, and against what (details in CALIBRATION.md):

    psf_sigma            1.55   the real movie's spots measure sigma 1.478
                                (stacked second moment) / 1.550 (radial-profile
                                FWHM). NOT "well below 1.5" as spec R1 predicts --
                                that prediction confused the detector's tuned LoG
                                sigma range with the physical spot width. See
                                measurements.psf_width.
    gain_adu_per_photon  5.0    frame-to-frame difference noise on the real movie
                                is 27.4 ADU at a mean of 147, i.e. variance/mean
                                about 5.1, so ADU are not photons.
    background           see background.py -- every constant there is measured.
    amplitudes / duty    from section 4d's photometry: a two-level trace of duty d
                                has CV = sqrt((1-d)/d), so the target CV of 0.791
                                implies a duty of about 0.62 and the off-target CV
                                of 1.596 implies about 0.28. Combined with the
                                bg-subtracted medians (60.6 and 18.9) that fixes
                                both peak amplitudes. Final values come from a
                                numerical fit -- see calibrate.py.

Deliberately NOT modelled: the off-target class is not N summed fluorophores. That
model predicts aggregates brighter than targets and the real ones are dimmer (spec
section 4d's warning, trap 6). Here an off-target is a fainter, more intermittent,
shorter-lived emitter, which reproduces the faint mean, the higher relative
variability and the shorter detector tracks at once.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

import background as bg
import photophysics as pp
import placement as pl

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "real_calibrated.yaml"

# Class names and the ground-truth class ids of section 6c. Order matters: it is
# the order coordinates are emitted in, so the .npy outputs are grouped by class.
CLASS_ORDER = ("target", "off_target", "moving", "unannotated")
CLASS_IDS = {"target": 0, "off_target": 1, "moving": 2}

# Classes that reach the label JSONs of section 6a.
LABELLED_CLASSES = ("target", "off_target")

# `unannotated` appears in NO ground-truth output -- not the label JSONs, not
# coord.npy, not class.npy -- which is why it has no entry in CLASS_IDS.
#
# It is not a fudge to soften precision; it is a measured feature of the real movie.
# realdata/coord.npy holds exactly 334 spots, but the movie plainly contains more
# emitters than a human marked: in regions at least 6 px from every annotation, the
# sigma=8-highpassed frame reaches a 99.9th percentile of 85.6 ADU against a
# simulated 57.2 with the same noise (frame-difference noise matches to 0.4%), and
# the reference detector finds 26.5 unexplained blobs per frame against a simulated
# 6.4. Those are localised bright features at unannotated positions -- dim
# non-specific emitters, which any real preparation has.
#
# Leaving them out is what made the simulated blob density come out at 0.53x the
# real one at every threshold, i.e. gate T2's failure. Including them, unlabelled,
# reproduces the real annotation situation exactly: a detection on one is scored as
# a false positive by compute_metrics, just as it is on the real movie.
UNLABELLED_CLASSES = ("unannotated",)


def load_config(path=None):
    """Read a simulator config, returning the `simulation` block."""
    path = DEFAULT_CONFIG if path is None else Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = raw["simulation"] if "simulation" in raw else raw
    return copy.deepcopy(cfg)


def active_classes(cfg):
    """Class names that are switched on and have a non-zero count."""
    out = []
    for name in CLASS_ORDER:
        c = cfg.get(name)
        if c and c.get("activate") and int(c.get("count", 0)) > 0:
            out.append(name)
    return out


# ===========================================================================
# RENDERING
# ===========================================================================

def _psf_patch(x, y, sigma, height, width, radius):
    """Sub-pixel, volume-normalised Gaussian patch for one emitter.

    Returns (slice_y, slice_x, patch) with patch scaled so that a unit PEAK
    amplitude gives a unit central value: patch = exp(-d^2 / 2 sigma^2) evaluated
    at true pixel centres offset from the emitter's continuous position. The
    integral is 2*pi*sigma^2 and is invariant to the sub-pixel offset, so this is
    volume-normalised in the sense R1 requires -- two emitters 3.6 px apart add
    their patches and the overlap is physically correct, whereas the old
    simulator's clip to [0, 1] before blurring fused them.

    Peak amplitude, rather than total volume, is the config unit because peak
    amplitude is what the photometry of section 4d measures and what the detector
    thresholds against.
    """
    cx, cy = int(np.round(x)), int(np.round(y))
    y0, y1 = cy - radius, cy + radius + 1
    x0, x1 = cx - radius, cx + radius + 1
    # Clip to the frame. The patch is evaluated from absolute pixel coordinates
    # rather than a fixed grid, so clipping needs no separate patch offset.
    y0c, y1c = max(0, y0), min(height, y1)
    x0c, x1c = max(0, x0), min(width, x1)
    if y0c >= y1c or x0c >= x1c:
        return None
    yy = np.arange(y0c, y1c)[:, None]
    xx = np.arange(x0c, x1c)[None, :]
    patch = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
    return slice(y0c, y1c), slice(x0c, x1c), patch


def render(positions, amplitudes, height, width, psf_sigma, tracks=None,
           per_emitter_sigma=None):
    """Sum every emitter's own PSF into every frame. Never blurs a summed frame.

    positions  (n, 2) static (x, y); ignored for emitters listed in `tracks`
    amplitudes (n_frames, n) peak amplitude in ADU, 0 where dark
    tracks     optional {emitter_index: (n_frames, 2) positions} for movers
    per_emitter_sigma  optional (n,) PSF width per emitter, overriding psf_sigma.
               Measured, the two classes differ: the real movie's targets give a
               stacked second-moment sigma of 1.478 and its off-targets 1.559, and
               that width feeds straight into how often the detector fires on them.

    Static emitters get their patch computed ONCE -- their geometry does not change
    between frames, only their amplitude -- which is what keeps a 501x256x512 movie
    inside R8's budget. Movers recompute per frame.
    """
    n_frames, n = amplitudes.shape
    sigmas = (np.full(n, float(psf_sigma)) if per_emitter_sigma is None
              else np.asarray(per_emitter_sigma, dtype=float))
    # One radius for all emitters, from the widest, so a per-class sigma cannot
    # silently truncate the broadest PSF.
    widest = float(np.max(sigmas)) if n else float(psf_sigma)
    radius = int(np.ceil(4.0 * widest))
    tracks = tracks or {}

    cache = {}
    for i in range(n):
        if i in tracks:
            continue
        cache[i] = _psf_patch(positions[i, 0], positions[i, 1], sigmas[i],
                              height, width, radius)

    signal = np.zeros((n_frames, height, width), dtype=np.float32)
    for t in range(n_frames):
        frame = signal[t]
        amps = amplitudes[t]
        for i in np.flatnonzero(amps):
            i = int(i)
            if i in tracks:
                p = tracks[i][t]
                if not np.isfinite(p).all():
                    continue
                entry = _psf_patch(p[0], p[1], sigmas[i], height, width, radius)
            else:
                entry = cache[i]
            if entry is None:
                continue
            sy, sx, patch = entry
            frame[sy, sx] += np.float32(amps[i]) * patch
    return signal


def apply_noise(signal, background_field, gain, read_noise, rng, dtype="uint16",
                saturation=65535):
    """Background, then Poisson, then read noise -- in that order (R4).

    The order is not cosmetic. Shot noise is a property of the photon arrivals, so
    it must be applied to the summed background-plus-signal photon rate; adding
    Gaussian read noise first and then drawing Poisson from the result would model
    a sensor that amplifies its own read noise.

    Photon conversion. Pixel values are ADU and the measured gain is about 5 ADU
    per photon, so the photon rate is value / gain, the Poisson draw happens in
    photon units, and the result is scaled back. Skipping that (drawing Poisson
    directly on ADU) would understate the noise by sqrt(gain) -- a factor of 2.2
    here -- and hand the detector a far cleaner movie than the real one.
    """
    n_frames, height, width = signal.shape
    out = np.empty((n_frames, height, width),
                   dtype=np.uint16 if dtype == "uint16" else np.float32)
    for t in range(n_frames):
        adu = background_field.frame(t) + signal[t]
        photons = np.clip(adu / gain, 0.0, None)
        noisy = rng.poisson(photons).astype(np.float64) * gain
        if read_noise > 0:
            noisy += rng.normal(0.0, read_noise, size=noisy.shape)
        np.clip(noisy, 0, saturation, out=noisy)
        out[t] = np.rint(noisy).astype(out.dtype) if dtype == "uint16" else noisy
    return out


# ===========================================================================
# ONE MOVIE
# ===========================================================================

def simulate(cfg, seed, verbose=False):
    """Generate one movie and its ground truth.

    Returns (movie, spots, meta):
      movie  (T, H, W) uint16, ready for SCP.load_tif_video
      spots  {class_name: (n, 2) array of (x, y)} -- sub-pixel, in placement order
      meta   provenance: resolved config, seed, realised photophysics statistics

    Seeding (R6). One seed reproduces the movie AND its ground truth exactly, and
    distinct seeds give independent movies. Every random draw goes through a
    generator derived from `seed` via SeedSequence.spawn, and generators are passed
    explicitly -- no global np.random state is touched anywhere, so a caller's own
    RNG use cannot perturb a movie.
    """
    g = cfg["global_settings"]
    n_frames = int(g["n_frames"])
    height, width = int(g["height"]), int(g["width"])
    psf_sigma = float(g["psf_sigma"])
    gain = float(g["gain_adu_per_photon"])
    read_noise = float(g["read_noise_adu"])
    dtype = g.get("dtype", "uint16")
    saturation = int(g.get("saturation", 65535))

    # Independent streams, so adding a class or changing the noise model does not
    # reshuffle the others.
    ss = np.random.SeedSequence(int(seed))
    rng_place, rng_photo, rng_bg, rng_noise = (np.random.default_rng(s)
                                               for s in ss.spawn(4))

    classes = active_classes(cfg)
    counts = {c: int(cfg[c]["count"]) for c in classes}

    place_cfg = cfg.get("placement", {})
    positions, class_of = pl.place_spots(
        counts, height, width,
        min_distance=float(place_cfg.get("min_distance", 3.5)),
        margin=float(place_cfg.get("margin", 2.0)),
        soft_distance=float(place_cfg.get("soft_distance", 6.0)),
        soft_gamma=float(place_cfg.get("soft_gamma", 3.0)),
        rng=rng_place)

    # Amplitudes per class, written back into one (n_frames, n) array in placement
    # order so the renderer stays class-agnostic.
    amplitudes = np.zeros((n_frames, len(positions)), dtype=np.float64)
    # Per-class PSF width, defaulting to the global one. Measured, the real movie's
    # off-target spots are wider than its targets (1.559 vs 1.478), and width drives
    # how often the detector fires on a spot.
    emitter_sigma = np.full(len(positions), psf_sigma, dtype=float)
    photo_info, tracks = {}, {}
    for cls in classes:
        idx = np.flatnonzero(class_of == cls)
        p = cfg[cls]["params"]
        emitter_sigma[idx] = float(p.get("psf_sigma", psf_sigma))
        trace, info = pp.emitter_amplitudes(
            len(idx), n_frames, rng_photo,
            peak_median=float(p["peak_amplitude_median"]),
            peak_sigma_log=float(p["peak_amplitude_sigma_log"]),
            tau_on=p.get("tau_on"), tau_off=p.get("tau_off"),
            tau_bleach=p.get("tau_bleach"),
            activation_tau=float(p.get("activation_tau", 0.0)),
            already_active_fraction=float(p.get("already_active_fraction", 1.0)),
            on_amplitude_cv=float(p.get("on_amplitude_cv", 0.0)),
            bleach_shape=float(p.get("bleach_shape", 1.0)))
        amplitudes[:, idx] = trace
        info["off_time_fraction_exceeding_tracker_gap"] = pp.geometric_tail_fraction(
            p.get("tau_off"))
        photo_info[cls] = info

        # The moving class stays implemented but is `activate: false` by default:
        # the real movie has no moving annotations at all (section 4a).
        if cls == "moving":
            step = float(p.get("step_size", 1.0))
            for i in idx:
                walk = np.cumsum(rng_photo.normal(0.0, step, size=(n_frames, 2)),
                                 axis=0) + positions[i]
                walk[:, 0] = np.clip(walk[:, 0], 0, width - 1)
                walk[:, 1] = np.clip(walk[:, 1], 0, height - 1)
                tracks[int(i)] = walk

    signal = render(positions, amplitudes, height, width, psf_sigma, tracks=tracks,
                    per_emitter_sigma=emitter_sigma)

    field = bg.BackgroundField(height, width, n_frames, rng_bg,
                               **{k: v for k, v in (cfg.get("background") or {}).items()})
    movie = apply_noise(signal, field, gain, read_noise, rng_noise,
                        dtype=dtype, saturation=saturation)

    spots = {cls: positions[class_of == cls] for cls in classes}
    meta = {
        "seed": int(seed),
        "generator": "Simulation/simulate.py",
        "shape": [n_frames, height, width],
        "dtype": str(movie.dtype),
        "psf_sigma": psf_sigma,
        "psf_sigma_per_class": {c: float(cfg[c]["params"].get("psf_sigma", psf_sigma))
                                for c in classes},
        "gain_adu_per_photon": gain,
        "read_noise_adu": read_noise,
        "counts": counts,
        "photophysics_realised": photo_info,
        "background_realised": field.describe(),
        "placement": {k: float(place_cfg.get(k, d)) for k, d in
                      (("min_distance", 3.5), ("margin", 2.0),
                       ("soft_distance", 6.0), ("soft_gamma", 3.0))},
        "resolved_config": copy.deepcopy(cfg),
    }
    if verbose:
        print(f"  seed {seed}: {n_frames}x{height}x{width} {movie.dtype}, "
              f"counts {counts}, range {movie.min()}-{movie.max()}")
        for cls, info in photo_info.items():
            print(f"    {cls:10s} peak median {info['peak_amplitude_median']:7.1f} ADU, "
                  f"blink duty {info['blink_duty_realised']:.3f}, "
                  f"overall duty {info['overall_duty_realised']:.3f}, "
                  f"live window median {info['live_window_median_frames']:.0f} fr")
    return movie, spots, meta


def ground_truth_arrays(spots):
    """spots -> (coord (N,2) in [y, x], classes (N,)) for the section 6c .npy output.

    The axis flip lives here and only here on the simulator side (trap 1): the
    label JSON of section 6a is (x, y) while the BachelorThesis .npy of section 6c
    is [y, x]. tests/test_contracts.py pins both against an asymmetric frame.
    """
    coords, classes = [], []
    for cls in CLASS_ORDER:
        if cls not in CLASS_IDS:      # unlabelled clutter: absent from ground truth
            continue
        pts = spots.get(cls)
        if pts is None or len(pts) == 0:
            continue
        for x, y in pts:
            coords.append([float(y), float(x)])       # [y, x]
            classes.append(CLASS_IDS[cls])
    if not coords:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.int64)
    return np.asarray(coords, dtype=np.float32), np.asarray(classes, dtype=np.int64)
