"""
Structured background field (SIMULATOR_SPEC.md R4, section 4g).

Section 4g was the one calibration target the spec could not hand over. Measured
on realdata/vid_ch1.tiff by measure_real_movie.py, the sigma=8 Gaussian low-pass
of a frame -- the very field SCP's bandpass subtracts -- looks like this:

    mean                                    147.19 ADU
    spatial std, per frame                    9.58
      of which static (time-averaged field)   8.57
      of which drifting (per-pixel over time) 5.46
    min / max                               111.60 / 222.98
    relative swing (max - min) / mean         0.529
    spatial autocorrelation length (1/e)     29.78 px
    per-frame mean                          142.85 .. 157.28
    pattern correlation with frame 0          0.664 .. 0.836 (it DRIFTS)

and the high-frequency part, isolated by differencing consecutive frames so static
structure cancels:

    frame-to-frame noise                     27.41 ADU
    implied variance/mean                     5.10  -> ADU are not photons;
                                                       the gain is about 5 ADU/photon

A flat `background_level` cannot produce any of that, which is spec section 3b's
row 5 and, through the per-frame min-max normalisation in SCP.detect_spots, a
direct cause of row 6 -- thresholds that do not transfer.

Model. A Gaussian random field: white noise smoothed by a Gaussian, then rescaled
to a target standard deviation. Smoothed white noise has a Gaussian
autocorrelation of width sigma*sqrt(2), whose 1/e point sits at 2*sigma, so a
field generated at sigma_f and then measured through the sigma=8 analysis filter
has an autocorrelation length of about 2*sqrt(sigma_f^2 + 64). Solving that for the
measured 29.78 px gives sigma_f = 12.6 px, which is the calibrated default.

Two components share that construction:
  * a static field, redrawn per movie but constant across its frames;
  * a drifting field, interpolated between a handful of independent draws so the
    pattern decorrelates over the movie the way the real one does.

A few broad blobs plus a gradient would also do (the spec suggests as much), but a
smoothed random field is what makes the autocorrelation length a single explicit
knob instead of an emergent property of hand-placed blobs.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def gaussian_random_field(height, width, smooth_sigma, target_std, rng):
    """Zero-mean Gaussian random field with a given std and smoothing scale.

    Rescaled to target_std AFTER smoothing, because smoothing reduces the variance
    by a factor that depends on the kernel and the grid; setting the pre-smoothing
    amplitude would make target_std meaningless.
    """
    field = rng.standard_normal((height, width))
    field = ndimage.gaussian_filter(field, smooth_sigma, mode="reflect")
    std = field.std()
    if std <= 0:
        return np.zeros((height, width))
    return field * (float(target_std) / std)


def autocorr_length_for(measured_length_px, analysis_sigma=8.0):
    """Generation smoothing sigma that yields a measured 1/e length.

    Inverts length = 2 * sqrt(sigma_f^2 + analysis_sigma^2). Returns None when the
    target is too small to be reachable through the analysis filter at all -- the
    analysis filter alone already imposes a floor of 2 * analysis_sigma.
    """
    half = float(measured_length_px) / 2.0
    inner = half ** 2 - float(analysis_sigma) ** 2
    return float(np.sqrt(inner)) if inner > 0 else None


class BackgroundField:
    """Per-frame background in ADU: base level + static field + drifting field.

    Built once per movie and then queried per frame, so the static component is
    genuinely static and the drift is genuinely continuous.
    """

    def __init__(self, height, width, n_frames, rng,
                 base_level=147.19, static_std=8.57, drift_std=5.46,
                 smooth_sigma=12.6, drift_knots=4, drift_mean_std=4.10):
        self.height, self.width, self.n_frames = height, width, n_frames
        self.base_level = float(base_level)

        self.static = (gaussian_random_field(height, width, smooth_sigma,
                                             static_std, rng)
                       if static_std > 0 else np.zeros((height, width)))

        # Drift: independent fields at a few knots, linearly interpolated between
        # them. drift_knots controls how fast the pattern decorrelates; 4 knots
        # over the movie reproduces the measured 0.66-0.84 correlation with frame 0.
        self.drift_knots = max(int(drift_knots), 1) if drift_std > 0 else 0
        if self.drift_knots:
            self._knots = np.stack([
                gaussian_random_field(height, width, smooth_sigma, drift_std, rng)
                for _ in range(self.drift_knots + 1)])
        else:
            self._knots = None

        # A global per-frame offset, reproducing the measured wander of the
        # frame mean (142.85-157.28, std 4.10) independently of spatial pattern.
        if drift_mean_std > 0 and n_frames > 1:
            walk = np.cumsum(rng.standard_normal(n_frames))
            walk -= walk.mean()
            s = walk.std()
            self.mean_offset = walk * (float(drift_mean_std) / s) if s > 0 else walk * 0
        else:
            self.mean_offset = np.zeros(n_frames)

    def frame(self, t):
        """Background of frame t, in ADU."""
        out = self.base_level + self.static + self.mean_offset[t]
        if self._knots is not None:
            pos = (t / max(self.n_frames - 1, 1)) * self.drift_knots
            i = min(int(pos), self.drift_knots - 1)
            f = pos - i
            out = out + (1.0 - f) * self._knots[i] + f * self._knots[i + 1]
        return out

    def describe(self):
        return {
            "base_level": self.base_level,
            "static_std": float(self.static.std()),
            "drift_knots": self.drift_knots,
            "mean_offset_std": float(self.mean_offset.std()),
        }
