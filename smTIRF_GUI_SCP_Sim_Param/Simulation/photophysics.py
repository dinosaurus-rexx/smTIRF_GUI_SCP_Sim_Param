"""
Emitter brightness over time: per-molecule brightness, blinking, bleaching (R2).

Requirement R2 is that intensity comes from photophysics and never from geometry.
Every emitter here has a constant PSF width and a time-varying AMPLITUDE. Nothing
in this module or in simulate.py modulates spot size -- the old simulator's
`Dot.get_size` sinusoid is exactly what R2 forbids, and when its radius rounded to
zero the spot vanished outright.

Three mechanisms, each calibrated against a measured quantity:

  1. per-molecule brightness  a lognormal peak amplitude in ADU. Sets the WIDTH of
     the photometry distribution, not just its centre: the real movie's targets have
     a bg-subtracted IQR of [42.4, 95.3] around a median of 60.6, a q3/q1 ratio of
     2.25, which a single fixed amplitude cannot produce. (The old simulator clipped
     every frame to [0, 1] before blurring and so gave every spot an identical
     amplitude -- spec section 3a, row 2.)

  2. blinking  a two-state Markov chain with geometric dwell times, mean on-time
     tau_on and mean off-time tau_off frames. Stochastic on/off in intensity, not a
     deterministic sinusoid. tau_off matters more than it looks: SCP's tracker
     bridges gaps of at most 3 frames, so off-times that sometimes exceed 3 frames
     are what fragment a molecule's track and produce section 4e's duration
     distribution. A tau_off of 2-3 frames puts a useful share of the off-time
     distribution past that cliff (a geometric with mean 3 exceeds 3 frames about
     30% of the time).

  3. photobleaching  an exponential time to a permanent dark state, optionally
     after a delayed activation. This, not the blinking, is what stops a molecule
     spanning the whole movie: section 4e's detector-mediated median duration is
     135 of 501 frames for targets and 49 for off-targets.

The off-target class is defined EMPIRICALLY, per R3 and section 4d's warning. It is
not "N summed fluorophores": that model predicts aggregates BRIGHTER than targets,
and the real movie's are dimmer (18.9 vs 60.6 bg-subtracted). What reproduces the
measurement is a lower duty cycle -- a shorter-lived, more intermittent emitter --
which simultaneously explains the fainter mean, the HIGHER relative variability
(CV 1.596 vs 0.791) and the shorter detector tracks (49 vs 135 frames). No claim is
made that this is the physical mechanism; the mechanism is undocumented (trap 6).
"""

from __future__ import annotations

import numpy as np


def draw_peak_amplitudes(n, median, sigma_log, rng, minimum=0.0):
    """Per-molecule peak amplitude in ADU, lognormal about `median`.

    Parameterised by the median rather than the mean so the config knob is the
    quantity the photometry table reports. sigma_log is the standard deviation of
    the underlying normal; the interquartile ratio it produces is
    exp(2 * 0.6745 * sigma_log), so sigma_log = 0.60 gives about 2.25 -- the real
    movie's target ratio.
    """
    if n == 0:
        return np.empty(0)
    a = float(median) * np.exp(rng.normal(0.0, float(sigma_log), size=n))
    return np.maximum(a, float(minimum))


def blink_traces(n, n_frames, tau_on, tau_off, rng, initially_on_prob=None):
    """Binary on/off state per emitter per frame, from a two-state Markov chain.

    Dwell times are geometric with means tau_on and tau_off, i.e. per-frame switch
    probabilities 1/tau_on and 1/tau_off. Emitters start in the stationary
    distribution unless initially_on_prob overrides it, so there is no transient at
    frame 0.

    Returns a boolean (n_frames, n) array. tau_on <= 0 means "always on".
    """
    if n == 0:
        return np.zeros((n_frames, 0), dtype=bool)
    if tau_on is None or tau_on <= 0:
        return np.ones((n_frames, n), dtype=bool)

    p_off = min(1.0, 1.0 / float(tau_on))       # on  -> off
    p_on = min(1.0, 1.0 / float(tau_off))       # off -> on
    duty = p_on / (p_on + p_off)
    start_p = duty if initially_on_prob is None else float(initially_on_prob)

    state = np.empty((n_frames, n), dtype=bool)
    state[0] = rng.random(n) < start_p
    # Pre-draw the uniforms: one vectorised comparison per frame instead of per
    # emitter, which is what keeps a 501x256x512 movie inside R8's time budget.
    u = rng.random((n_frames, n))
    for t in range(1, n_frames):
        prev = state[t - 1]
        state[t] = np.where(prev, u[t] >= p_off, u[t] < p_on)
    return state


def bleach_windows(n, n_frames, tau_bleach, rng, activation_tau=0.0,
                   already_active_fraction=1.0, bleach_shape=1.0):
    """Per-emitter [start, end) frame window of being bleach-capable.

    An emitter is dark before `start` (not yet activated) and permanently dark from
    `end` (bleached). Time to bleach is gamma-distributed with mean tau_bleach
    frames and shape `bleach_shape`.

    bleach_shape = 1 is the exponential single-hit model. Higher values concentrate
    the lifetimes, which is the Erlang model of a fluorophore needing several
    independent hits before it dies -- and it is a knob the calibration needs, not
    just a generalisation for its own sake. Under the exponential model a
    substantial share of emitters bleach almost immediately, which puts a heavy low
    tail on the per-spot brightness distribution: measured, that gave a target IQR
    ratio of 2.78 against the real 2.25 and an off-target ratio of 4.17 against
    1.97, dragging gate T3's brightness AUC down to 0.85 even with every median
    matched. Concentrating the lifetimes narrows the spread without touching the
    medians.

    already_active_fraction is the share of emitters lit at frame 0. The real movie
    is a continuous recording of a pre-existing sample, so the default is all of
    them; a value below 1 staggers the rest with mean activation_tau.

    tau_bleach <= 0 means "never bleaches".
    """
    if n == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    start = np.zeros(n, dtype=int)
    if already_active_fraction < 1.0 and activation_tau > 0:
        late = rng.random(n) >= already_active_fraction
        if late.any():
            start[late] = np.minimum(
                rng.exponential(float(activation_tau), size=int(late.sum())).astype(int),
                n_frames)

    if tau_bleach is None or tau_bleach <= 0:
        end = np.full(n, n_frames, dtype=int)
    else:
        k = max(float(bleach_shape), 1e-3)
        # scale = mean / shape keeps the mean at tau_bleach for any shape.
        life = rng.gamma(k, float(tau_bleach) / k, size=n)
        end = np.minimum(start + np.maximum(life.astype(int), 1), n_frames)
    return start, end


def bleach_live_fraction_median(tau_bleach, n_frames, bleach_shape=1.0):
    """Median of min(bleach_life, T) / T -- the live fraction of the median emitter.

    Calibration targets the MEDIAN emitter (see calibrate.median_duty), so it needs
    the median of the truncated lifetime, not its mean. For the exponential case
    this is min(tau * ln 2, T) / T; for a general gamma shape it comes from the
    distribution's own quantile function.
    """
    if tau_bleach is None or tau_bleach <= 0:
        return 1.0
    k = max(float(bleach_shape), 1e-3)
    if abs(k - 1.0) < 1e-9:
        median_life = float(tau_bleach) * np.log(2.0)
    else:
        from scipy.stats import gamma as _gamma
        median_life = float(_gamma.ppf(0.5, k, scale=float(tau_bleach) / k))
    return float(min(median_life / n_frames, 1.0))


def bleach_tau_for_live_fraction(need_live, n_frames, bleach_shape=1.0):
    """Invert bleach_live_fraction_median(): mean lifetime for a wanted live fraction."""
    k = max(float(bleach_shape), 1e-3)
    want_median = float(need_live) * n_frames
    if abs(k - 1.0) < 1e-9:
        return want_median / np.log(2.0)
    from scipy.stats import gamma as _gamma
    # median = ppf(0.5, k, scale=tau/k), and ppf is linear in scale.
    unit_median = float(_gamma.ppf(0.5, k, scale=1.0 / k))
    return want_median / unit_median


def emitter_amplitudes(n, n_frames, rng, peak_median, peak_sigma_log,
                       tau_on, tau_off, tau_bleach, activation_tau=0.0,
                       already_active_fraction=1.0, on_amplitude_cv=0.0,
                       bleach_shape=1.0):
    """Full (n_frames, n) amplitude trace in ADU for one emitter class.

    Composes the three mechanisms: a per-molecule amplitude, gated by the blink
    state, gated again by the activation/bleach window. on_amplitude_cv adds
    frame-to-frame jitter to the on-state amplitude (a lognormal multiplier),
    covering the residual variability that on/off switching alone does not explain.

    Returns (amplitudes, info) where info records the realised duty cycle and
    window lengths -- the harness prints them, and they are what a calibration
    round actually turns.
    """
    amp = draw_peak_amplitudes(n, peak_median, peak_sigma_log, rng)
    on = blink_traces(n, n_frames, tau_on, tau_off, rng)
    start, end = bleach_windows(n, n_frames, tau_bleach, rng,
                                activation_tau=activation_tau,
                                already_active_fraction=already_active_fraction,
                                bleach_shape=bleach_shape)

    if n:
        t_idx = np.arange(n_frames)[:, None]
        live = (t_idx >= start[None, :]) & (t_idx < end[None, :])
        trace = np.where(on & live, amp[None, :], 0.0)
        if on_amplitude_cv > 0:
            jitter = np.exp(rng.normal(0.0, float(on_amplitude_cv),
                                       size=(n_frames, n)))
            trace = trace * jitter
    else:
        trace = np.zeros((n_frames, 0))
        live = np.zeros((n_frames, 0), dtype=bool)

    with np.errstate(invalid="ignore"):
        per_emitter_duty = (trace > 0).mean(axis=0) if n else np.empty(0)
        info = {
            "n": int(n),
            "peak_amplitude_median": float(np.median(amp)) if n else float("nan"),
            "peak_amplitude_iqr": ([float(np.percentile(amp, 25)),
                                    float(np.percentile(amp, 75))] if n else [np.nan] * 2),
            "blink_duty_realised": float(on.mean()) if n else float("nan"),
            "live_window_median_frames": float(np.median(end - start)) if n else float("nan"),
            "overall_duty_realised": float(per_emitter_duty.mean()) if n else float("nan"),
            # The MEDIAN per-emitter duty, which is the one that matters for
            # calibration: section 4d aggregates each spot's own trace statistic and
            # then takes the median ACROSS spots, so the median emitter -- not the
            # average one -- sets the reported CV. Bleaching gives the two very
            # different values (a handful of early-bleached emitters drag the mean
            # down while leaving the median at the blink duty), and confusing them
            # is what made a first calibration pass miss the target CV by 27%.
            "overall_duty_median": float(np.median(per_emitter_duty)) if n else float("nan"),
        }
    return trace, info


def geometric_tail_fraction(tau, gap=3):
    """Share of off-times longer than `gap` frames, for a geometric of mean tau.

    Documents the tracker interaction R2 calls for: SCP drops a track after a gap
    of more than 3 frames, so this is the fraction of blink-offs that fragment a
    track rather than being bridged. Reported in the manifest so the choice of
    tau_off is legible rather than a bare number.
    """
    if tau is None or tau <= 0:
        return 0.0
    p = min(1.0, 1.0 / float(tau))
    return float((1.0 - p) ** gap)
