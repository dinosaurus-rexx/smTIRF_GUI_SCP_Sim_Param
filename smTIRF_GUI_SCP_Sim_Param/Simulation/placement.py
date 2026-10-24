"""
Spot placement by hard-core rejection sampling (SIMULATOR_SPEC.md R5, section 4c).

The floor is 3.5 px, not BachelorThesis's 7. Measured on the real movie, the
closest pair of annotated targets is 3.61 px apart and 12 of 243 targets have a
nearest neighbour under 6 px, so a 7 px floor forbids spacings that genuinely
occur -- and it shows: the BT dataset's target nearest-neighbour minimum is
exactly 7.04 px with zero pairs under 6, which gate T5 rejects.

Matching the floor is necessary but nowhere near sufficient; T5 also runs a
two-sample KS test against the real distribution, and a hard core alone fails it.
Measured: a pure Poisson process with a 3.5 px hard core gives a target NN median
of 10.90 px with 12.3% of targets under 6 px, while the real annotations give
12.04 px with 4.9% under 6 px. The real process is more REGULAR than Poisson --
close pairs are suppressed but not forbidden -- so the model needs a soft core, not
a hard one:

    d < d_hard              always reject
    d_hard <= d < d_soft    accept with probability
                            ((d - d_hard) / (d_soft - d_hard)) ** gamma
    d >= d_soft             always accept

Fitted against the real target and all-spot distributions, d_soft = 6.0 and
gamma = 3.0 give a median of 12.04 px, 3.8% under 6 px, and KS p-values of 0.91
(targets) and 0.91 (all) -- versus 0.0000 for the hard-core-only process. That the
suppression sets in right at the 6 px mark is unsurprising for hand annotation:
two molecules a PSF width apart are hard to mark as two.

One caveat, recorded rather than papered over: with gamma = 3 the acceptance
probability at 3.6 px is about 6e-5, so simulated minima land near 4.5-4.7 px while
the real movie has 2 of 243 targets under 4 px. T5's floor requirement (>= 3.5 px)
is met and its KS test passes; the extreme 0.8% tail is not reproduced.

Both classes are placed by ONE process with one shared floor, because the real
movie's constraint is shared: its closest pair over all annotations is the same
3.61 px as its closest target pair.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class PlacementError(RuntimeError):
    """Raised when the requested spot count cannot be placed at the given floor."""


def place_spots(counts, height, width, min_distance=3.5, margin=2.0, rng=None,
                soft_distance=6.0, soft_gamma=3.0, max_attempts_per_spot=4000):
    """Draw spot positions with a soft-core minimum separation.

    counts is a mapping {class_name: n}. Positions are returned as one (N, 2)
    array of (x, y) floats plus an (N,) array of class names, interleaved in a
    shuffled order so no class is systematically placed into the gaps left by
    another -- placing all targets first and then all off-targets would make the
    off-target spacing distribution depend on the target one.

    min_distance is the hard floor; between it and soft_distance a candidate is
    accepted with probability ((d - min) / (soft - min)) ** soft_gamma. Setting
    soft_distance <= min_distance recovers a pure hard core, which is measurably
    wrong for this data -- see the module docstring.

    margin keeps spots that far from the frame edge. It is recorded in the
    manifest because it matters downstream: the background annulus of a spot too
    close to an edge is clipped, which biases its photometry (see
    measurements._spot_window and SCP._aperture_intensity). The real movie's own
    annotations reach x=1 and y=1 of a 256x512 frame, so the default is
    deliberately small.

    Sub-pixel by construction: positions are continuous, never snapped to a pixel
    centre, so the renderer exercises its sub-pixel PSF path (R1).
    """
    rng = np.random.default_rng() if rng is None else rng

    queue = []
    for name, n in counts.items():
        queue.extend([name] * int(n))
    if not queue:
        return np.empty((0, 2)), np.empty(0, dtype=object)
    rng.shuffle(queue)

    lo_x, hi_x = float(margin), float(width - margin)
    lo_y, hi_y = float(margin), float(height - margin)
    if hi_x <= lo_x or hi_y <= lo_y:
        raise PlacementError(f"margin {margin} leaves no room in {height}x{width}")

    accepted = np.empty((len(queue), 2), dtype=float)
    classes = np.empty(len(queue), dtype=object)
    n_placed = 0
    hard = float(min_distance)
    soft = float(soft_distance)
    span = soft - hard
    gamma = float(soft_gamma)

    for cls in queue:
        for _attempt in range(max_attempts_per_spot):
            x = rng.uniform(lo_x, hi_x)
            y = rng.uniform(lo_y, hi_y)
            if n_placed:
                d = np.sqrt(((accepted[:n_placed, 0] - x) ** 2
                             + (accepted[:n_placed, 1] - y) ** 2).min())
                if d < hard:
                    continue
                if span > 0 and d < soft:
                    if rng.random() > ((d - hard) / span) ** gamma:
                        continue
            accepted[n_placed] = (x, y)
            classes[n_placed] = cls
            n_placed += 1
            break
        else:
            raise PlacementError(
                f"could not place spot {n_placed + 1} of {len(queue)} in "
                f"{height}x{width} after {max_attempts_per_spot} attempts "
                f"(min_distance={min_distance}, soft_distance={soft_distance}, "
                f"soft_gamma={soft_gamma}) -- lower the count, the floor or the "
                "soft-core strength")

    return accepted, classes


def nn_median_expected_poisson(n, height, width):
    """Median nearest-neighbour distance of a Poisson process, for orientation.

    For density lambda, P(NN > r) = exp(-pi lambda r^2), so the median is
    sqrt(ln 2 / (pi lambda)). Used only in diagnostics and docstrings: it is why
    the real movie's 12.04 px median is read as "uniform with a small hard core"
    rather than as evidence of spatial structure.
    """
    lam = n / (height * width)
    return float(np.sqrt(np.log(2.0) / (np.pi * lam))) if lam > 0 else float("nan")


def nn_distances(xy):
    """Nearest-neighbour distance per point (diagnostics; mirrors measurements)."""
    pts = np.asarray(xy, dtype=float)
    if len(pts) < 2:
        return np.empty(0)
    d, _ = cKDTree(pts).query(pts, k=2)
    return d[:, 1]
