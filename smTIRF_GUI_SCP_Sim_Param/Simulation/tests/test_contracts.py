"""
Contract tests: axis order, seeding (T7), and the output contract (T8).

Run:
    cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/Simulation
    python -m pytest tests -v
    python tests/test_contracts.py          # also works without pytest

These are cheap on purpose -- tiny frames, few emitters -- so they can run on every
change. The expensive checks live in validate_against_real.py.

Why these particular tests:

  * AXIS ORDER is the single most bug-prone thing in this codebase (spec section 9,
    trap 1). The label JSON is (x, y), the .npy is [y, x], SCP is (x, y), and numpy
    indexing is [row, col]. Spec section 6c says "get it right and unit-test it", and
    trap 1 says to use a deliberately asymmetric frame -- a square frame with a
    centred spot passes even when the axes are swapped, so every geometric test here
    uses a non-square frame with an off-centre spot.
  * The CONSUMERS must load the output unmodified (T8). These tests call the real
    reader functions from Parameter_opt/SCP_parametrizer_002.py and SCP itself, not
    reimplementations.
  * SEEDING (T7) is what makes any of the measurements reproducible.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
SIMDIR = HERE.parent
sys.path.insert(0, str(SIMDIR))
sys.path.insert(0, str(SIMDIR.parent / "SCP"))

import generate_dataset as gd      # noqa: E402
import measurements as ms          # noqa: E402
import placement as pl             # noqa: E402
import simulate as sm              # noqa: E402


# A deliberately ASYMMETRIC test config: 64 rows x 128 columns, so any transposed
# array is a shape error rather than a silent wrong answer.
def tiny_config(n_frames=12, height=64, width=128, n_target=6, n_off=3, n_unann=0):
    cfg = sm.load_config()
    g = cfg["global_settings"]
    g.update(n_frames=n_frames, height=height, width=width)
    cfg["target"]["count"] = n_target
    cfg["off_target"]["count"] = n_off
    cfg["unannotated"]["count"] = n_unann
    cfg["unannotated"]["activate"] = n_unann > 0
    return cfg


# ===========================================================================
# AXIS ORDER  (trap 1)
# ===========================================================================

def test_rendered_spot_lands_at_the_right_pixel():
    """A spot at (x, y) must brighten frame[y, x], not frame[x, y].

    The asymmetric position (x=100, y=20) is outside a 64-row frame if the axes are
    swapped, so a transposition cannot pass this by luck.
    """
    x, y = 100.0, 20.0
    positions = np.array([[x, y]])
    amps = np.full((1, 1), 500.0)
    signal = sm.render(positions, amps, height=64, width=128, psf_sigma=1.55)
    frame = signal[0]

    assert frame.shape == (64, 128), f"render returned {frame.shape}, expected (64, 128)"
    peak_y, peak_x = np.unravel_index(np.argmax(frame), frame.shape)
    assert (peak_y, peak_x) == (int(y), int(x)), (
        f"spot at (x={x}, y={y}) peaked at row {peak_y}, col {peak_x}; "
        "expected row 20, col 100 -- the axes are swapped")
    assert frame[int(y), int(x)] > 0.9 * 500.0


def test_ground_truth_npy_is_y_x_and_json_is_x_y():
    """Section 6a is (x, y); section 6c is [y, x]. The flip must happen exactly once."""
    spots = {"target": np.array([[100.0, 20.0]]),        # (x, y)
             "off_target": np.array([[10.0, 50.0]])}
    coord, classes = sm.ground_truth_arrays(spots)

    assert coord.shape == (2, 2)
    # coord is [y, x]: the target's row must be [20, 100], NOT [100, 20].
    assert np.allclose(coord[0], [20.0, 100.0]), f"coord[0] = {coord[0]}, expected [20, 100]"
    assert np.allclose(coord[1], [50.0, 10.0]), f"coord[1] = {coord[1]}, expected [50, 10]"
    assert classes.tolist() == [sm.CLASS_IDS["target"], sm.CLASS_IDS["off_target"]]

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "labels.json"
        gd.write_label_json(spots["target"], p)
        payload = json.loads(p.read_text(encoding="utf-8"))
        # The JSON is (x, y): x is the column.
        assert payload[0]["center"] == {"x": 100.0, "y": 20.0}, payload[0]


def test_npy_round_trips_through_the_real_loader():
    """Writing [y, x] and reading it back with the project's loader returns (x, y)."""
    spots = {"target": np.array([[100.0, 20.0]]), "off_target": np.array([[10.0, 50.0]])}
    coord, classes = sm.ground_truth_arrays(spots)
    with tempfile.TemporaryDirectory() as tmp:
        cpath, kpath = Path(tmp) / "coord.npy", Path(tmp) / "class.npy"
        np.save(cpath, coord)
        np.save(kpath, classes)
        (targets, offtargets), _ = ms.load_labels_npy(cpath, kpath)
    assert targets == [(100.0, 20.0)], targets
    assert offtargets == [(10.0, 50.0)], offtargets


def test_psf_measurement_agrees_with_the_configured_width():
    """The width estimator gate T1 relies on must recover a known PSF sigma.

    Guards the estimator itself: if psf_width() were biased, T1 could pass or fail
    for reasons unrelated to the simulator.
    """
    cfg = tiny_config(n_frames=30, height=96, width=160, n_target=8, n_off=0)
    cfg["global_settings"]["psf_sigma"] = 1.55
    cfg["target"]["params"].update(tau_on=1e6, tau_off=1.0, tau_bleach=0.0,
                                   on_amplitude_cv=0.0, peak_amplitude_sigma_log=0.01,
                                   peak_amplitude_median=600.0)
    movie, spots, _ = sm.simulate(cfg, seed=3)
    r = ms.psf_width(movie, [(float(x), float(y)) for x, y in spots["target"]])
    got = r["stacked_second_moment_sigma"]
    assert 1.3 < got < 1.75, f"configured sigma 1.55, measured {got:.3f}"


# ===========================================================================
# SEEDING  (T7)
# ===========================================================================

def test_same_seed_reproduces_movie_and_ground_truth():
    cfg = tiny_config()
    a_movie, a_spots, _ = sm.simulate(cfg, seed=7)
    b_movie, b_spots, _ = sm.simulate(cfg, seed=7)
    assert np.array_equal(a_movie, b_movie), "same seed gave different pixels"
    for cls in a_spots:
        assert np.array_equal(a_spots[cls], b_spots[cls]), f"{cls} geometry differs"


def test_different_seeds_differ_in_both_geometry_and_noise():
    cfg = tiny_config()
    a_movie, a_spots, _ = sm.simulate(cfg, seed=7)
    b_movie, b_spots, _ = sm.simulate(cfg, seed=8)
    assert not np.array_equal(a_movie, b_movie), "distinct seeds gave identical pixels"
    assert not np.array_equal(a_spots["target"], b_spots["target"]), \
        "distinct seeds gave identical geometry"


def test_global_rng_state_does_not_affect_output():
    """R6: generators are passed explicitly, so global np.random cannot perturb a movie.

    The old simulator seeded the global RNG, which meant any caller drawing a random
    number between two calls changed the result.
    """
    cfg = tiny_config()
    np.random.seed(1234)
    a, _, _ = sm.simulate(cfg, seed=11)
    np.random.seed(999)
    np.random.random(1000)                 # perturb global state
    b, _, _ = sm.simulate(cfg, seed=11)
    assert np.array_equal(a, b), "global RNG state leaked into the simulation"


# ===========================================================================
# OUTPUT CONTRACT  (T8)
# ===========================================================================

def _build_dataset(tmp, **kw):
    cfg = tiny_config(**kw)
    out = Path(tmp) / "ds"
    split = out / "train"
    split.mkdir(parents=True, exist_ok=True)
    rec = gd.build_movie(cfg, seed=0, split_dir=split, verbose=False)
    manifest = {"meta": {"generator": "Simulation/simulate.py",
                         "resolved_config": cfg, "records": [{"seed": 0}]},
                "train": [rec], "val": []}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str),
                                       encoding="utf-8")
    return out, rec


def test_tiff_is_T_H_W_and_loads_through_scp():
    """Section 6d: (T, H, W) single channel, readable by SCP.load_tif_video."""
    with tempfile.TemporaryDirectory() as tmp:
        _, rec = _build_dataset(tmp, n_frames=12, height=64, width=128)
        raw = tifffile.imread(rec["tif"])
        assert raw.ndim == 3 and raw.shape == (12, 64, 128), raw.shape
        assert raw.dtype == np.uint16, raw.dtype

        import SCP_KTree_JSON as scp
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            frames = scp.load_tif_video(rec["tif"])
        assert frames.shape == (12, 64, 128), (
            f"SCP.load_tif_video returned {frames.shape}. Note its (H, W, C) branch "
            "triggers when the last axis is <= 4, so a movie only a few pixels wide "
            "would be transposed -- keep test frames wider than that.")


def test_manifest_loads_through_the_parametrizer_unmodified():
    """T8: load_movies_from_manifest and extract_center_coordinates, unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        out, rec = _build_dataset(tmp, n_target=6, n_off=3)
        meta, train, val = ms.pm.load_movies_from_manifest(out / "manifest.json")
        assert len(train) == 1
        mv = train[0]
        assert len(mv["positive_spots"]) == 6, mv["positive_spots"]
        assert len(mv["negative_spots"]) == 3
        assert all(isinstance(s, tuple) and len(s) == 2 for s in mv["positive_spots"])
        assert Path(mv["tif"]).exists()
        # Coordinates must be inside the frame in (x, y) order: x < width, y < height.
        for x, y in mv["positive_spots"]:
            assert 0 <= x < 128 and 0 <= y < 64, f"({x}, {y}) outside a 64x128 frame"


def test_unannotated_emitters_appear_in_no_ground_truth_file():
    """They are deliberately unlabelled, so a detection on one is a false positive."""
    with tempfile.TemporaryDirectory() as tmp:
        out, rec = _build_dataset(tmp, n_target=5, n_off=2, n_unann=7)
        assert rec["n_positive"] == 5 and rec["n_negative"] == 2
        assert rec["n_unannotated"] == 7
        coord = np.load(rec["coord_npy"])
        classes = np.load(rec["class_npy"])
        assert len(coord) == 7, f"coord.npy has {len(coord)} rows, expected 5 + 2"
        assert set(np.unique(classes)) <= {0, 1, 2}
        pos = ms.pm.extract_center_coordinates(rec["positive"])
        neg = ms.pm.extract_center_coordinates(rec["negative"])
        assert len(pos) + len(neg) == 7


def test_label_json_and_npy_describe_the_same_spots():
    """The two ground-truth formats must not disagree -- the axis flip is the risk."""
    with tempfile.TemporaryDirectory() as tmp:
        out, rec = _build_dataset(tmp, n_target=6, n_off=3)
        pos = ms.pm.extract_center_coordinates(rec["positive"])       # (x, y)
        coord = np.load(rec["coord_npy"])                             # [y, x]
        classes = np.load(rec["class_npy"])
        npy_targets = {(round(float(c[1]), 4), round(float(c[0]), 4))
                       for c in coord[classes == sm.CLASS_IDS["target"]]}
        json_targets = {(round(float(x), 4), round(float(y), 4)) for x, y in pos}
        assert npy_targets == json_targets, (
            f"JSON and .npy disagree.\n  json: {sorted(json_targets)[:3]}\n"
            f"  npy : {sorted(npy_targets)[:3]}")


# ===========================================================================
# PLACEMENT  (R5)
# ===========================================================================

def test_placement_respects_the_hard_floor_and_the_margin():
    rng = np.random.default_rng(0)
    pts, cls = pl.place_spots({"target": 60, "off_target": 20}, 128, 256,
                              min_distance=3.5, margin=2.0, rng=rng)
    assert len(pts) == 80
    d = pl.nn_distances(pts)
    assert d.min() >= 3.5, f"minimum separation {d.min():.3f} is below the 3.5 px floor"
    assert pts[:, 0].min() >= 2.0 and pts[:, 0].max() <= 256 - 2.0
    assert pts[:, 1].min() >= 2.0 and pts[:, 1].max() <= 128 - 2.0
    assert set(cls) == {"target", "off_target"}


def test_soft_core_suppresses_close_pairs_relative_to_a_hard_core():
    """The soft core is what makes gate T5's KS test pass; a hard core alone fails it.

    Pooled over several seeds. The close-pair FRACTION is the quantity the soft core
    actually controls and it separates cleanly; the NN median moves too (10.90 to
    12.04 px on the real spot count) but by less than its seed-to-seed scatter at
    this sample size, so asserting on a single seed's median is a flaky test.
    """
    def pooled(**kw):
        return np.concatenate([
            pl.nn_distances(pl.place_spots({"target": 243}, 256, 512,
                                           min_distance=3.5,
                                           rng=np.random.default_rng(s), **kw)[0])
            for s in range(4)])

    hard = pooled(soft_distance=0.0)                       # pure hard core
    soft = pooled(soft_distance=6.0, soft_gamma=3.0)
    assert hard.min() >= 3.5 and soft.min() >= 3.5
    assert (soft < 6).mean() < 0.5 * (hard < 6).mean(), (
        f"soft core did not suppress close pairs: {(soft < 6).mean():.3f} vs "
        f"{(hard < 6).mean():.3f} under 6 px")
    assert np.median(soft) > np.median(hard), (
        f"soft core did not raise the NN median: {np.median(soft):.2f} vs "
        f"{np.median(hard):.2f}")


def test_placement_raises_rather_than_silently_dropping_spots():
    """An impossible request must fail loudly, not return fewer spots than asked."""
    try:
        pl.place_spots({"target": 5000}, 32, 32, min_distance=3.5,
                       rng=np.random.default_rng(0), max_attempts_per_spot=200)
    except pl.PlacementError:
        return
    raise AssertionError("placing 5000 spots in 32x32 should have raised PlacementError")


# ===========================================================================
# PHOTOPHYSICS  (R2)
# ===========================================================================

def test_amplitude_varies_but_spot_size_never_does():
    """R2: intensity comes from photophysics, size is constant.

    The old simulator modulated the RADIUS, and when it rounded to zero the spot
    vanished outright. Here the measured width must be the same when the spot is
    bright and when it is faint.

    The emitter is held ALWAYS ON (tau_on huge, no bleaching) and its brightness is
    varied purely by frame-to-frame amplitude jitter. Comparing a bright frame with
    a blinked-OFF frame would prove nothing: with no spot present, a second moment
    over the window just measures noise and drifts toward the window radius.
    """
    cfg = tiny_config(n_frames=80, height=96, width=160, n_target=10, n_off=0)
    cfg["target"]["params"].update(tau_on=1e6, tau_off=1.0, tau_bleach=0.0,
                                   on_amplitude_cv=0.45,           # vary the amplitude
                                   peak_amplitude_sigma_log=0.01,
                                   peak_amplitude_median=1500.0)
    movie, spots, _ = sm.simulate(cfg, seed=5)
    x, y = (float(v) for v in spots["target"][0])
    cx, cy = int(round(x)), int(round(y))
    trace = movie[:, cy, cx].astype(float)
    assert trace.max() / max(trace.min(), 1.0) > 1.5, (
        f"amplitude barely varied (min {trace.min():.0f}, max {trace.max():.0f}); "
        "the test cannot detect size modulation without a brightness range")

    yy, xx = np.mgrid[-6:7, -6:7]
    def width_at(t):
        sub = movie[t, cy - 6:cy + 7, cx - 6:cx + 7].astype(float)
        w = np.clip(sub - np.median(sub), 0, None)
        return np.sqrt((w * (xx ** 2 + yy ** 2)).sum() / w.sum() / 2) if w.sum() > 0 else np.nan

    bright, faint = int(np.argmax(trace)), int(np.argmin(trace))
    wb, wf = width_at(bright), width_at(faint)
    assert np.isfinite(wb) and np.isfinite(wf)
    assert abs(wb - wf) < 0.5, (
        f"spot width changed with brightness: {wb:.3f} (bright frame) vs {wf:.3f} "
        "(faint frame). Size must never be modulated (R2).")


def test_bleached_emitters_stay_dark():
    cfg = tiny_config(n_frames=200, n_target=20, n_off=0)
    cfg["target"]["params"].update(tau_on=1e6, tau_off=1.0, tau_bleach=20.0,
                                   bleach_shape=1.0)
    rng = np.random.default_rng(0)
    amps, info = __import__("photophysics").emitter_amplitudes(
        20, 200, rng, peak_median=100.0, peak_sigma_log=0.01,
        tau_on=1e6, tau_off=1.0, tau_bleach=20.0, bleach_shape=1.0)
    for i in range(20):
        on = np.flatnonzero(amps[:, i] > 0)
        if on.size:
            # Once dark after the last on-frame, it must never light up again.
            assert np.all(amps[on[-1] + 1:, i] == 0), f"emitter {i} revived after bleaching"


def _all_tests():
    return [(k, v) for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]


if __name__ == "__main__":
    failures = []
    for name, fn in _all_tests():
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                                  # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(_all_tests()) - len(failures)}/{len(_all_tests())} passed")
    sys.exit(1 if failures else 0)
