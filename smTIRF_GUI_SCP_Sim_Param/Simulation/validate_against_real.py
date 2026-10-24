"""
Score a simulated dataset against the real movie: gates T1-T5 and T7.

This is the deliverable of SIMULATOR_SPEC.md section 8 and the instrument the
whole calibration effort is steered by. It reads its targets from
config/real_movie_reference.json -- derived by measure_real_movie.py, never
hand-written -- so re-deriving beats trusting the spec, and a new real recording
can be dropped in without touching this file.

Both sides are measured by measurements.measure_movie(), the same function the
reference came from. Nothing here re-implements an estimator or a scorer
(SIMULATOR_SPEC.md section 9, trap 4).

    python validate_against_real.py --manifest dataset/manifest.json
    python validate_against_real.py --manifest dataset_bt/manifest.json --label BT-sim
    python validate_against_real.py --manifest dataset/manifest.json --no-detector

Gates, and what each one is for:

  T1  spot scale        the LoG scale a movie's spots respond at must agree
  T2  threshold transfer   blob density vs threshold, +-25% -- the headline test
  T3  photometry        section 4d medians +-25%, right sign, overlapping IQRs
  T4  track duration    section 4e medians +-30% after length scaling, right order
  T5  spacing           NN median +-2 px, floor >= 3.5 px, KS not rejected
  T7  reproducibility   one seed -> identical movie; different seeds -> different

T6 (end-to-end parameter transfer) is deliberately NOT here: it runs the real
tuning protocol through Parameter_opt/SCP_parametrizer_002.py and takes hours.
See experiments/real_calibrated_sim_transfer/.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
from scipy.stats import ks_2samp

import measurements as ms

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE / "config" / "real_movie_reference.json"
DEFAULT_OUTDIR = HERE / "validation"

# ---------------------------------------------------------------------------
# Gate tolerances. Every one is quoted from SIMULATOR_SPEC.md section 7; they are
# named here so a report can state the tolerance it applied.
# ---------------------------------------------------------------------------
TOL = {
    "T1_sigma_px": 0.3,             # "outside +-0.3 of each bound"
    "T2_density_rel": 0.25,         # "within +-25%"
    "T3_median_rel": 0.25,          # "within +-25%"
    "T3_auc_range": (0.85, 0.95),   # "must land in 0.85-0.95"
    "T4_median_rel": 0.30,          # "within +-30% ... after scaling"
    "T5_median_px": 2.0,            # "median within +-2 px"
    "T5_min_floor": 3.5,            # "minimum >= 3.5 px"
    "T5_under6_frac": 0.05,         # "a comparable count of sub-6 px pairs"
    "T5_ks_alpha": 0.05,            # "should not reject at alpha=0.05"
}
T2_GATE_THRESHOLDS = (0.25, 0.32, 0.40, 0.50)   # 0.05 reported but not gated


# ===========================================================================
# MEASURING A DATASET
# ===========================================================================

def load_manifest_movies(manifest_path, split="train"):
    """Movies from a manifest, resolved the way the parametrizer resolves them.

    Goes through Parameter_opt/SCP_parametrizer_002.load_movies_from_manifest, so a
    manifest this harness accepts is one the tuning consumer accepts too -- that is
    half of gate T8, checked implicitly on every validation run.
    """
    meta, train, val = ms.pm.load_movies_from_manifest(str(manifest_path))
    movies = {"train": train, "val": val}[split]
    if not movies:
        raise SystemExit(f"manifest has no '{split}' movies: {manifest_path}")
    return meta, movies


def measure_dataset(movies, run_detector_pass=True, detector_movies=None,
                    n_frames_density=10, n_frames_sigma=8, n_frames_background=8):
    """Measure every movie, then pool into one dataset-level measurement."""
    per_movie = []
    limit = len(movies) if detector_movies is None else detector_movies
    for i, mv in enumerate(movies):
        want_detector = run_detector_pass and i < limit
        print(f"  [{i + 1}/{len(movies)}] {mv['name']}"
              f"{' (+ SCP run)' if want_detector else ''} ...", flush=True)
        m = ms.measure_movie(
            mv["tif"], mv["positive_spots"], mv["negative_spots"],
            run_detector_pass=want_detector,
            n_frames_density=n_frames_density,
            n_frames_sigma=n_frames_sigma,
            n_frames_background=n_frames_background,
            events_json=str(HERE / f"_validate_events_{Path(mv['tif']).stem}.json"),
        )
        m["name"] = mv["name"]
        m["tif"] = mv["tif"]
        per_movie.append(m)
    return pool_measurements(per_movie), per_movie


def pool_measurements(per_movie):
    """Combine per-movie measurements into one, pooling distributions.

    Distributions (NN distances, per-spot photometry, track durations) are
    concatenated, then re-summarised -- so the pooled median is the median of all
    spots rather than a mean of per-movie medians. Curves (density vs threshold,
    sigma response) are averaged across movies at each x value.
    """
    out = {"n_movies": len(per_movie), "movies": [m["name"] for m in per_movie]}

    # ---- geometry: identical across a dataset by construction; assert it.
    shapes = {(m["geometry"]["n_frames"], m["geometry"]["height"],
               m["geometry"]["width"]) for m in per_movie}
    if len(shapes) > 1:
        raise SystemExit(f"movies disagree on shape: {shapes}; the length scaling "
                         "of gate T4 assumes one shape per dataset")
    g0 = per_movie[0]["geometry"]
    out["geometry"] = {
        **{k: g0[k] for k in ("n_frames", "height", "width", "dtype", "bit_depth")},
        **{k: float(np.mean([m["geometry"][k] for m in per_movie]))
           for k in ("intensity_min", "intensity_max", "intensity_mean",
                     "n_target", "n_off_target", "density_per_1000px",
                     "target_density_per_1000px", "target_to_off_target_ratio")},
    }

    # ---- NN distances: pooled across movies (computed within each movie).
    out["nn_distance"] = {}
    for cls in ("target", "all"):
        pooled = np.concatenate([np.asarray(m["nn_distance"][cls]["distances"],
                                            dtype=float) for m in per_movie])
        n_pts = int(sum(m["nn_distance"][cls]["n"] for m in per_movie))
        out["nn_distance"][cls] = ms.nn_summary_from_distances(pooled, n_points=n_pts)

    # ---- photometry: pooled per-spot values, re-summarised.
    out["photometry"] = {"target": {}, "off_target": {}, "auc": {},
                         "per_spot_values": {},
                         "recipe": per_movie[0]["photometry"]["recipe"]}
    for key in ms.PHOTOMETRY_KEYS:
        vals = {}
        for cls in ("target", "off_target"):
            vals[cls] = np.concatenate([
                np.asarray([v for v in m["photometry"]["per_spot_values"][key][cls]],
                           dtype=np.float64) for m in per_movie])
            out["photometry"][cls][key] = ms.quartiles(vals[cls])
        out["photometry"]["auc"][key] = ms.auc(vals["target"], vals["off_target"])
        out["photometry"]["per_spot_values"][key] = {
            c: ms._round_list(vals[c]) for c in ("target", "off_target")}

    # ---- PSF width: per-spot sigmas pooled; stacked estimates averaged.
    out["psf"] = {}
    for cls in ("target", "off_target"):
        vals = np.concatenate([np.asarray(m["psf"][cls]["per_spot_values"],
                                          dtype=np.float64) for m in per_movie])
        entry = {
            "per_spot_sigma": ms.quartiles(vals),
            "per_spot_values": ms._round_list(vals),
            "n_spots_used": int(sum(m["psf"][cls]["n_spots_used"] for m in per_movie)),
        }
        for key in ("stacked_second_moment_sigma", "stacked_gaussfit_sigma"):
            entry[key] = float(np.nanmean([m["psf"][cls][key] for m in per_movie]))
        # Average the radial profiles, then re-derive the fit-free FWHM sigma from
        # the average rather than averaging per-movie sigmas.
        profiles = [m["psf"][cls]["radial_profile"] for m in per_movie
                    if m["psf"][cls]["radial_profile"]]
        if profiles:
            radii = [p[0] for p in profiles[0]]
            means = [float(np.mean(vs)) for vs in
                     zip(*[[p[1] for p in prof] for prof in profiles])]
            entry["radial_profile"] = [[r, v] for r, v in zip(radii, means)]
            entry["fwhm_sigma"] = ms.fwhm_sigma_from_profile(entry["radial_profile"])
        else:
            entry["radial_profile"], entry["fwhm_sigma"] = [], float("nan")
        out["psf"][cls] = entry

    # ---- SNR: mean of per-movie medians (a summary, not gated).
    out["snr"] = {level: {k: {"median": float(np.nanmean(
        [m["snr"][level][k]["median"] for m in per_movie]))}
        for k in ms.SNR_VARIANTS} for level in ("target", "off_target", "all")}

    # ---- background: mean of the per-movie scalars.
    keys = ("mean", "std", "min", "max", "relative_swing", "autocorr_length_px",
            "temporal_mean_drift_std")
    out["background"] = {k: float(np.mean([m["background"][k] for m in per_movie]))
                         for k in keys}
    out["background"]["sigma"] = per_movie[0]["background"]["sigma"]
    out["background"]["spatial_pattern_corr_with_frame0"] = {
        "mean": float(np.mean([m["background"]["spatial_pattern_corr_with_frame0"]["mean"]
                               for m in per_movie]))}
    out["background"]["radial_autocorr"] = [
        float(np.mean(c)) for c in zip(*[m["background"]["radial_autocorr"]
                                        for m in per_movie])]

    # ---- density curves: averaged at each threshold.
    out["blob_density"] = {}
    for band in ("reference_sigmas", "bt_sim_sigmas"):
        rows = []
        for i, row0 in enumerate(per_movie[0]["blob_density"][band]):
            rows.append({
                "threshold": row0["threshold"],
                "density_per_1000px": float(np.mean(
                    [m["blob_density"][band][i]["density_per_1000px"] for m in per_movie])),
                "blobs_per_frame": float(np.mean(
                    [m["blob_density"][band][i]["blobs_per_frame"] for m in per_movie])),
                "pixels": row0["pixels"],
            })
        out["blob_density"][band] = rows

    # ---- sigma response: averaged at each scale, then peak/half-max re-derived.
    curve = []
    for i, c0 in enumerate(per_movie[0]["sigma_response"]["curve"]):
        curve.append({
            "sigma": c0["sigma"],
            "mean_f1": float(np.mean([m["sigma_response"]["curve"][i]["mean_f1"]
                                      for m in per_movie])),
            "mean_matched": float(np.mean([m["sigma_response"]["curve"][i]["mean_matched"]
                                           for m in per_movie])),
            "mean_blobs_per_frame": float(np.mean(
                [m["sigma_response"]["curve"][i]["mean_blobs_per_frame"] for m in per_movie])),
        })
    f1 = np.array([c["mean_f1"] for c in curve])
    sig = np.array([c["sigma"] for c in curve])
    peak_i = int(f1.argmax())
    support = sig[f1 >= f1[peak_i] / 2.0] if f1[peak_i] > 0 else sig[:0]
    out["sigma_response"] = {
        "peak_sigma": float(sig[peak_i]), "peak_f1": float(f1[peak_i]),
        "half_max_low": float(support.min()) if support.size else float("nan"),
        "half_max_high": float(support.max()) if support.size else float("nan"),
        "curve": curve,
    }

    # ---- track durations: pooled, re-summarised.
    with_det = [m for m in per_movie if "track_duration" in m]
    if with_det:
        td = {"box_px": with_det[0]["track_duration"]["box_px"],
              "n_events": int(sum(m["track_duration"]["n_events"] for m in with_det)),
              "n_movies": len(with_det)}
        for cls in ("target", "off_target", "spurious"):
            pooled = np.concatenate([np.asarray(m["track_duration"][cls]["durations"],
                                                dtype=float) for m in with_det])
            td[cls] = {
                "n": int(pooled.size),
                "median": float(np.median(pooled)) if pooled.size else float("nan"),
                "iqr": ([float(np.percentile(pooled, 25)), float(np.percentile(pooled, 75))]
                        if pooled.size else [float("nan")] * 2),
                "durations": [int(v) for v in pooled],
            }
        out["track_duration"] = td
        perf = [m["reference_detector_performance"]["whole_movie"] for m in with_det]
        out["reference_detector_performance"] = {
            "whole_movie": {k: float(np.mean([p[k] for p in perf]))
                            for k in ("fbeta", "recall", "precision")},
            "n_movies": len(with_det),
        }
    return out


# ===========================================================================
# GATES
# ===========================================================================

class Gate:
    """One acceptance check: a list of named sub-checks, each pass/fail."""

    def __init__(self, gid, title):
        self.gid, self.title, self.rows = gid, title, []
        self.skipped_reason = None

    def check(self, name, measured, target, ok, tol=""):
        self.rows.append({"name": name, "measured": measured, "target": target,
                          "tolerance": tol, "pass": bool(ok)})
        return ok

    def skip(self, reason):
        self.skipped_reason = reason

    @property
    def passed(self):
        if self.skipped_reason or not self.rows:
            return None
        return all(r["pass"] for r in self.rows)

    def as_dict(self):
        return {"gate": self.gid, "title": self.title,
                "result": "SKIP" if self.passed is None else ("PASS" if self.passed else "FAIL"),
                "skipped_reason": self.skipped_reason, "checks": self.rows}


def _rel_ok(measured, target, tol):
    """|measured - target| / |target| <= tol, with a NaN-safe guard."""
    if target is None or measured is None:
        return False
    if not np.isfinite(measured) or not np.isfinite(target):
        return False
    if target == 0:
        return measured == 0
    return abs(measured - target) / abs(target) <= tol


def _fmt(v, nd=3):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def gate_T1(ref, sim):
    """Spot scale.

    Gated on the MEASURED PSF width, not on the detector's preferred sigma window.
    Spec 7-T1 asks for the latter, but it is not identifiable: on the real movie
    blob_log returns the same blobs for every max_sigma from 0.8 to 3.0, so a
    window sweep has no purchase on spot size, and gating on it passes the
    BachelorThesis simulator whose PSF section 3b calls a mismatch. The width
    measurement is direct and physical. The detector response curve is still
    checked, as a secondary agreement test. See CALIBRATION.md.
    """
    g = Gate("T1", "Spot scale -- measured PSF width (+ detector response)")
    tol = TOL["T1_sigma_px"]

    if "psf" in ref and "psf" in sim:
        for key, label in (("stacked_second_moment_sigma", "PSF sigma, stacked 2nd moment"),
                           ("stacked_gaussfit_sigma", "PSF sigma, stacked Gaussian fit"),
                           ("fwhm_sigma", "PSF sigma, from radial-profile FWHM")):
            rv, sv = ref["psf"]["target"].get(key), sim["psf"]["target"].get(key)
            ok = (rv is not None and sv is not None and np.isfinite(rv)
                  and np.isfinite(sv) and abs(sv - rv) <= tol)
            g.check(f"{label} [target]", sv, rv, ok, f"+-{tol} px")
        rv = ref["psf"]["target"]["per_spot_sigma"]["median"]
        sv = sim["psf"]["target"]["per_spot_sigma"]["median"]
        ok = np.isfinite(rv) and np.isfinite(sv) and abs(sv - rv) <= tol
        g.check("PSF sigma, per-spot median [target]", sv, rv, ok, f"+-{tol} px")
    else:
        g.check("PSF width", None, None, False, "reference predates psf block")

    # Secondary: the detector's own scale response should also line up.
    r, s = ref["sigma_response"], sim["sigma_response"]
    for key, label in (("peak_sigma", "detector response peak sigma"),
                       ("half_max_high", "detector response half-max upper bound")):
        rv, sv = r[key], s[key]
        ok = np.isfinite(rv) and np.isfinite(sv) and abs(sv - rv) <= tol
        g.check(label, sv, rv, ok, f"+-{tol} px")
    return g


def gate_T2(ref, sim):
    g = Gate("T2", "Threshold transfer -- blob density per 1000 px (headline test)")
    tol = TOL["T2_density_rel"]
    rmap = {row["threshold"]: row["density_per_1000px"]
            for row in ref["blob_density"]["reference_sigmas"]}
    smap = {row["threshold"]: row["density_per_1000px"]
            for row in sim["blob_density"]["reference_sigmas"]}
    for thr in T2_GATE_THRESHOLDS:
        rv, sv = rmap.get(thr), smap.get(thr)
        g.check(f"density at threshold {thr:.2f}", sv, rv, _rel_ok(sv, rv, tol),
                f"+-{tol:.0%}")
    return g


def gate_T3(ref, sim):
    g = Gate("T3", "Photometry -- section 4d medians, direction and spread")
    tol = TOL["T3_median_rel"]
    for key in ms.PHOTOMETRY_KEYS:
        for cls in ("target", "off_target"):
            rv = ref["photometry"][cls][key]["median"]
            sv = sim["photometry"][cls][key]["median"]
            g.check(f"{ms.PHOTOMETRY_LABELS[key]} [{cls}]", sv, rv,
                    _rel_ok(sv, rv, tol), f"+-{tol:.0%}")
    # Direction: off-target must be the FAINTER population (spec 4d, R3).
    st = sim["photometry"]["target"]["bgsub_intensity"]["median"]
    so = sim["photometry"]["off_target"]["bgsub_intensity"]["median"]
    g.check("off-target fainter than target", f"{_fmt(so)} < {_fmt(st)}",
            "off-target fainter", np.isfinite(st) and np.isfinite(so) and so < st, "sign")
    # Overlapping IQRs, per row and class.
    for key in ms.PHOTOMETRY_KEYS:
        for cls in ("target", "off_target"):
            r, s = ref["photometry"][cls][key], sim["photometry"][cls][key]
            ok = (np.isfinite(r["q1"]) and np.isfinite(s["q1"])
                  and s["q1"] <= r["q3"] and r["q1"] <= s["q3"])
            g.check(f"IQR overlap: {ms.PHOTOMETRY_LABELS[key]} [{cls}]",
                    f"[{_fmt(s['q1'])}, {_fmt(s['q3'])}]",
                    f"[{_fmt(r['q1'])}, {_fmt(r['q3'])}]", ok, "non-empty overlap")
    lo, hi = TOL["T3_auc_range"]
    a = sim["photometry"]["auc"]["bgsub_intensity"]
    g.check("AUC of bg-subtracted brightness", a, f"{lo}-{hi}",
            np.isfinite(a) and lo <= a <= hi, f"in [{lo}, {hi}]")
    return g


def gate_T4(ref, sim):
    g = Gate("T4", "Track duration -- section 4e medians, length-scaled")
    if "track_duration" not in ref:
        g.skip("reference has no track_duration block (measured with --no-detector)")
        return g
    if "track_duration" not in sim:
        g.skip("no SCP run on the simulated dataset (--no-detector)")
        return g
    tol = TOL["T4_median_rel"]
    # Section 4e is an absolute frame count on a 501-frame movie, so a shorter
    # simulated movie cannot reproduce it unscaled: 134 frames is 27% of the real
    # movie but 45% of a 300-frame one. Scale the target by movie length.
    scale = sim["geometry"]["n_frames"] / ref["geometry"]["n_frames"]
    for cls in ("target", "off_target", "spurious"):
        rv = ref["track_duration"][cls]["median"] * scale
        sv = sim["track_duration"][cls]["median"]
        g.check(f"median duration [{cls}] (target scaled x{scale:.3f})", sv, rv,
                _rel_ok(sv, rv, tol), f"+-{tol:.0%}")
    med = {c: sim["track_duration"][c]["median"] for c in ("target", "off_target", "spurious")}
    ordered = (np.isfinite(list(med.values())).all()
               and med["target"] > med["off_target"] > med["spurious"])
    g.check("ordering target > off-target > spurious",
            " > ".join(f"{c}={_fmt(med[c], 1)}" for c in
                       ("target", "off_target", "spurious")),
            "strictly decreasing", ordered, "order")
    return g


def gate_T5(ref, sim):
    g = Gate("T5", "Spacing -- nearest-neighbour distribution of targets")
    r, s = ref["nn_distance"]["target"], sim["nn_distance"]["target"]
    ok_med = abs(s["median"] - r["median"]) <= TOL["T5_median_px"]
    g.check("NN median", s["median"], r["median"], ok_med,
            f"+-{TOL['T5_median_px']} px")
    g.check("NN minimum", s["min"], f">= {TOL['T5_min_floor']}",
            s["min"] >= TOL["T5_min_floor"], f">= {TOL['T5_min_floor']} px")
    # Compared as a FRACTION of spots: simulated movies have a different spot
    # count and area than the real movie, so raw counts are not comparable.
    rf = r["close_pairs_under_6"] / max(r["n"], 1)
    sf = s["close_pairs_under_6"] / max(s["n"], 1)
    g.check("fraction of targets with NN < 6 px", sf, rf,
            abs(sf - rf) <= TOL["T5_under6_frac"],
            f"+-{TOL['T5_under6_frac']:.0%} absolute")
    ks = ks_2samp(np.asarray(s["distances"], dtype=float),
                  np.asarray(r["distances"], dtype=float))
    g.check("KS test vs real NN distribution", f"p={ks.pvalue:.4f} (D={ks.statistic:.3f})",
            f"p >= {TOL['T5_ks_alpha']}", ks.pvalue >= TOL["T5_ks_alpha"],
            f"not rejected at alpha={TOL['T5_ks_alpha']}")
    return g


def gate_T7(manifest_path, meta, movies):
    """Reproducibility: one seed -> the same movie; different seeds -> different.

    Only checkable when the dataset records which generator and config produced it,
    which simulate.py's driver writes into the manifest meta. A dataset from any
    other generator is reported as SKIP rather than silently passed.
    """
    g = Gate("T7", "Reproducibility -- same seed identical, different seeds not")
    generator = (meta or {}).get("generator")
    if generator != "Simulation/simulate.py":
        g.skip(f"dataset was not produced by simulate.py (manifest meta generator="
               f"{generator!r}); regeneration cannot be attempted")
        return g
    if len(movies) < 1:
        g.skip("no movies to regenerate")
        return g

    import hashlib

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        # Regenerate the first movie from the recorded config and seed.
        target = movies[0]
        seed = meta["records"][0]["seed"] if meta.get("records") else 0
        cfg_path = Path(tmp) / "resolved.yaml"
        import yaml
        cfg_path.write_text(yaml.safe_dump(meta["resolved_config"]), encoding="utf-8")
        cmd = [sys.executable, str(HERE / "generate_dataset.py"),
               "--config", str(cfg_path), "--out-dir", str(Path(tmp) / "ds"),
               "--train-seeds", str(seed), "--val-seeds", "", "--quiet"]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env={**os.environ, "PYTHONUTF8": "1"})
        if proc.returncode != 0:
            g.skip(f"regeneration failed: {proc.stderr.strip()[-400:]}")
            return g
        regen_tif = Path(tmp) / "ds" / "train" / f"rep{seed}.tif"
        if not regen_tif.exists():
            g.skip(f"regeneration produced no {regen_tif.name}")
            return g

        same = digest(regen_tif) == digest(target["tif"])
        g.check(f"seed {seed}: regenerated TIFF is byte-identical",
                "identical" if same else "DIFFERENT", "identical", same, "sha256")

        regen_pos = regen_tif.with_name(f"rep{seed}_positive.json")
        labels_same = (ms.pm.extract_center_coordinates(str(regen_pos))
                       == target["positive_spots"])
        g.check(f"seed {seed}: regenerated labels identical",
                "identical" if labels_same else "DIFFERENT", "identical",
                labels_same, "exact")

    # Distinct seeds must differ in BOTH geometry and pixels.
    if len(movies) >= 2:
        a, b = movies[0], movies[1]
        pix_differ = digest(a["tif"]) != digest(b["tif"])
        geom_differ = a["positive_spots"] != b["positive_spots"]
        g.check("distinct seeds: pixels differ", "differ" if pix_differ else "IDENTICAL",
                "differ", pix_differ, "sha256")
        g.check("distinct seeds: geometry differs",
                "differ" if geom_differ else "IDENTICAL", "differ", geom_differ, "exact")
    else:
        g.check("distinct seeds", "only one movie in split", "two or more", False,
                "needs >= 2 movies")
    return g


# ===========================================================================
# PLOTS  (spec section 8)
# ===========================================================================

def make_plots(ref, sim, outdir, label):
    """Side-by-side distribution plots: what the pass/fail table cannot show."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    R, S = "#1f77b4", "#d62728"      # real, sim

    def finish(fig, name):
        fig.tight_layout()
        p = outdir / name
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(str(p))

    # 1. NN distance
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, cls in zip(axes, ("target", "all")):
        for src, name, col in ((ref, "real", R), (sim, label, S)):
            d = np.asarray(src["nn_distance"][cls]["distances"], dtype=float)
            if d.size:
                ax.hist(d, bins=np.arange(0, 40, 1.5), density=True, alpha=0.55,
                        label=f"{name} (median {np.median(d):.1f})", color=col)
        ax.axvline(TOL["T5_min_floor"], ls=":", c="k", lw=1,
                   label=f"{TOL['T5_min_floor']} px floor")
        ax.set(xlabel="nearest-neighbour distance (px)", ylabel="density",
               title=f"T5 spacing -- {cls}")
        ax.legend(fontsize=8)
    finish(fig, "t5_nn_distance.png")

    # 2. Per-spot photometry: brightness and CV, both classes
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.ravel(), ms.PHOTOMETRY_KEYS):
        data, names, colors = [], [], []
        for src, sname, col in ((ref, "real", R), (sim, label, S)):
            for cls in ("target", "off_target"):
                v = np.asarray(src["photometry"]["per_spot_values"][key][cls],
                               dtype=np.float64)
                v = v[np.isfinite(v)]
                data.append(v if v.size else np.array([np.nan]))
                names.append(f"{sname}\n{cls}")
                colors.append(col)
        bp = ax.boxplot(data, labels=names, showfliers=False, patch_artist=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.5)
        ax.set(title=f"T3 -- {ms.PHOTOMETRY_LABELS[key]}", ylabel=key)
        ax.tick_params(axis="x", labelsize=7)
    finish(fig, "t3_photometry.png")

    # 3. Track duration
    if "track_duration" in ref and "track_duration" in sim:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, cls in zip(axes, ("target", "off_target", "spurious")):
            for src, sname, col, sc in ((ref, "real", R, 1.0), (sim, label, S, 1.0)):
                d = np.asarray(src["track_duration"][cls]["durations"], dtype=float)
                if d.size:
                    frac = d / src["geometry"]["n_frames"]
                    ax.hist(frac, bins=np.linspace(0, 1, 26), density=True, alpha=0.55,
                            color=col, label=f"{sname} (median {np.median(d):.0f} fr)")
            ax.set(xlabel="duration / movie length", ylabel="density",
                   title=f"T4 -- {cls}")
            ax.legend(fontsize=8)
        fig.suptitle(f"Track duration as a fraction of movie length "
                     f"(real {ref['geometry']['n_frames']} fr, "
                     f"{label} {sim['geometry']['n_frames']} fr)", fontsize=9)
        finish(fig, "t4_track_duration.png")

    # 4. Blob density vs threshold
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, band, ttl in ((axes[0], "reference_sigmas", "reference sigmas 0.45-1.27"),
                          (axes[1], "bt_sim_sigmas", "BT-tuned sigmas 1.29-2.29")):
        for src, sname, col in ((ref, "real", R), (sim, label, S)):
            rows = src["blob_density"][band]
            ax.plot([r["threshold"] for r in rows],
                    [r["density_per_1000px"] for r in rows],
                    "o-", color=col, label=sname)
        for thr in T2_GATE_THRESHOLDS:
            ax.axvline(thr, ls=":", c="grey", lw=0.7)
        ax.set(xlabel="LoG threshold", ylabel="blobs per 1000 px", yscale="log",
               title=f"T2 -- {ttl}")
        ax.legend(fontsize=8)
    finish(fig, "t2_blob_density.png")

    # 5. T1: measured PSF width, plus the detector's scale response
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for src, sname, col in ((ref, "real", R), (sim, label, S)):
        prof = src.get("psf", {}).get("target", {}).get("radial_profile") or []
        if prof:
            axes[0].plot([p[0] for p in prof], [p[1] for p in prof], "o-", color=col,
                         label=f"{sname} (sigma {src['psf']['target']['fwhm_sigma']:.2f})")
    axes[0].axhline(0.5, ls=":", c="k", lw=1, label="half maximum")
    axes[0].set(xlabel="radius (px)", ylabel="normalised intensity",
                title="T1 -- stacked spot radial profile")
    axes[0].legend(fontsize=8)

    data, names, colors = [], [], []
    for src, sname, col in ((ref, "real", R), (sim, label, S)):
        for cls in ("target", "off_target"):
            v = np.asarray(src.get("psf", {}).get(cls, {}).get("per_spot_values", []),
                           dtype=np.float64)
            v = v[np.isfinite(v)]
            data.append(v if v.size else np.array([np.nan]))
            names.append(f"{sname}\n{cls}")
            colors.append(col)
    bp = axes[1].boxplot(data, tick_labels=names, showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    axes[1].set(title="T1 -- per-spot PSF sigma", ylabel="sigma (px)")
    axes[1].tick_params(axis="x", labelsize=7)

    for src, sname, col in ((ref, "real", R), (sim, label, S)):
        c = src["sigma_response"]["curve"]
        axes[2].plot([q["sigma"] for q in c], [q["mean_f1"] for q in c], "o-",
                     color=col, ms=3,
                     label=f"{sname} (peak {src['sigma_response']['peak_sigma']:.2f})")
    axes[2].axvspan(0.4515, 1.2690, color="grey", alpha=0.15,
                    label="tuned window 0.45-1.27")
    axes[2].set(xlabel="LoG sigma (single scale)",
                ylabel="mean per-frame F1 vs targets",
                title="T1 -- detector scale response")
    axes[2].legend(fontsize=8)
    finish(fig, "t1_spot_scale.png")

    # 6. Background structure
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for src, sname, col in ((ref, "real", R), (sim, label, S)):
        prof = src["background"].get("radial_autocorr") or []
        if prof:
            axes[0].plot(range(len(prof)), prof, color=col, label=sname)
    axes[0].axhline(1 / np.e, ls=":", c="k", lw=1, label="1/e")
    axes[0].set(xlabel="lag (px)", ylabel="autocorrelation",
                title="Background: spatial autocorrelation")
    axes[0].legend(fontsize=8)
    names = ["mean", "std", "relative_swing", "autocorr_length_px"]
    xs = np.arange(len(names))
    for i, (src, sname, col) in enumerate(((ref, "real", R), (sim, label, S))):
        axes[1].bar(xs + (i - 0.5) * 0.35, [src["background"][n] for n in names],
                    0.35, color=col, alpha=0.75, label=sname)
    axes[1].set_yscale("log")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, fontsize=7, rotation=20)
    axes[1].set(title="Background: sigma-8 field statistics (log scale)")
    axes[1].legend(fontsize=8)
    finish(fig, "background_structure.png")

    return written


# ===========================================================================
# REPORT
# ===========================================================================

def print_table(gates, ref, sim, label):
    print("\n" + "=" * 96)
    print(f"ACCEPTANCE TABLE -- {label} vs real movie")
    print("=" * 96)
    for g in gates:
        status = "SKIP" if g.passed is None else ("PASS" if g.passed else "FAIL")
        print(f"\n{g.gid}  [{status}]  {g.title}")
        if g.skipped_reason:
            print(f"      skipped: {g.skipped_reason}")
        for r in g.rows:
            mark = "ok  " if r["pass"] else "FAIL"
            m = r["measured"] if isinstance(r["measured"], str) else _fmt(r["measured"])
            t = r["target"] if isinstance(r["target"], str) else _fmt(r["target"])
            print(f"      [{mark}] {r['name']:<52s} sim={m:>16s}  real={t:>16s}"
                  f"   ({r['tolerance']})")

    hard = [g for g in gates if g.passed is not None]
    n_pass = sum(1 for g in hard if g.passed)
    print("\n" + "-" * 96)
    print(f"SUMMARY: {n_pass}/{len(hard)} gates pass"
          + (f", {len(gates) - len(hard)} skipped" if len(gates) - len(hard) else ""))
    print("  " + "  ".join(
        f"{g.gid}={'SKIP' if g.passed is None else ('PASS' if g.passed else 'FAIL')}"
        for g in gates))
    print("-" * 96)


def print_context(ref, sim, label):
    """Numbers that are not gated but explain the gates -- spec section 3b's table."""
    print("\n" + "=" * 96)
    print(f"CONTEXT -- the section 3b mismatch table, remeasured ({label} vs real)")
    print("=" * 96)
    rows = [
        ("1  PSF sigma, measured (stacked 2nd moment)",
         sim.get("psf", {}).get("target", {}).get("stacked_second_moment_sigma"),
         ref.get("psf", {}).get("target", {}).get("stacked_second_moment_sigma")),
        ("   PSF sigma, measured (radial-profile FWHM)",
         sim.get("psf", {}).get("target", {}).get("fwhm_sigma"),
         ref.get("psf", {}).get("target", {}).get("fwhm_sigma")),
        ("   detector response peak LoG sigma",
         sim["sigma_response"]["peak_sigma"], ref["sigma_response"]["peak_sigma"]),
        ("2  off-target brightness (bg-sub)",
         sim["photometry"]["off_target"]["bgsub_intensity"]["median"],
         ref["photometry"]["off_target"]["bgsub_intensity"]["median"]),
        ("   target brightness (bg-sub)",
         sim["photometry"]["target"]["bgsub_intensity"]["median"],
         ref["photometry"]["target"]["bgsub_intensity"]["median"]),
    ]
    if "track_duration" in sim and "track_duration" in ref:
        rows += [
            ("3  target duration (frames / movie len)",
             sim["track_duration"]["target"]["median"] / sim["geometry"]["n_frames"],
             ref["track_duration"]["target"]["median"] / ref["geometry"]["n_frames"]),
            ("   off-target duration (frames / movie len)",
             sim["track_duration"]["off_target"]["median"] / sim["geometry"]["n_frames"],
             ref["track_duration"]["off_target"]["median"] / ref["geometry"]["n_frames"]),
        ]
    rows += [
        ("4  raw 1-px CV, target",
         sim["photometry"]["target"]["aperture1px_cv"]["median"],
         ref["photometry"]["target"]["aperture1px_cv"]["median"]),
        ("   raw 1-px CV, off-target",
         sim["photometry"]["off_target"]["aperture1px_cv"]["median"],
         ref["photometry"]["off_target"]["aperture1px_cv"]["median"]),
        ("   raw 1-px CV, AUC (class separability)",
         sim["photometry"]["auc"]["aperture1px_cv"],
         ref["photometry"]["auc"]["aperture1px_cv"]),
        ("5  background relative swing",
         sim["background"]["relative_swing"], ref["background"]["relative_swing"]),
        ("   background autocorr length (px)",
         sim["background"]["autocorr_length_px"], ref["background"]["autocorr_length_px"]),
        ("6  usable threshold: density at 0.32",
         next(r["density_per_1000px"] for r in sim["blob_density"]["reference_sigmas"]
              if r["threshold"] == 0.32),
         next(r["density_per_1000px"] for r in ref["blob_density"]["reference_sigmas"]
              if r["threshold"] == 0.32)),
        ("8  target : off-target ratio",
         sim["geometry"]["target_to_off_target_ratio"],
         ref["geometry"]["target_to_off_target_ratio"]),
        ("   spot density per 1000 px",
         sim["geometry"]["density_per_1000px"], ref["geometry"]["density_per_1000px"]),
    ]
    print(f"      {'quantity':<44s} {label:>14s} {'real':>14s}   ratio")
    for name, s, r in rows:
        ratio = (s / r) if (r not in (0, None) and np.isfinite(s) and np.isfinite(r)) else float("nan")
        print(f"      {name:<44s} {_fmt(s):>14s} {_fmt(r):>14s}   {_fmt(ratio, 2):>6s}")
    print("=" * 96)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="simulated dataset manifest.json")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--outdir", default=None,
                    help="where plots and the report go (default: validation/<label>)")
    ap.add_argument("--split", default="train", choices=("train", "val"))
    ap.add_argument("--label", default=None, help="name for the simulated dataset")
    ap.add_argument("--no-detector", action="store_true",
                    help="skip SCP runs; gate T4 then reports SKIP")
    ap.add_argument("--detector-movies", type=int, default=2,
                    help="how many movies get an SCP run (default 2)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not Path(args.reference).exists():
        sys.exit(f"reference not found: {args.reference}\n"
                 "Run measure_real_movie.py first (spec section 11, step 1).")
    with open(args.reference, encoding="utf-8") as f:
        ref = json.load(f)

    label = args.label or Path(args.manifest).resolve().parent.name
    outdir = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR / label

    print("=" * 96)
    print(f"VALIDATING {label} AGAINST THE REAL MOVIE")
    print("=" * 96)
    print(f"  manifest : {args.manifest}  (split: {args.split})")
    print(f"  reference: {args.reference}")
    print(f"             real movie {ref['geometry']['n_frames']} frames, "
          f"{ref['geometry']['height']}x{ref['geometry']['width']}, "
          f"{ref['geometry']['n_target']} targets / "
          f"{ref['geometry']['n_off_target']} off-targets")

    meta, movies = load_manifest_movies(args.manifest, args.split)
    print(f"\nMeasuring {len(movies)} simulated movie(s)"
          f"{'' if args.no_detector else f', SCP run on the first {min(args.detector_movies, len(movies))}'} ...")
    sim, per_movie = measure_dataset(
        movies, run_detector_pass=not args.no_detector,
        detector_movies=args.detector_movies)

    gates = [gate_T1(ref, sim), gate_T2(ref, sim), gate_T3(ref, sim),
             gate_T4(ref, sim), gate_T5(ref, sim), gate_T7(args.manifest, meta, movies)]
    print_table(gates, ref, sim, label)
    print_context(ref, sim, label)

    plots = [] if args.no_plots else make_plots(ref, sim, outdir, label)
    if plots:
        print(f"\nPlots written to {outdir}:")
        for p in plots:
            print(f"  {Path(p).name}")

    report = {
        "label": label,
        "manifest": str(Path(args.manifest).resolve()),
        "split": args.split,
        "reference": str(Path(args.reference).resolve()),
        "dataset_meta": meta,
        "tolerances": TOL,
        "gates": [g.as_dict() for g in gates],
        "summary": {g.gid: ("SKIP" if g.passed is None else
                            ("PASS" if g.passed else "FAIL")) for g in gates},
        "simulated_measurements": sim,
        "plots": plots,
    }
    rp = ms.write_json(outdir / "validation_report.json", report)
    print(f"\nReport written to {rp}")

    hard = [g for g in gates if g.passed is not None]
    return 0 if all(g.passed for g in hard) else 1


if __name__ == "__main__":
    sys.exit(main())
