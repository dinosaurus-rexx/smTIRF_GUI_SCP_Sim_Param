"""
Derive config/real_movie_reference.json from the real annotated movie.

Step 1 of SIMULATOR_SPEC.md section 11, and the file every other piece of the
simulator work reads its targets from. Re-deriving beats trusting the spec's prose:
if a new real recording is annotated, drop it in, re-run this, and the calibration
targets and every acceptance gate follow it automatically.

What it measures (spec section in brackets):
  * geometry, class counts, spot density                              [4a]
  * per-spot signal-to-noise, several definitions                     [4b]
  * nearest-neighbour distance distribution, targets and all          [4c]
  * per-spot aperture photometry and class separability (AUC)         [4d]
  * detector-mediated track duration per annotation class             [4e]
  * sigma-8 background structure, incl. autocorrelation and drift     [4g]
  * blob density vs threshold, and the LoG scale response             [7-T1, 7-T2]

All measurements come from measurements.measure_movie(), the same function the
validation harness applies to simulated movies -- so the two sides of every
acceptance comparison are assembled by identical code.

Status of the reproduction (see CALIBRATION.md for the full account):
  4a, 4c, 4d   reproduce the spec's numbers exactly
  4e, 4f       reproduce (target duration 135 vs 134; held-out R/P/F1 exact)
  4b           does NOT reproduce the quoted "~11"; that number's definition was
               never recorded. Four candidate definitions are reported instead.
  4g           newly measured here -- the spec left it open.

Run:
    cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/Simulation
    python measure_real_movie.py                 # full, includes one SCP run
    python measure_real_movie.py --no-detector   # skip the ~2 min SCP run
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")      # SCP imports pyplot at module scope
os.environ.setdefault("PYTHONUTF8", "1")        # trap 2
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np

import measurements as ms

HERE = Path(__file__).resolve().parent
# Simulation -> smTIRF_GUI_SCP_Sim_Param -> smTIRF_GUI_SCP_Sim_Param -> Const
BT_REPO = HERE.parent.parent.parent / "BachelorThesis-FRETAlgorithms"
REALDATA = BT_REPO / "realdata"

DEFAULT_TIF = REALDATA / "vid_ch1.tiff"
DEFAULT_COORD = REALDATA / "coord.npy"
DEFAULT_CLASS = REALDATA / "class.npy"
DEFAULT_OUT = HERE / "config" / "real_movie_reference.json"

# Frame sampling. The photometry recipe of section 4d fixes every 5th frame; the
# rest are cheap enough to be generous with but not free, so they sample evenly.
N_FRAMES_BACKGROUND = 12
N_FRAMES_DENSITY = 10
N_FRAMES_SIGMA = 8

# Spec section 4d's own values, printed alongside the measurement so any drift is
# visible immediately rather than having to be looked up.
SPEC_4D = {
    "bgsub_intensity": (60.6, 18.9, 0.909),
    "bgsub_cv": (0.791, 1.596, 0.748),
    "aperture1px_intensity": (220.5, 164.9, 0.899),
    "aperture1px_cv": (0.305, 0.258, 0.580),
}
SPEC_4E = {"target": 134, "off_target": 41, "spurious": 25}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", default=str(DEFAULT_TIF))
    ap.add_argument("--coord", default=str(DEFAULT_COORD))
    ap.add_argument("--classes", default=str(DEFAULT_CLASS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-detector", action="store_true",
                    help="skip the section 4e SCP run (the only slow step)")
    ap.add_argument("--band", default=None, metavar="XMIN,XMAX",
                    help="restrict LABEL-derived targets to an x band, e.g. '0,256'. "
                         "Used to build a reference that never saw the held-out test "
                         "band's annotations -- see the note below.")
    args = ap.parse_args()

    for p in (args.tif, args.coord, args.classes):
        if not Path(p).exists():
            sys.exit(f"missing input: {p}")

    print("=" * 78)
    print("MEASURING THE REAL MOVIE")
    print("=" * 78)
    print(f"  movie : {args.tif}")
    print(f"  labels: {Path(args.coord).name}, {Path(args.classes).name}")
    if not args.no_detector:
        print("  (includes one SCP run over 501 frames -- about two minutes)")

    movie = ms.load_movie(args.tif)
    (targets, offtargets), classes = ms.load_labels_npy(args.coord, args.classes)

    # ---- optional band restriction -------------------------------------------
    # Restricts the LABEL-derived targets (photometry, PSF width, nearest-neighbour
    # spacing, track duration) to an x band. Pixel-derived targets -- the background
    # field, the noise level, the blob-density-vs-threshold curve -- stay measured
    # over the whole frame, deliberately.
    #
    # The distinction is what makes a held-out claim honest. On a new recording you
    # would have every pixel available but no labels, so using unlabelled pixels from
    # the whole field is realistic; using the test band's ANNOTATIONS is not. Cropping
    # the frame instead would also be wrong for a second reason: SCP min-max normalises
    # each frame (trap 3), so a 256x256 crop has a different maximum from a 256x512
    # frame and its density targets would not apply to a full-size simulated movie.
    band = None
    if args.band:
        lo, hi = (float(v) for v in args.band.split(","))
        band = (lo, hi)
        n_t, n_o = len(targets), len(offtargets)
        targets = [s for s in targets if lo <= s[0] < hi]
        offtargets = [s for s in offtargets if lo <= s[0] < hi]
        print(f"\n  BAND RESTRICTION x[{lo:.0f}, {hi:.0f}): labels {n_t}->{len(targets)} "
              f"target, {n_o}->{len(offtargets)} off-target")
        print("  Label-derived targets use this band only; pixel-derived targets "
              "(background,\n  noise, blob density) still use the whole frame -- "
              "unlabelled pixels are not a leak.")

    ref = ms.measure_movie(
        args.tif, targets, offtargets, movie=movie,
        run_detector_pass=not args.no_detector,
        n_frames_background=N_FRAMES_BACKGROUND,
        n_frames_density=N_FRAMES_DENSITY,
        n_frames_sigma=N_FRAMES_SIGMA,
        events_json=str(HERE / "_real_reference_events.json"),
        bands=((f"band_x{int(band[0])}_{int(band[1])}", band),) if band else
              (("val_x0_256", (0, 256)), ("test_x256_512", (256, 512))),
        label_band=band,
    )
    ref["geometry"]["n_moving"] = int(np.sum(classes == ms.CLASS_MOVING))
    if band:
        # Spot counts and density must reflect the band's AREA, or the simulator would
        # be told to place band-many spots over a full-size frame.
        area_1000 = (ref["geometry"]["height"] * (band[1] - band[0])) / 1000.0
        ref["geometry"]["band"] = [band[0], band[1]]
        ref["geometry"]["density_per_1000px"] = round(
            (len(targets) + len(offtargets)) / area_1000, 4)
        ref["geometry"]["target_density_per_1000px"] = round(len(targets) / area_1000, 4)
        ref["label_band_restriction"] = {
            "x_range": [band[0], band[1]],
            "applies_to": ["photometry", "psf", "nn_distance", "track_duration",
                           "sigma_response", "reference_detector_performance"],
            "not_applied_to": ["background", "blob_density", "geometry.n_frames/height/width"],
            "why": "label-derived targets must not see the held-out test band; "
                   "unlabelled pixels may, since a new recording would supply them",
        }
    ref["provenance"] = {
        "movie": str(Path(args.tif).resolve()),
        "coord_npy": str(Path(args.coord).resolve()),
        "class_npy": str(Path(args.classes).resolve()),
        "generated_by": "Simulation/measure_real_movie.py",
        "measured_by": "measurements.measure_movie (shared with validate_against_real.py)",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "note": "Targets for the simulator and for every acceptance gate in "
                "SIMULATOR_SPEC.md section 7. Re-derive by running "
                "measure_real_movie.py; never edit by hand.",
    }
    ref["reference_detector"] = {
        **ms.REFERENCE_DETECTOR,
        "use_bandpass": ms.REFERENCE_BANDPASS,
        "bandpass_bg_sigma": ms.REFERENCE_BANDPASS_BG_SIGMA,
        "event_nms_radius": ms.REFERENCE_EVENT_NMS_RADIUS,
        "intensity_aperture": ms.REFERENCE_INTENSITY_APERTURE,
    }

    # ---------------------------------------------------------------- report
    g = ref["geometry"]
    print(f"\n[4a] {g['n_frames']} frames, {g['height']}x{g['width']}, {g['dtype']} "
          f"(range {g['intensity_min']:.0f}-{g['intensity_max']:.0f})")
    print(f"     {g['n_annotated']} annotated: {g['n_target']} target, "
          f"{g['n_off_target']} off-target, {g['n_moving']} moving")
    print(f"     density {g['density_per_1000px']:.3f} per 1000 px, "
          f"ratio {g['target_to_off_target_ratio']:.2f} : 1        "
          f"[spec 4a: 2.55, 2.67 : 1]")

    print("\n[4b] per-spot SNR (several definitions -- see measurements.spot_snr) ...")
    for k in ms.SNR_VARIANTS:
        mark = "  <- primary" if k == ms.SNR_PRIMARY else ""
        print(f"     {k:30s} target {ref['snr']['target'][k]['median']:6.2f}  "
              f"off {ref['snr']['off_target'][k]['median']:6.2f}  "
              f"all {ref['snr']['all'][k]['median']:6.2f}{mark}")
    print("     [spec 4b quotes ~11; no variant reproduces it. The definition "
          "behind that number\n      was never recorded and no section 7 gate "
          "depends on it -- see CALIBRATION.md.]")

    nt, na = ref["nn_distance"]["target"], ref["nn_distance"]["all"]
    print("\n[4c] nearest-neighbour distances")
    print(f"     targets: min {nt['min']:.2f}  p1 {nt['p1']:.2f}  p2 {nt['p2']:.2f}  "
          f"p5 {nt['p5']:.2f}  p10 {nt['p10']:.2f}  p25 {nt['p25']:.2f}  "
          f"median {nt['median']:.2f}")
    print("     [spec 4c: 3.61, 5.00, 5.39, 6.32, 7.07, 9.22, 12.04]")
    print(f"     under 4px: {nt['close_pairs_under_4']}  under 6px: "
          f"{nt['close_pairs_under_6']}  under 8px: {nt['close_pairs_under_8']}"
          f"        [spec: 2, 12, 33]")
    print(f"     all spots: min {na['min']:.2f}  p1 {na['p1']:.2f}  p5 {na['p5']:.2f}"
          f"                    [spec: 3.61, 4.24, 5.83]")

    ph = ref["photometry"]
    print("\n[4d] aperture photometry at the annotated coordinates "
          "(spec value in brackets)")
    for key in ms.PHOTOMETRY_KEYS:
        t, o, a = ph["target"][key], ph["off_target"][key], ph["auc"][key]
        s = SPEC_4D[key]
        print(f"     {ms.PHOTOMETRY_LABELS[key]:36s} target {t['median']:8.3f} "
              f"[{s[0]:7.3f}]  off {o['median']:8.3f} [{s[1]:7.3f}]  "
              f"AUC {a:.3f} [{s[2]:.3f}]")
    print(f"     (windows clipped at the frame edge: "
          f"{ph['target'].get('n_edge_clipped', 0)} target, "
          f"{ph['off_target'].get('n_edge_clipped', 0)} off-target; all spots kept)")

    b = ref["background"]
    print(f"\n[4g] background structure at sigma={b['sigma']} "
          f"-- the target the spec left open")
    print(f"     mean {b['mean']:.2f}  std {b['std']:.2f}  "
          f"min {b['min']:.2f}  max {b['max']:.2f}")
    print(f"     relative swing (max-min)/mean = {b['relative_swing']:.3f}")
    print(f"     spatial autocorrelation length {b['autocorr_length_px']:.2f} px")
    print(f"     temporal: per-frame mean std {b['temporal_mean_drift_std']:.2f} "
          f"over range {b['temporal_mean_range'][0]:.1f}-"
          f"{b['temporal_mean_range'][1]:.1f}")
    print(f"     spatial pattern correlation with frame 0: min "
          f"{b['spatial_pattern_corr_with_frame0']['min']:.3f}, mean "
          f"{b['spatial_pattern_corr_with_frame0']['mean']:.3f}  "
          f"-> the field DRIFTS, it is not static")

    d = ref["reference_detector"]
    print(f"\n[7-T2] blob density vs threshold at sigma "
          f"{d['min_sigma']:.3f}-{d['max_sigma']:.3f} (bandpass on)")
    for row in ref["blob_density"]["reference_sigmas"]:
        print(f"     thr {row['threshold']:.2f} -> {row['blobs_per_frame']:8.2f} "
              f"blobs/frame = {row['density_per_1000px']:7.3f} per 1000 px")
    bt05 = next(r for r in ref["blob_density"]["bt_sim_sigmas"] if r["threshold"] == 0.05)
    print(f"     at sigma 1.29-2.29, thr 0.05: {bt05['density_per_1000px']:.3f} "
          f"per 1000 px        [spec 7-T2: 1.45-1.52]")

    p = ref["psf"]
    print("\n[7-T1 / R1] measured PSF width -- the physical spot size")
    for cls in ("target", "off_target"):
        q = p[cls]
        print(f"     {cls:11s} stacked 2nd moment {q['stacked_second_moment_sigma']:.3f}  "
              f"Gaussian fit {q['stacked_gaussfit_sigma']:.3f}  "
              f"per-spot median {q['per_spot_sigma']['median']:.3f}  "
              f"FWHM-implied {q['fwhm_sigma']:.3f}   (n={q['n_spots_used']})")
    print("     [spec 3b row 1 expects the real PSF to be well BELOW the BT "
          "simulator's 1.5.\n      It is not: measured about 1.5, the same as BT's. "
          "See CALIBRATION.md.]")

    sr = ref["sigma_response"]
    print("\n[7-T1] LoG single-scale response sweep (secondary -- not identifiable)")
    print(f"     peak at sigma {sr['peak_sigma']:.2f} (mean per-frame F1 "
          f"{sr['peak_f1']:.4f}); half-max support "
          f"{sr['half_max_low']:.2f}-{sr['half_max_high']:.2f}")
    print("     response by scale: " + "  ".join(
        f"{c['sigma']:.2f}:{c['mean_f1']:.3f}" for c in sr["curve"]
        if abs(c["sigma"] * 100 - round(c["sigma"] * 100)) < 1e-6
        and round(c["sigma"] * 100) in (15, 30, 45, 60, 90, 130, 150, 200, 250)))
    print("     [the tuned window is 0.45-1.27: its lower bound sits at the "
          "response collapse\n      edge, its upper bound is inert -- nothing is "
          "detected at those scales]")

    if "track_duration" in ref:
        td = ref["track_duration"]
        print(f"\n[4e] detector-mediated track duration ({td['n_events']} events: "
              f"{td['target']['n']} on targets, {td['off_target']['n']} on "
              f"off-targets,\n     {td['spurious']['n']} on nothing annotated)")
        for key in ("target", "off_target", "spurious"):
            k = td[key]
            print(f"     {key:11s} median {k['median']:6.1f} frames  "
                  f"IQR {k['iqr'][0]:.0f}-{k['iqr'][1]:.0f}"
                  f"        [spec 4e: {SPEC_4E[key]}]")
        # Band names follow whatever --band asked for, so look them up rather than
        # hard-coding: under a band restriction there is no 'test_x256_512' key.
        bands = ref["reference_detector_performance"]["bands"]
        for name, perf in bands.items():
            note = ("   [spec 4f: 94.6% / 76.6% / 0.847]"
                    if name == "test_x256_512" else "")
            print(f"\n[4f] band {name}: R {perf['recall']:.1%} "
                  f"P {perf['precision']:.1%} F1 {perf['fbeta']:.3f}{note}")
    else:
        print("\n[4e] SKIPPED (--no-detector): the section 4e and 4f blocks are "
              "absent from the\n     reference, and gate T4 will report a missing "
              "target rather than a pass.")

    out = ms.write_json(args.out, ref)
    print("\n" + "=" * 78)
    print(f"Reference written to {out}")
    print("=" * 78)
    return ref


if __name__ == "__main__":
    main()
