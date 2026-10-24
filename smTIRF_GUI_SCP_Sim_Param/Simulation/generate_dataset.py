"""
Build a seeded train/validation dataset of real-calibrated simulated movies.

Driver for simulate.py. Emits every output SIMULATOR_SPEC.md section 6 requires, in
the layout its consumers already expect:

    <out_dir>/manifest.json
    <out_dir>/{train,val}/rep<seed>.tif             (T, H, W) uint16      [6d]
    <out_dir>/{train,val}/rep<seed>_positive.json   [{"center": {...}}]   [6a]
    <out_dir>/{train,val}/rep<seed>_negative.json
    <out_dir>/{train,val}/rep<seed>_coord.npy       (N, 2) as [y, x]      [6c]
    <out_dir>/{train,val}/rep<seed>_class.npy       0=target 1=off 2=moving

Both ground-truth formats are written (R7), which lets the same movies feed the SCP
parametrizer and the BachelorThesis benchmark pipeline. The label JSON is (x, y) and
the .npy is [y, x]; that flip is the most bug-prone thing in this codebase (trap 1),
so it happens in exactly one place -- simulate.ground_truth_arrays -- and
tests/test_contracts.py pins it against a deliberately asymmetric frame.

The previous occupant of this filename, which drove the old smTIRF_simulation.py,
is preserved as legacy_generate_dataset_smtirf.py so the recorded baseline numbers
stay reproducible (section 10).

Run:
    cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/Simulation
    python generate_dataset.py                       # defaults below
    python generate_dataset.py --frames 300 --train-seeds 0,1,2 --val-seeds 1000,1001
    python generate_dataset.py --out-dir dataset_small --frames 150

Then:
    python validate_against_real.py --manifest dataset/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import tifffile

import simulate as sm

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "dataset"
DEFAULT_CONFIG = HERE / "config" / "real_calibrated.yaml"

# Disjoint seed ranges so a train movie and a validation movie never coincide.
DEFAULT_TRAIN_SEEDS = [0, 1, 2]
DEFAULT_VAL_SEEDS = [1000, 1001]


def write_label_json(points_xy, path):
    """Section 6a: [{"center": {"x": .., "y": ..}}, ...], sub-pixel floats.

    Read by Parameter_opt/SCP_parametrizer_002.extract_center_coordinates, which
    ignores extra keys -- so per-spot metadata may be added alongside `center`, but
    `center` itself must be present and correctly named.
    """
    payload = [{"center": {"x": float(x), "y": float(y)}} for x, y in points_xy]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return len(payload)


def build_movie(cfg, seed, split_dir, stem=None, verbose=True):
    """Generate one movie plus both ground-truth formats; return a manifest record."""
    stem = stem or f"rep{seed}"
    movie, spots, meta = sm.simulate(cfg, seed=seed, verbose=verbose)

    tif = split_dir / f"{stem}.tif"
    # (T, H, W) single channel, axes tagged TYX so metadata readers agree with the
    # array. NEVER (T, C, H, W) -- section 6d, and scp_wrapper.py:load_frames is
    # the trap it refers to.
    tifffile.imwrite(str(tif), movie, imagej=True, metadata={"axes": "TYX"})

    pos_json = split_dir / f"{stem}_positive.json"
    neg_json = split_dir / f"{stem}_negative.json"
    n_pos = write_label_json(spots.get("target", []), pos_json)
    n_neg = write_label_json(spots.get("off_target", []), neg_json)

    coord, classes = sm.ground_truth_arrays(spots)
    coord_npy = split_dir / f"{stem}_coord.npy"
    class_npy = split_dir / f"{stem}_class.npy"
    np.save(coord_npy, coord)
    np.save(class_npy, classes)

    n_moving = int(np.sum(classes == sm.CLASS_IDS["moving"])) if len(classes) else 0
    n_unannotated = sum(len(spots.get(c, [])) for c in sm.UNLABELLED_CLASSES)
    if verbose:
        extra = ""
        if n_moving:
            extra += (f", {n_moving} moving -- in coord.npy but in NEITHER label "
                      "file, so a detection on one scores as a false positive")
        if n_unannotated:
            extra += (f", {n_unannotated} unannotated -- in NO ground-truth output "
                      "at all, reproducing the real movie's unmarked emitters")
        print(f"    -> {tif.name}: {n_pos} positive, {n_neg} negative{extra}")

    return {
        "seed": int(seed),
        "tif": str(tif), "positive": str(pos_json), "negative": str(neg_json),
        "coord_npy": str(coord_npy), "class_npy": str(class_npy),
        "n_positive": n_pos, "n_negative": n_neg,
        "n_moving_unlabelled": n_moving, "n_unannotated": n_unannotated,
        "photophysics_realised": meta["photophysics_realised"],
        "background_realised": meta["background_realised"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--train-seeds", default=",".join(map(str, DEFAULT_TRAIN_SEEDS)))
    ap.add_argument("--val-seeds", default=",".join(map(str, DEFAULT_VAL_SEEDS)))
    ap.add_argument("--frames", type=int, default=None,
                    help="override n_frames (the config's 501 matches the real movie)")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--scale-counts", action="store_true",
                    help="scale spot counts with frame area, preserving the measured "
                         "density of 2.55 spots per 1000 px when the frame size changes")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = sm.load_config(args.config)
    g = cfg["global_settings"]
    base_area = g["height"] * g["width"]
    if args.frames:
        g["n_frames"] = int(args.frames)
    if args.height:
        g["height"] = int(args.height)
    if args.width:
        g["width"] = int(args.width)
    if args.scale_counts:
        factor = (g["height"] * g["width"]) / base_area
        for cls in sm.CLASS_ORDER:
            if cls in cfg:
                cfg[cls]["count"] = int(round(cfg[cls]["count"] * factor))

    train_seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
    val_seeds = [int(s) for s in args.val_seeds.split(",") if s.strip()]
    overlap = set(train_seeds) & set(val_seeds)
    if overlap:
        sys.exit(f"train and val seeds overlap: {sorted(overlap)}. Distinct seeds are "
                 "what make the two splits independent (R6).")
    if not train_seeds:
        sys.exit("no train seeds given")

    out_dir = Path(args.out_dir)
    verbose = not args.quiet
    if verbose:
        counts = {c: cfg[c]["count"] for c in sm.active_classes(cfg)}
        density = sum(counts.values()) / (g["height"] * g["width"] / 1000.0)
        print("=" * 78)
        print(f"GENERATING REAL-CALIBRATED DATASET in {out_dir}")
        print("=" * 78)
        print(f"  config : {args.config}")
        print(f"  movie  : {g['n_frames']} frames, {g['height']}x{g['width']}, "
              f"{g['dtype']}, PSF sigma {g['psf_sigma']}")
        print(f"  spots  : {counts}  ({density:.2f} per 1000 px; real movie 2.55)")
        print(f"  seeds  : train {train_seeds}   val {val_seeds}")
        if g["n_frames"] != 501:
            print(f"  NOTE   : {g['n_frames']} frames, not the real movie's 501. Gate "
                  "T4 scales its duration\n           targets by movie length, but SCP "
                  "only keeps a track that reaches\n           min_pulsation_frames "
                  "(20 in the reference config) -- shortening the movie\n           is "
                  "not a free speed knob. See generate_dataset_bt.py's note.")

    manifest = {
        "meta": {
            "generator": "Simulation/simulate.py",
            "driver": "Simulation/generate_dataset.py",
            "config_path": str(Path(args.config).resolve()),
            "calibrated_against": "BachelorThesis-FRETAlgorithms/realdata/vid_ch1.tiff",
            "reference": "Simulation/config/real_movie_reference.json",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "train_seeds": train_seeds, "val_seeds": val_seeds,
            # The FULL resolved config (R6). Gate T7 regenerates a movie from this
            # and compares bytes, so it has to be complete enough to reproduce the
            # dataset on its own.
            "resolved_config": cfg,
            "records": [],
        }
    }

    for split, seeds in (("train", train_seeds), ("val", val_seeds)):
        if not seeds:
            manifest[split] = []
            continue
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for s in seeds:
            if verbose:
                print(f"\n  [{split}] seed {s}")
            records.append(build_movie(cfg, s, split_dir, verbose=verbose))
        manifest[split] = records
        manifest["meta"]["records"].extend(
            [{"split": split, "seed": r["seed"], "tif": r["tif"]} for r in records])

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str),
                             encoding="utf-8")
    if verbose:
        print(f"\nManifest written to {manifest_path}")
        print("\nNext:")
        print(f"  python validate_against_real.py --manifest {manifest_path}")
        print("  # then point DATASET_MANIFEST in Parameter_opt/SCP_parametrizer_002.py "
              "at it")
    return manifest_path


if __name__ == "__main__":
    main()
