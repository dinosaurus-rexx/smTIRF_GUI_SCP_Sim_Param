"""
Build an SCP tuning dataset from the BachelorThesis-FRETAlgorithms simulator.

Why this exists: tuning on smTIRF_simulation.py did not transfer to the real movie.
Its target spots pulse smoothly and never dim below the detection threshold, so the
search learned to demand a high intensity coefficient of variation (0.25); real spots
blink hard, drop out of the track entirely while dark, and therefore present a LOW
measured CV -- so that bar rejected them and recall collapsed to 7-18%.

The BachelorThesis simulator models the target differently: a two-state FRET HMM
(E~0.2/0.8, 0.95/0.05 transition matrix, mean dwell ~20 frames) redistributes light
between donor and acceptor in discrete jumps. Summed to one channel that produces
step-like intensity changes rather than a smooth sinusoid -- closer to real blinking.
It also supplies two distractor classes the smTIRF simulator lacks: aggregates and
mobile random-walkers.

Output layout matches Simulation/generate_dataset.py exactly, so
SCP_parametrizer_002.load_movies_from_manifest() consumes it unchanged:

    <out_dir>/manifest.json
    <out_dir>/{train,val}/rep<seed>.tif             single-channel (T, H, W)
    <out_dir>/{train,val}/rep<seed>_positive.json   [{"center": {"x":..,"y":..}}, ...]
    <out_dir>/{train,val}/rep<seed>_negative.json

Run:
    cd smTIRF_GUI_SCP_Sim_Param/Simulation
    python generate_dataset_bt.py
"""

import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")   # the BT script imports pyplot at module scope

import numpy as np
import tifffile as tiff
import yaml

HERE = Path(__file__).resolve().parent
# Simulation -> smTIRF_GUI_SCP_Sim_Param -> smTIRF_GUI_SCP_Sim_Param -> Const
BT_REPO = HERE.parent.parent.parent / "BachelorThesis-FRETAlgorithms"

# ============================================================================
# DATASET CONFIG
# ============================================================================
OUT_DIR = HERE / "dataset_bt"

# Which BachelorThesis config to borrow simulation settings from. realdata.yaml's
# noise settings (background 0.3, read 8) are the ones matched to the real movie's
# SNR (~11), which is the regime we need parameters to transfer INTO. Use
# config/snr709.yaml for a noisier (SNR ~6.5) variant.
BORROW_CONFIG = BT_REPO / "config" / "realdata.yaml"

# Scaled down from the benchmark's own 500 frames / 512 px / 2000 particles: SCP is
# per-frame LoG plus a cv2 contour call per spot, and a 60-trial search runs the
# detector on every train movie per trial. Particle counts keep spot DENSITY close to
# the real movie (334 spots over 256x512 = 1 per ~392 px^2; 160 over 256^2 = 1 per ~410).
#
# DURATION MATTERS MORE THAN IT LOOKS. SCP only keeps a track if the spot was detected
# in at least min_pulsation_frames frames, and per-frame detection recall is low at any
# usable threshold (~18% at 0.15, ~1% at 0.25). The real movie survives that because it
# has 501 frames to accumulate over. A 60-frame movie yields <1 detection per spot and
# scores a flat zero for EVERY parameter set, which silently makes the whole search
# degenerate -- shortening the movie is not a free speed knob. 300 frames keeps a spot
# detected ~54 times at threshold 0.15, comfortably above min_pulsation_frames.
N_FRAMES = 300
IMG_SIZE = 256
COUNTS = {"stationary": 90, "aggregate": 45, "moving": 25}

# Which simulated channel becomes the movie. 1 = acceptor: its per-spot intensity CV
# (0.193) lands closest to the real movie's (0.261) of the three options -- see
# to_single_channel() for the full measurement and why summing is wrong here.
CHANNEL = 1

# Disjoint seed ranges so train and validation movies never coincide.
TRAIN_SEEDS = [0, 1, 2]
VAL_SEEDS = [1000, 1001]

# Ground-truth class ids emitted by data_simulation.py, and how the parametrizer
# should read them. Movers are deliberately in NEITHER list: detecting one is a
# false positive (compute_metrics counts every detection unmatched to a positive),
# which is the correct penalty and mirrors the real movie, whose annotations only
# distinguish stationary (0) from aggregate (1).
POSITIVE_CLASS = 0   # stationary = the detection target
NEGATIVE_CLASS = 1   # aggregate  = off-target distractor


# ============================================================================
# DRIVING THE BACHELORTHESIS SIMULATOR
# ============================================================================
# data_simulation.py is a Snakemake script: it reads module-level `snakemake.*`
# globals rather than exposing a function. Rather than fork it, we exec it with a
# stand-in object providing the same attributes.

class _Attrs:
    """Namespace with the .get() that Snakemake's Params object also provides."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def build_sim_settings():
    """Borrow a BT config's simulation block, then scale it down and force all
    three particle classes on (realdata.yaml ships with `moving` deactivated)."""
    with open(BORROW_CONFIG) as f:
        sim = copy.deepcopy(yaml.safe_load(f)["simulation"])

    sim["global_settings"]["n_frames"] = N_FRAMES
    sim["global_settings"]["img_size"] = IMG_SIZE
    for cls, count in COUNTS.items():
        sim[cls]["activate"] = True
        sim[cls]["count"] = count
    return sim


def run_bt_simulation(seed, sim_settings, out_tiff, out_coords, out_classes):
    """Exec data_simulation.py once, with sim_seed set for full reproducibility."""
    script = BT_REPO / "scripts" / "data_analysis" / "data_simulation.py"
    # Resolve to absolute BEFORE the chdir below, or a relative output path would be
    # written inside the BachelorThesis repo instead of where the caller asked.
    out_tiff, out_coords, out_classes = (
        Path(p).resolve() for p in (out_tiff, out_coords, out_classes)
    )
    fake = _Attrs(
        output=_Attrs(tiff=str(out_tiff), coords=str(out_coords), classes=str(out_classes)),
        params=_Attrs(sim_settings=sim_settings, use_real_data=False, sim_seed=seed),
        wildcards=_Attrs(rep=str(seed)),
    )
    src = script.read_text(encoding="utf-8")
    g = {"__name__": "__bt_sim__", "__file__": str(script), "snakemake": fake}

    cwd = os.getcwd()
    os.chdir(BT_REPO)          # the script uses repo-relative paths on its realdata branch
    try:
        exec(compile(src, str(script), "exec"), g)
    finally:
        os.chdir(cwd)


# ============================================================================
# FORMAT CONVERSION
# ============================================================================

def to_single_channel(tiff_path, out_path, channel=CHANNEL):
    """
    (T, C, H, W) dual-channel -> (T, H, W) by taking ONE channel.

    Do NOT sum donor+acceptor here. FRET redistributes light between the two channels
    rather than creating it, so for a fixed total brightness the sum is constant and
    only shot noise survives. Measured on this dataset, median per-spot intensity CV:

        summed          0.014      <- signal destroyed
        donor (ch0)     0.190
        acceptor (ch1)  0.193
        REAL movie      0.261      <- the target
        smTIRF sim      0.789      <- the simulator that failed to transfer

    SCP's event classifier thresholds exactly this quantity, so summing would make
    every target spot look perfectly steady and unclassifiable. A single channel also
    matches the real test movie, which is single-channel to begin with.
    """
    video = tiff.imread(tiff_path)
    if video.ndim != 4:
        raise ValueError(f"expected (T,C,H,W) from the simulator, got {video.shape}")
    single = video[:, channel, :, :].astype(np.float32)
    tiff.imwrite(out_path, single, imagej=True, metadata={'axes': 'TYX'})
    return single.shape


def write_label_json(coords_yx, out_path):
    """Write spots in the detector's label format: [{"center": {"x":.., "y":..}}, ...]."""
    payload = [{"center": {"x": float(c[1]), "y": float(c[0])}} for c in coords_yx]
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return len(payload)


def build_movie(seed, split, split_dir, sim_settings):
    """Generate one movie + labels, returning a manifest record."""
    stem = f"rep{seed}"
    raw_tif = split_dir / f"_{stem}_dual.tiff"       # intermediate (T,C,H,W)
    coords_npy = split_dir / f"_{stem}_coord.npy"
    classes_npy = split_dir / f"_{stem}_class.npy"

    tif = split_dir / f"{stem}.tif"
    pos_json = split_dir / f"{stem}_positive.json"
    neg_json = split_dir / f"{stem}_negative.json"

    run_bt_simulation(seed, sim_settings, raw_tif, coords_npy, classes_npy)
    shape = to_single_channel(raw_tif, tif)

    coords = np.load(coords_npy)
    classes = np.load(classes_npy)
    n_pos = write_label_json(coords[classes == POSITIVE_CLASS], pos_json)
    n_neg = write_label_json(coords[classes == NEGATIVE_CLASS], neg_json)
    n_other = int(np.sum((classes != POSITIVE_CLASS) & (classes != NEGATIVE_CLASS)))

    for tmp in (raw_tif, coords_npy, classes_npy):
        tmp.unlink()

    print(f"  [{split}] {stem}: {shape} | {n_pos} positive, {n_neg} negative, "
          f"{n_other} mover(s) unlabelled (scored as false positives if detected)")

    return {"seed": seed, "tif": str(tif), "positive": str(pos_json),
            "negative": str(neg_json), "n_positive": n_pos, "n_negative": n_neg}


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not BT_REPO.is_dir():
        sys.exit(f"BachelorThesis repo not found at {BT_REPO}")

    sim_settings = build_sim_settings()
    g = sim_settings["global_settings"]
    print(f"Generating BT-simulator dataset in {OUT_DIR}")
    print(f"  borrowed from : {BORROW_CONFIG.name}")
    print(f"  movie         : {N_FRAMES} frames, {IMG_SIZE}x{IMG_SIZE}, "
          f"channel {CHANNEL} ({'acceptor' if CHANNEL else 'donor'})")
    print(f"  noise         : background={g['background_level']}, read_noise={g['read_noise']}")
    print(f"  particles     : {COUNTS}")
    print(f"  train seeds   : {TRAIN_SEEDS}   val seeds: {VAL_SEEDS}\n")

    manifest = {"meta": {
        "source": "BachelorThesis-FRETAlgorithms/scripts/data_analysis/data_simulation.py",
        "borrowed_config": BORROW_CONFIG.name,
        "frames": N_FRAMES, "img_size": IMG_SIZE, "counts": COUNTS,
        "background_level": g["background_level"], "read_noise": g["read_noise"],
        "channel": CHANNEL, "channel_note": "single channel (acceptor); summing "
                                            "cancels the FRET intensity variation",
    }}

    for split, seeds in (("train", TRAIN_SEEDS), ("val", VAL_SEEDS)):
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest[split] = [build_movie(s, split, split_dir, sim_settings) for s in seeds]

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print("Point DATASET_MANIFEST in the parametrizer at this file.")
    return manifest_path


if __name__ == "__main__":
    main()
