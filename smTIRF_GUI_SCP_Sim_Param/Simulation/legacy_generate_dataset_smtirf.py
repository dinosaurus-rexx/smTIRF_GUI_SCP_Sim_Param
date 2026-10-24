"""
LEGACY driver for the old smTIRF_simulation.py. Preserved, not maintained.

This is the file that used to be Simulation/generate_dataset.py, renamed when the
real-calibrated simulator took that name. SIMULATOR_SPEC.md section 10 requires the
old simulator to stay reproducible until gates T1-T6 pass, and this is its driver --
without it, the recorded "tuned on BT sim -> real test F1 0.256" baseline could not
be regenerated. Nothing imports it (the experiments/ drivers call
smTIRF_simulation.generate directly), so the rename is safe.

Do not add to this. For new work use generate_dataset.py, which drives simulate.py.

Generate a reproducible train / validation dataset of simulated smTIRF movies.

Each movie comes from a distinct random seed (so geometry AND noise differ), and
is written alongside detector-format positive/negative label JSONs. A manifest.json
ties them together for the SCP parametrizer to consume.

The real (non-simulated) labelled movie is NOT produced here -- it is the held-out
TEST set and is configured separately in the parametrizer.

Run:
    cd Simulation
    python legacy_generate_dataset_smtirf.py     # uses the defaults below
"""

import json
from pathlib import Path

import smTIRF_simulation as sim

# ============================================================================
# DATASET CONFIG  ("Standard" size: 8 train + 3 val, ~300 frames each)
# ============================================================================
OUT_DIR = Path(__file__).resolve().parent / "dataset"
FRAMES = 300
NOISE_PROFILE = "high"          # must be a key in sim.NOISE_PROFILES

# Disjoint seed ranges so train and validation movies never coincide.
TRAIN_SEEDS = list(range(0, 8))          # 0..7
VAL_SEEDS = list(range(1000, 1003))      # 1000..1002


def _generate_split(split_name, seeds, split_dir):
    split_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds:
        stem = f"rep{seed}"
        rec = sim.generate(
            seed=seed,
            out_tif=str(split_dir / f"{stem}.tif"),
            positive_json=str(split_dir / f"{stem}_positive.json"),
            negative_json=str(split_dir / f"{stem}_negative.json"),
            frames=FRAMES,
            noise_profile=NOISE_PROFILE,
            verbose=True,
        )
        records.append(rec)
    print(f"\n{split_name}: generated {len(records)} movie(s) in {split_dir}")
    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating dataset in {OUT_DIR}")
    print(f"  train seeds: {TRAIN_SEEDS}")
    print(f"  val seeds:   {VAL_SEEDS}")
    print(f"  frames/movie: {FRAMES}   noise: {NOISE_PROFILE}\n")

    train = _generate_split("train", TRAIN_SEEDS, OUT_DIR / "train")
    val = _generate_split("val", VAL_SEEDS, OUT_DIR / "val")

    manifest = {
        "meta": {
            "frames": FRAMES,
            "noise_profile": NOISE_PROFILE,
            "num_dots": sim.NUM_DOTS,
            "width": sim.WIDTH,
            "height": sim.HEIGHT,
        },
        "train": train,
        "val": val,
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"Point DATASET_MANIFEST in the parametrizer at this file.")


if __name__ == "__main__":
    main()
