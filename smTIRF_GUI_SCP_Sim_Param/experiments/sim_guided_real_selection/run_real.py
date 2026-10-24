"""
Sim-guided -> select-on-real run (goal: best params FOR THE REAL DATA).

  1. Search on a few seeded sim movies (30 Optuna trials) for breadth.
  2. Take the top-K configs and run each ONCE on the real movie, sliced by X into
     a validation band (to select) and a held-out test band (to report).

Only K slow real-movie detections; the sim search is cheap. Scale N_TRIALS /
sim movie count / TOP_K on the target machine.
"""
import os, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Simulation"))
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import smTIRF_simulation as sim

# ---- small sim set just for the search ------------------------------------
DATA = HERE / "dataset"
FRAMES = 50
TRAIN_SEEDS = [0, 1, 2]

# "mid" SNR: the real spots are dim (~2x background), so a dimmer sim regime makes
# the search favor thresholds that actually transfer to the real movie.
NOISE_PROFILE = "mid"

def make_train():
    sd = DATA / "train"; sd.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in TRAIN_SEEDS:
        stem = f"rep{s}"
        recs.append(sim.generate(
            seed=s, out_tif=str(sd / f"{stem}.tif"),
            positive_json=str(sd / f"{stem}_positive.json"),
            negative_json=str(sd / f"{stem}_negative.json"),
            frames=FRAMES, noise_profile=NOISE_PROFILE, verbose=False))
    return recs

print("=== generating sim search set (3 movies, 40 frames) ===")
manifest = {"meta": {"frames": FRAMES}, "train": make_train(), "val": []}
mpath = DATA / "manifest.json"
mpath.write_text(json.dumps(manifest, indent=2))

# ---- configure and run the sim-guided -> real-select pipeline --------------
import SCP_parametrizer_002 as pm
pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]
pm.DATASET_MANIFEST = str(mpath)
pm.METHOD = "sim_guided_real"
pm.N_TRIALS = 60     # bumped search
pm.TOP_K = 8         # judge more candidates on the real movie
# bandpass is on by default (pm.USE_BANDPASS=True); threshold floored at 0.25 so
# every candidate is tractable on the real movie.
pm.RESULTS_FILE = str(HERE / "optimization_results.json")
# TEST_* already point at the real vid_ch1.tiff + coord/class npy; bands are the
# defaults REAL_VAL_XBAND=(0,256), REAL_TEST_XBAND=(256,512).

print("\n=== running sim-guided -> real-select pipeline ===")
pm.run_sim_guided()
print("\n=== DONE ===")
