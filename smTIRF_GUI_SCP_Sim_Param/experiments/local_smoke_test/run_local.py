"""
Local end-to-end smoke run of the full pipeline, driven on THIS machine.

Because this machine has no `py -3.11` and no real test movie, it:
  - generates a REDUCED dataset (3 train / 2 val, 40 frames) into ./dataset,
  - points the parametrizer at the LOCAL SCP detector and THIS interpreter,
  - runs a small Optuna search (5 trials) + validation model-selection,
  - skips the TEST stage (no real movie present -> handled gracefully).

It proves the train -> validate -> test wiring against the REAL detector on
REAL simulated data. Scale N_TRIALS / dataset size on the target machine.
"""
import os, sys, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent               # smTIRF_GUI_SCP_Sim_Param/
sys.path.insert(0, str(PROJ / "Simulation"))
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import smTIRF_simulation as sim

# ---- 1. generate a small dataset -----------------------------------------
DATA = HERE / "dataset"
FRAMES = 40
TRAIN_SEEDS = [0, 1, 2]
VAL_SEEDS = [1000, 1001]

def make_split(split, seeds):
    sd = DATA / split; sd.mkdir(parents=True, exist_ok=True)
    recs = []
    for s in seeds:
        stem = f"rep{s}"
        recs.append(sim.generate(
            seed=s,
            out_tif=str(sd / f"{stem}.tif"),
            positive_json=str(sd / f"{stem}_positive.json"),
            negative_json=str(sd / f"{stem}_negative.json"),
            frames=FRAMES, verbose=True))
    return recs

print("=== generating reduced dataset ===")
manifest = {"meta": {"frames": FRAMES, "note": "reduced local smoke run"},
            "train": make_split("train", TRAIN_SEEDS),
            "val": make_split("val", VAL_SEEDS)}
mpath = DATA / "manifest.json"
mpath.write_text(json.dumps(manifest, indent=2))
print(f"manifest -> {mpath}")

# ---- 2. drive the parametrizer against the local detector ----------------
import SCP_parametrizer_002 as pm
pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]         # same interpreter where deps are installed
pm.DATASET_MANIFEST = str(mpath)
pm.N_TRIALS = 5
pm.TOP_K = 3
pm.RESULTS_FILE = str(HERE / "optimization_results.json")
# TEST_* keep their default C:\Users\PC paths (absent) -> test stage skips.

print("\n=== running train -> validate -> test pipeline ===")
payload = pm.run_pipeline()
print("\n=== DONE ===")
