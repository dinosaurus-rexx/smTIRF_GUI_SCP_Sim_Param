"""
Direct-on-real tuning: search parameters directly on the real movie with a
spatial train/val/test split of the field of view. Bandpass on, threshold floored
so every trial is tractable (~130-250s each). ~40 trials -> ~2 hours.

This targets the real data directly, so it should fix the sim-guided failure mode
(sim-tuned pulsation/circularity filters rejecting real spots).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))

import SCP_parametrizer_002 as pm
pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]
pm.METHOD = "direct_real"
pm.N_TRIALS = 40
pm.TOP_K = 8
pm.RESULTS_FILE = str(HERE / "optimization_results.json")
# TEST_* already point at the real vid_ch1.tiff + coord/class npy.
# USE_BANDPASS=True and intensity_threshold floored at 0.25 (both in the module).

print("=== direct-on-real tuning ===")
pm.run_direct_real()
print("=== DONE ===")
