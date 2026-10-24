"""
Tune SCP on the BACHELORTHESIS simulator, then calibrate the one non-transferable
parameter on the real movie.

Counterpart to ../sim_guided_real_selection/run_real.py, which searched on smTIRF_simulation.py
and transferred badly (real test F1 0.26, recall 18%). Two things differ here.

1. The simulator. The BT target switches between two FRET states via an HMM, giving
   step-like intensity changes on a single channel: median per-spot intensity CV 0.193,
   against the real movie's irregular blinking at 0.261, where smTIRF_simulation.py's
   smooth pulsation gives 0.789. SCP's event classifier thresholds exactly that
   quantity, so this is the mismatch that sank the earlier run. The BT simulator also
   supplies aggregate and mobile distractor classes, keeping precision under real
   pressure during the search. (Note: one channel, NOT the sum -- summing a FRET pair
   cancels the variation, giving CV 0.014.)

2. intensity_threshold is no longer transferred. SCP min-max normalises each frame, so
   that parameter is a fraction of the frame's brightest object rather than an absolute
   brightness, and the two datasets' usable ranges do not overlap (sim <=0.10, real
   >=0.20). Brightness rescaling cannot fix it -- min-max normalisation is
   scale-invariant, and a 3x sweep left the measured LoG response flat at 0.55-0.60x of
   the real movie's. So the six shape/behaviour parameters are searched on the sim and
   this one is calibrated on the real validation band. The real TEST band is held out.

Because of (2) this is NOT a like-for-like rerun of run_real.py: it answers "does BT's
blinking teach better filters?", not "does a pure sim-guided transfer work?".

Run:
    cd smTIRF_GUI_SCP_Sim_Param/experiments/bt_simulator_threshold_calibration
    python run_bt_sim.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Simulation"))
sys.path.insert(0, str(PROJ / "Parameter_opt"))

# ---- 1. build the BT-simulator dataset (idempotent; regenerates from seeds) ----
import generate_dataset_bt as gen

gen.OUT_DIR = HERE / "dataset"
print("=== generating BachelorThesis-simulator search set ===")
manifest_path = gen.main()

# ---- 2. run the sim-guided -> real-select pipeline ----------------------------
import SCP_parametrizer_002 as pm

pm.scp_script = str(PROJ / "SCP" / "SCP_KTree_JSON.py")
pm.PYTHON_CMD = [sys.executable]
pm.DATASET_MANIFEST = str(manifest_path)
pm.METHOD = "sim_shape_real_threshold"

# The six shape parameters transfer; intensity_threshold does not, and is recalibrated
# on the real validation band. See run_sim_shape_real_threshold() for the measurements
# behind that split. The sim's own usable threshold band is well below the module
# default floor of 0.25 -- at 0.25 the simulation yields ~2 blobs/frame and every trial
# scores a flat zero -- so widen it here for the search.
pm.PARAM_BOUNDS['intensity_threshold'] = (0.05, 0.15)

# ~41s per sim movie at these settings, so 3 movies x 30 trials ~ 60-90 min, plus
# TOP_K x len(REAL_THRESHOLD_GRID) real-movie runs at ~200s each.
pm.N_TRIALS = 30
pm.TOP_K = 4
pm.DETECTION_TIMEOUT = 300      # cut pathological low-threshold trials
pm.RESULTS_FILE = str(HERE / "optimization_results.json")
# USE_BANDPASS=True stays at its module default, as in run_real.py. TEST_* already
# point at the real vid_ch1.tiff + coord/class npy; bands default to
# REAL_VAL_XBAND=(0,256), REAL_TEST_XBAND=(256,512).

print("\n=== running sim-shape -> real-threshold pipeline (BT simulator) ===")
pm.run_sim_shape_real_threshold()
print("\n=== DONE ===")
