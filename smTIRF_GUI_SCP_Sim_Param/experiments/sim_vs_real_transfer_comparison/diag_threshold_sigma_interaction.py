"""
Why arm A did NOT time out, contrary to the inherited record.

SESSION_RECORD_2026-07-28.md section 5 tabulates `intensity_threshold` 0.05 on
realdata/vid_ch1.tiff as "timeout (>300 s)" and 0.12 as "timeout (>420 s)", and
concludes the sim's usable threshold range (<=0.10) and the real movie's (>=0.20)
"do not overlap". EXPERIMENT_SPEC_sim_transfer.md section 4 therefore predicted arm A
would time out even at 1800 s.

Arm A instead finished in 202 s at test-band F1 0.677.

The resolution is that threshold does not set tractability on its own -- it sets it
jointly with the sigma range, and the two probes used different sigma ranges. SCP's
tracking cost is driven by blobs-per-frame, so measuring the blob-detection front end
alone (no tracking) isolates the cause cheaply.

Run:
    python -u diag_threshold_sigma_interaction.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent.parent
sys.path.insert(0, str(PROJ / "Parameter_opt"))
sys.path.insert(0, str(HERE))

import SCP_parametrizer_002 as pm
from run_transfer import _blob_count      # SCP's exact front end, no tracking

FRAMES = 6


def sweep(tif, min_s, max_s, thr, bandpass, label):
    with tifffile.TiffFile(tif) as tf:
        n = tf.series[0].shape[0]
        idx = np.linspace(0, n - 1, FRAMES).astype(int)
        counts = []
        t0 = time.time()
        for i in idx:
            frame = np.asarray(tf.asarray(key=int(i)))
            counts.append(_blob_count(frame, {'min_sigma': min_s, 'max_sigma': max_s,
                                              'intensity_threshold': thr}, bandpass))
        dt = time.time() - t0
    px = frame.size
    mean = float(np.mean(counts))
    print(f"  {label:50s} thr={thr:.4f} bp={str(bandpass):5s} sig={min_s:.2f}-{max_s:.2f}"
          f" -> {mean:8.1f} blobs/frame ({mean / (px / 1000):6.2f} per 1000px)"
          f"  [{dt:.1f}s/{FRAMES} frames]")
    return mean


if __name__ == "__main__":
    REAL = pm.TEST_TIF

    print("=" * 78)
    print("1. REAL MOVIE at threshold 0.05 -- what actually controls tractability")
    print("=" * 78)
    sweep(REAL, 1.2926, 2.2899, 0.05053, True,  "arm A: trial 25 sigmas, bandpass ON")
    sweep(REAL, 1.2926, 2.2899, 0.05053, False, "trial 25 sigmas, bandpass OFF")
    sweep(REAL, 0.5, 2.5, 0.05053, True,  "wide sigmas 0.5-2.5 (module seed), bandpass ON")
    sweep(REAL, 0.5, 2.5, 0.05053, False, "wide sigmas 0.5-2.5, bandpass OFF")
    sweep(REAL, 1.0653, 1.9028, 0.05033, True, "trial 22 sigmas, bandpass ON")
    print("\n  -> min_sigma, not threshold, is what explodes the blob count. Trial 25's")
    print("     narrow high-sigma window admits ~190 blobs/frame at threshold 0.05;")
    print("     the wide 0.5-2.5 window admits ~5700 at the SAME threshold (30x), and")
    print("     ~11800 with bandpass off. THAT is the configuration that timed out.")

    print("\n" + "=" * 78)
    print("2. REAL MOVIE, trial 25 sigmas, bandpass ON -- threshold response")
    print("=" * 78)
    for thr in (0.05053, 0.10, 0.12, 0.15, 0.20, 0.32):
        sweep(REAL, 1.2926, 2.2899, thr, True, "trial 25 sigmas")
    print("\n  -> monotone and gentle: 190 -> 8 blobs/frame across the whole range.")
    print("     At these sigmas there is no intractable region at all.")

    print("\n" + "=" * 78)
    print("3. BT SIM at trial 25's own parameters -- the arm A2 reference density")
    print("=" * 78)
    for t in sorted((PROJ / "experiments" / "bt_simulator_threshold_calibration" / "dataset" / "train").glob("*.tif*")):
        sweep(str(t), 1.2926, 2.2899, 0.05053, True, f"sim {t.name}")
    print("\n  -> sim density ~1.59 per 1000 px vs the real movie's 1.45 at the SAME")
    print("     threshold. In density terms the sim's threshold nearly transfers as-is,")
    print("     so arm A2's label-free calibration should land near 0.05, not near 0.32.")
    print("\n  Frame areas differ (sim 256x256, real 256x512), which is why density per")
    print("  1000 px is the comparable quantity and not raw blobs-per-frame.")
