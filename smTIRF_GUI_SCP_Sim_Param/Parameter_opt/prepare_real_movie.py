"""
Prepare the real (non-simulated) test movie for the SCP detector.

The raw real movie is a 2-channel smFRET stack, shape (T, C, Y, X) = (501, 2, 256, 512).
The detector expects a single-channel (T, Y, X) movie, and the annotations
(realdata/coord.npy, class.npy) were drawn from CHANNEL 1 -- verified: annotated
spots sit ~2x above background in channel 1 vs ~1.2x in channel 0.

This extracts that channel to a (T, Y, X) TIFF the detector can read, saved next to
the labels as vid_ch1.tiff. Point TEST_TIF in the parametrizer at the output.

Run:
    python prepare_real_movie.py
"""
import tifffile

SRC = r"C:\Users\NahuelJimenezPol\Downloads\vid.tiff"
DST = r"C:\Users\NahuelJimenezPol\Desktop\Const\BachelorThesis-FRETAlgorithms\realdata\vid_ch1.tiff"
CHANNEL = 1   # annotations correspond to channel 1 (the bright/detection channel)


def main():
    a = tifffile.imread(SRC)          # (T, C, Y, X)
    print(f"source shape: {a.shape} {a.dtype}")
    if a.ndim != 4:
        raise ValueError(f"expected 4D (T,C,Y,X), got shape {a.shape}")
    single = a[:, CHANNEL, :, :]      # (T, Y, X)
    print(f"extracted channel {CHANNEL}: {single.shape} {single.dtype}")
    tifffile.imwrite(DST, single, imagej=True, metadata={"axes": "TYX"})
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
