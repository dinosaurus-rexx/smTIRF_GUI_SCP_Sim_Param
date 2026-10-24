# Methods — SCP detector parameter optimization and evaluation

> Draft for a thesis/paper Methods section. Past tense, third person. `[bracketed]`
> items are placeholders to fill in (software versions, exact counts). Trim to taste.

## Spot detection algorithm (SCP)

Fluorescent spots were detected with an in-house detector ("SCP") that combines
per-frame blob detection with cross-frame tracking. In each frame, candidate spots
were found using the Laplacian-of-Gaussian (LoG) detector (`skimage.feature.blob_log`,
[scikit-image version]) over a scale range `[min_sigma, max_sigma]` with an absolute
response `threshold`. Detected blobs were linked across frames by nearest-neighbour
association using a KD-tree, subject to a maximum inter-frame displacement
(`max_displacement`). Linked tracks were then classified as valid events using a
circularity criterion (`circularity_threshold`) and a temporal-variation criterion
requiring intensity fluctuation over at least `min_pulsation_frames` frames with a
coefficient of variation of at least `intensity_variation_threshold`. The detector
therefore exposes seven free parameters (Table 1), which jointly determine both
detection sensitivity and the classification of spots as genuine events.

## Image preprocessing (background subtraction)

Prior to blob detection, each frame was band-pass filtered to suppress the broad,
spatially uneven background characteristic of the real recordings. A background
estimate was formed by Gaussian smoothing of the frame (σ = [8] px) and subtracted
from the original frame, with negative values clipped to zero; the result was then
min–max normalized before LoG detection. Intensity values used downstream (e.g. for
per-spot intensity) were taken from the unfiltered frame. This step was necessary for
the real data: without it, the uneven background dominated the per-frame
normalization, so recovering the (dim) real spots required detection thresholds so
low that the number of spurious candidates made cross-frame tracking computationally
intractable. The same preprocessing was applied to all movies (simulated and real) to
keep the tuned parameters consistent with the pipeline used at evaluation time.

## Datasets and ground truth

**Simulated data.** Synthetic smTIRF movies were produced by a custom generator that
places stationary–pulsating spots (the detection target), together with moving and
static-bright distractor spots, on a Poisson + Gaussian (shot + read) noise
background at a configurable signal-to-noise regime. The generator seeds both the
Python and NumPy random number generators, so a given seed reproduces a movie and its
ground-truth spot list exactly; distinct seeds yield statistically independent movies,
which were used to build training sets. Ground-truth positive (target) and negative
(distractor) spot coordinates were exported per movie.

**Real data.** A single dual-channel smFRET movie ([N] = 501 frames, 2 × 256 × 512,
16-bit) was used as the real test case. The channel containing the annotated spots was
identified by comparing spot-region intensity against background (median SNR ≈ [11] in
that channel) and analyzed as a single-channel `(T, Y, X)` stack. Ground truth
consisted of [334] manually annotated spots ([243] stationary/target, treated as
positives; [91] aggregate/off-target, treated as negatives); annotation coordinates
were stored in `[y, x]` order and converted to the detector's `(x, y)` convention.

## Detection–annotation matching and performance metrics

Detections were matched to annotated positives by greedy one-to-one nearest-neighbour
assignment: all detection–annotation pairs within a fixed pixel margin (Chebyshev
distance ≤ [3] px) were formed and assigned in order of increasing Euclidean distance,
each detection and annotation used at most once. Matched pairs were counted as true
positives (TP); unmatched annotated positives as false negatives (FN); and every
detection not matched to a positive as a false positive (FP), so that over-detection
(including detections on annotated distractors) was penalized directly. Performance
was summarized by

- recall = TP / (number of annotated positives),
- precision = TP / (TP + FP),
- Fβ = (1+β²)·precision·recall / (β²·precision + recall), with β = 1 (F1).

F1 (a harmonic combination) was used as the scalar objective in preference to a linear
weighted sum of precision and recall, because a weighted sum is fully compensatory
(a high recall can mask poor precision) and its weights carry no interpretable
trade-off, whereas F1 penalizes imbalance and is standard for spot-detection tasks.
(This assumes the annotations are exhaustive within the scored region; the real movie
was fully annotated across its field of view.)

## Parameter optimization

The seven detector parameters were optimized by Bayesian optimization using the
Tree-structured Parzen Estimator (TPE) sampler (`optuna`, [version]), maximizing F1.
Unlike one-parameter-at-a-time (coordinate-descent) tuning, TPE samples the full
seven-dimensional space jointly and therefore accounts for interactions between
parameters (e.g. the blob scale range and the detection threshold are strongly
coupled). Each objective evaluation ran the detector on the movie(s) and computed F1
against the ground truth. The sampler was seeded for reproducibility, and each
detector run was given a fixed wall-clock timeout so a single pathological parameter
set could not stall the search.

Search bounds are given in Table 1. For the real-data optimization the detection
threshold was floored at [0.25] to keep every trial computationally tractable (lower
thresholds produced enough candidates to make tracking exceed the timeout), and the
three classification-filter parameters were constrained to a range that preserves
recall, since exploratory runs showed that aggressive filtering — appropriate for the
smoothly pulsating simulated spots — rejected the irregularly blinking real spots. The
search was additionally warm-started by enqueuing a small number of
manually-identified, recall-preserving configurations, which seeds TPE in the
productive region of an otherwise large and mostly-unproductive space.

**Table 1.** Optimized parameters, search bounds, and selected values. *(fill values)*

| Parameter | Role | Search range | Selected |
|---|---|---|---|
| min_sigma | min blob scale (px) | [0, 1.5] | 0.45 |
| max_sigma | max blob scale (px) | [0.5, 3.5] | 1.27 |
| intensity_threshold | LoG response threshold | [0.25, 0.5] | 0.32 |
| circularity_threshold | min circularity | [0.1, 0.6] | 0.20 |
| max_displacement | max inter-frame move (px) | [0.5, 5] | 5.0 |
| min_pulsation_frames | min frames of variation | [3, 15] | 11 |
| intensity_variation_threshold | min coeff. of variation | [0.05, 0.25] | 0.07 |

## Evaluation design and model selection

Because only a single annotated real movie was available, parameters were tuned and
evaluated using a **spatial cross-validation** of its field of view: the image was
partitioned into three disjoint vertical bands along x — train `[0, 170)`,
validation `[170, 340)`, and test `[340, 512)` px — which partition the annotated
spots into disjoint sets. Since the detector processes the entire movie in one pass,
each trial produced detections across the whole field, and the three band scores were
obtained by restricting the matching to spots and detections within each band. This
made every trial cost a single detection while still yielding an independent
train/validation/test split. Parameters were optimized on the training band; the
final configuration was chosen from the top-[K] training trials by its
validation-band F1; and the held-out test-band metrics were reported as the unbiased
estimate of performance. The test band was never used during optimization or
selection.

For comparison, a **simulation-guided** variant was also evaluated, in which the search
was performed on simulated movies (objective: mean per-movie F1 over the training
movies) and the top-[K] configurations were then scored on the real movie, selecting
on the real validation band and reporting the real test band. This isolates whether
parameters tuned on simulation transfer to real data.

## Reproducibility and implementation

All stochastic components were seeded (simulation RNGs, TPE sampler). Detector outputs
were written to explicit per-run paths and scored directly, avoiding any dependence on
file-discovery heuristics. Software: Python [3.x], scikit-image [ver], optuna [ver],
NumPy [ver], SciPy [ver], tifffile [ver]. [Hardware, run time.]

## Summary of the main methodological finding

Parameters optimized directly on the real movie (spatial cross-validation) achieved a
held-out test-band F1 of [0.80] (recall [95]%, precision [69]%), with closely matching
train/validation/test scores indicating no overfitting to a region. In contrast,
parameters optimized on the simulation and transferred to the real movie achieved F1
[0.26] (recall [18]%, precision [44]%). The gap is attributable to the temporal
event-classification filters, which the simulation-based search tuned to the smoothly
pulsating synthetic spots and which then rejected the irregularly blinking real spots.
We therefore report the directly-tuned configuration as the recommended parameter set
and note that, in its current form, the simulator is not an adequate surrogate for
parameter selection on this real data.
