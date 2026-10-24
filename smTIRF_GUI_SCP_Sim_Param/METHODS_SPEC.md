# Methods specification — simulator, SCP detector, and parameter optimization

**Purpose.** This is the complete methods brief for whoever writes the paper or thesis
chapter. It describes what was built, why each choice was made, how every reported
number was produced, and which numbers must not be quoted. It is written to be read
once, top to bottom, by someone who did not build the code.

**Scope.** Three components, all in
`smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/`:

| § | component | directory | one-line role |
|---|---|---|---|
| 2 | Simulator + validation harness | `Simulation/` | generates synthetic movies calibrated against a real recording, and judges how well they match it |
| 3 | SCP detector | `SCP/` | finds and classifies single-molecule fluorescence events in a movie |
| 4 | Parametrizer | `Parameter_opt/`, `experiments/` | tunes the detector's parameters and evaluates them honestly |
| 1.2 | Annotation tool | `GUI2/` | produces the manual event labels that everything else is scored against |

**Relationship to existing documents.**
- `METHODS_parameter_optimization.md` is an earlier partial draft covering only §3 and
  §4. Two of its statements are now superseded and are corrected here: its concluding
  claim that the simulator "is not an adequate surrogate for parameter selection"
  (§4.5), and its "median SNR ≈ 11" (§2.8.3).
- `SIMULATOR_SPEC.md` is the build specification handed to the implementer. Three of
  its claims were contradicted by measurement and were deliberately not built to
  (§2.8).
- `Simulation/CALIBRATION.md` is the measured-versus-target record. It is the primary
  source for every simulator number in §2.
- Where those documents disagree with this one, this one is the reconciled version.

**Conventions used throughout.** Distances are in pixels; times in frames.
Coordinates are written `(x, y)` with `x` the column index, except where the
BachelorThesis `.npy` format is explicitly named, which is `[y, x]`. "CV" means
coefficient of variation (standard deviation divided by mean). "AUC" means the area
under the receiver-operating characteristic curve, computed from a Mann–Whitney
statistic and reported as `max(auc, 1 − auc)` so it never falls below 0.5.

---

## 1. Data

### 1.1 The real recording

A single annotated movie is the entire real dataset. Everything about the evaluation
design in §4.3 follows from the fact that there is exactly one.

| property | value |
|---|---|
| source file | `BachelorThesis-FRETAlgorithms/realdata/vid_ch1.tiff` |
| frames | 501 |
| frame size | 256 (height) × 512 (width) |
| bit depth | 16-bit, single channel |
| origin | one channel of a dual-channel single-molecule FRET recording, analysed as a `(frames, height, width)` stack |
| annotations | 334 spots total |
| — class 0, "stationary" / target | 243, treated as positives (the detection target) |
| — class 1, "aggregate" / off-target | 91, treated as negatives (distractors) |
| — class 2, "moving" | 0 |
| annotation density | 2.55 spots per 1000 pixels; target-to-off-target ratio 2.67 : 1 |
| label format | `coord.npy`, shape `(N, 2)`, in `[y, x]` order; `class.npy`, shape `(N,)` |

**Placeholders the author must fill in.** The acquisition metadata is not recorded
anywhere in the codebase and must come from the experimenters: microscope and
objective, excitation wavelength and power, exposure time and frame interval, pixel
size in nanometres, camera model and its nominal gain, sample preparation, and who
annotated the movie under what written criterion.

**Two properties of these annotations that matter downstream, and should be stated
in the paper rather than buried:**

1. **The annotations are not exhaustive.** The movie contains more emitters than were
   marked. Measured in regions at least 6 px away from every annotation, the
   background-suppressed frame reaches a 99.9th percentile of 85.6 ADU, and the
   reference detector configuration finds 26.5 blobs per frame that no annotation
   explains. These are dim, non-specific emitters of the kind any real preparation
   contains. The evaluation treats a detection on one of them as a false positive
   (§4.1), which is the conservative choice, and the simulator reproduces the same
   situation deliberately (§2.4).
2. **The meaning of class 1 is undocumented.** "Aggregate" is whatever the annotator
   marked; no written criterion survives. Measured, class 1 spots are *fainter* than
   class 0 (median background-subtracted brightness 18.9 versus 60.6), which is the
   opposite of what a physical aggregate of several fluorophores would give. The class
   is therefore treated throughout as an empirical population to be reproduced and
   scored against, with no mechanistic claim attached.

### 1.2 Manual annotation (`GUI2/`)

Event labels are produced with a purpose-built browser-based annotation tool: a Flask
web application that loads a multi-frame TIFF, lets the operator step through frames
and mark events, and exports two JSON files — positive (target) and negative
(off-target) event lists. Those files are the direct input to the scorer in §4.1, and
their format is one of the two ground-truth contracts the simulator must satisfy
(§2.5).

The real movie's labels reached this project already in the BachelorThesis `.npy`
form described in §1.1; the tool is the route by which a *new* recording is brought
in. State it in the methods as the annotation instrument, and note the obvious
limitation: labels are one person's judgement, frame by frame, with no
inter-annotator agreement measured.

### 1.3 Simulated movies

Produced by `Simulation/generate_dataset.py` (see §2). Each dataset is a set of
movies matching the real recording's geometry (501 × 256 × 512, 16-bit), split by
random seed into disjoint training and validation groups — default seeds `0, 1, 2`
for training and `1000, 1001` for validation. Every movie carries ground truth in
both formats required by the consumers (§2.5), and a manifest records the fully
resolved configuration used to build it.

---

## 2. The simulator

### 2.1 Why simulate at all

Two reasons, and the paper should give both, because they are independent.

**Reason 1 — the previous simulator's parameters did not transfer.** Detector
parameters tuned on synthetic movies and then applied to the real recording performed
badly, measured on a held-out region of the real movie:

| how the detector was tuned | real held-out F1 | recall | precision |
|---|---|---|---|
| on the earlier simulator, then applied to the real movie | 0.256 | 18.0% | 44.4% |
| directly on the real movie's own labels | 0.800 | 94.9% | 69.2% |

A simulator that cannot be used to choose parameters has little practical value, so
the first question was whether that gap was intrinsic or merely a calibration
failure.

**Reason 2 — there is only one annotated real recording.** With a single movie, every
performance estimate rests on partitioning one field of view (§4.3), and no estimate
of between-recording variability is possible at all. A simulator that is faithful
enough to be believed is the only available route to a second, independent evaluation
set. This is the deeper motivation and it survives regardless of how §2.7's gates
turn out.

A third, practical reason is worth one sentence: manual annotation is the expensive
resource. If parameters can be searched on simulation and only *selected* against a
small labelled region, the annotation burden on a new recording drops substantially
(§4.4 quantifies exactly how much of this claim is supported).

### 2.2 Design principle: measure first, then build, then judge with separate code

The simulator was built under three rules that are worth stating explicitly in the
methods, because they are what make its validation meaningful.

1. **Every calibration target is derived from the real movie by code, not quoted from
   prose.** `Simulation/measure_real_movie.py` reads the real recording and its
   labels and writes a machine-readable reference file
   (`config/real_movie_reference.json`) containing geometry, spot counts and density,
   nearest-neighbour spacing distribution, per-spot photometry and class
   separability, detector-mediated track durations, background structure, and
   blob-density-versus-threshold curves. If a new recording is annotated, it can be
   dropped in and every target and every acceptance threshold follows automatically.
2. **Both sides of every comparison go through one shared estimator module.**
   `Simulation/measurements.py` implements each measurement exactly once and is
   applied identically to the real movie and to simulated movies, so a difference
   between them can never be an artefact of two different implementations of the same
   quantity, including arbitrary choices such as aperture radii, frame sampling, or
   preprocessing order. Detection scoring is not re-implemented at all: the harness
   calls the same scorer the tuning experiments use (§4.1), and invokes the detector
   through the same subprocess wrapper.
3. **Fitting and judging are separate programs.** `Simulation/calibrate.py`
   *searches* for photometry parameters; `Simulation/validate_against_real.py`
   *judges* against the acceptance gates. Keeping them apart is what prevents the
   acceptance criteria from silently becoming the fitting objective.

A fourth rule was followed as a matter of order of work and should be mentioned: the
validation harness was first run against the *older* simulator, before it was used to
steer anything, and it reproduced that simulator's known mismatches (§2.8.4). The
instrument was validated before it was trusted.

### 2.3 The image formation model

One movie is generated per call from a YAML configuration file and an integer seed.
The model is described here in the order the code applies it.

**(a) Spot placement — soft-core rejection sampling.**
Positions are continuous (sub-pixel) and drawn uniformly over the frame, subject to a
separation rule against all previously accepted spots: a candidate closer than
`min_distance` = 3.5 px is always rejected; between 3.5 px and `soft_distance` = 6.0 px
it is accepted with probability `((d − 3.5) / 2.5) ** 3`; beyond 6 px it is always
accepted. All classes are placed by one interleaved, shuffled process with one shared
floor, so no class is systematically pushed into the gaps left by another.

Both parts of this rule are measurement-driven. The floor is 3.5 px rather than the
7 px used by the earlier BachelorThesis simulator because the real movie's closest
annotated pair is 3.61 px apart and 12 of 243 targets have a neighbour closer than
6 px — a 7 px floor forbids spacings that genuinely occur. A hard floor alone is not
sufficient either: a purely random (Poisson) process with a 3.5 px hard core gives a
nearest-neighbour median of 10.90 px with 12.3% of spots under 6 px, against the real
12.04 px with 4.9%, and a two-sample Kolmogorov–Smirnov test rejects it at
p < 10⁻⁴. The real annotation pattern is *more regular* than random — close pairs are
suppressed but not forbidden — which the soft-core ramp reproduces (median 12.04 px,
3.8% under 6 px, KS p = 0.91). A plausible reading, offered as interpretation rather
than fact: two molecules within about a spot width of each other are hard for a human
to mark as two.

**(b) Point spread function — one per emitter, never a blurred frame.**
Each emitter contributes its own two-dimensional Gaussian, evaluated at true pixel
centres offset from its continuous position, and these are *summed*. The frame is
never blurred after summing. This matters for overlapping spots: adding two Gaussians
is the physically correct treatment of two nearby molecules, whereas blurring a summed
frame fuses them before the optics are applied.

The width is `psf_sigma` = 1.55 px, and this contradicts the build specification,
which predicted a value "well below" the earlier simulator's 1.5 (§2.8.1). Measured
four ways, the real spots give σ = 1.478 (stacked second moment), 1.473 (per-spot
median), 1.718 (stacked Gaussian fit) and 1.550 (from the radial-profile full width
at half maximum, which involves no fitting). Feeding 1.55 into the simulator
reproduces those, because the estimators are slightly biased low. The off-target class
is given its own measured width, 1.63 px, because it is genuinely wider than the
target class (1.559 versus 1.478 by stacked second moment) and that width feeds
directly into how often the detector fires on it.

Amplitude, not integrated volume, is the configuration unit, because amplitude is
what the photometry in (d) measures and what the detector thresholds against.

**(c) Intensity comes from photophysics; spot size never changes.**
Every emitter has a fixed width and a time-varying amplitude, composed of three
mechanisms:

- *Per-molecule brightness*: a lognormal peak amplitude, parameterised by its median
  so the configuration knob is the quantity the photometry table reports. This sets
  the *width* of the brightness distribution, not only its centre — the real targets
  have an interquartile ratio of 2.25, which no single fixed amplitude can produce.
- *Blinking*: a two-state Markov chain with geometric dwell times of mean `tau_on`
  and `tau_off` frames, starting from the stationary distribution so there is no
  transient at frame 0. Amplitude is gated on or off; size is untouched.
- *Photobleaching*: a gamma-distributed time to a permanent dark state, with an
  optional delayed activation. Implemented and calibrated, but **inert in the final
  configuration** (`tau_bleach = 0`): under the slow switching the data turned out to
  require, blinking alone accounts for both the duty cycle and the observed track
  durations, and adding bleaching on top would push the duty cycle below the value
  the measured variability implies. It is retained because a longer recording or
  brighter illumination would need it.

An additional frame-to-frame lognormal jitter of the on-state amplitude
(`on_amplitude_cv` = 0.25) covers residual variability that on/off switching alone
does not explain. This knob turned out to be unexpectedly consequential, in a
counter-intuitive direction, for reasons that belong to the detector rather than the
simulator; see §2.7.3.

**The switching is slow, not fast — this is a substantive finding, not a parameter
choice.** An initial implementation used rapid blinking (mean on-time about 8 frames),
following the intuition that off-times should occasionally exceed the tracker's
3-frame tolerance. That cannot reproduce the real track-duration distribution at all.
Because the detector bridges gaps of at most 3 frames and then requires a minimum
number of detections, a track is effectively one contiguous on-period seen through an
imperfect detector. With fast blinking the off-target class produced *one* surviving
event per movie against the real 46, across six tested configurations. Moving to
on/off periods of order 100 frames took the target class's median track duration from
264 frames to 132 (real: 135) and brought the off-target class to 33–38 (real: 49).

**(d) Structured background.**
The real recording's background is broad, uneven and slowly drifting; a flat offset
cannot represent it, and — because of how the detector normalises each frame (§3.3) —
that mismatch is a direct cause of detection thresholds not transferring between
simulated and real data. Measured on the σ = 8 px low-pass of the real frames (the
very field the detector's preprocessing subtracts):

| quantity | measured |
|---|---|
| mean level | 147.19 ADU |
| spatial standard deviation per frame | 9.58 ADU (8.57 static, 5.46 drifting) |
| minimum / maximum | 111.60 / 222.98 ADU |
| relative swing, (max − min) / mean | 0.529 |
| spatial autocorrelation length (1/e) | 29.78 px |
| per-frame mean, range over the movie | 142.85 – 157.28 ADU |
| pattern correlation with frame 0 | 0.664 – 0.836, i.e. it drifts |

The model is a Gaussian random field — white noise smoothed by a Gaussian kernel and
rescaled to a target standard deviation — in three parts: a static field redrawn per
movie, a drifting field interpolated between four independent draws so the pattern
decorrelates across the movie as the real one does, and a global per-frame offset
reproducing the wander of the frame mean. Because smoothed white noise has a known
autocorrelation, the generation scale needed to produce a *measured* 29.78 px
autocorrelation length through the σ = 8 analysis filter can be solved for directly:
12.6 px. A handful of broad blobs plus a gradient would also work; the random field
was chosen so the autocorrelation length is one explicit knob rather than an emergent
property of hand-placed features.

**(e) Noise: background, then shot noise, then read noise — in that order.**
The order is not cosmetic. Shot noise is a property of photon arrivals, so it must be
applied to the combined background-plus-signal rate; adding read noise first and then
drawing shot noise from the result would model a sensor that amplifies its own read
noise.

Pixel values are in analogue-to-digital units (ADU), not photons. Differencing
consecutive real frames — which cancels static structure and isolates the
high-frequency part — gives 27.41 ADU of noise at a mean of 147.3, so
variance/mean ≈ 5.1. The conversion gain is therefore set to 5.0 ADU per photon, the
Poisson draw is performed in photon units, and the result is scaled back. Skipping
that conversion would understate the noise by a factor of about 2.2 and hand the
detector a far cleaner movie than the real one. Gaussian read noise of 3.8 ADU is
added last — the residual once the gain-times-mean term is removed from the measured
751 ADU² of variance — and the frame is rounded and clipped to 16-bit.

### 2.4 Emitter classes

Four classes, three of them active by default.

| class | count | in ground truth? | model |
|---|---|---|---|
| target (class 0) | 243 | yes, as positives | peak amplitude median 153 ADU, lognormal spread 0.36; `tau_on`/`tau_off` = 170/115 frames (duty 0.60); σ = 1.55 px |
| off-target (class 1) | 91 | yes, as negatives | peak amplitude median 110 ADU, essentially no per-molecule spread (0.047); `tau_on`/`tau_off` = 120/265 frames (duty 0.31); σ = 1.63 px |
| unannotated | 500 | **no — in no ground-truth output at all** | peak amplitude median 70 ADU, spread 0.25; `tau_on`/`tau_off` = 90/200 frames |
| moving (class 2) | 0 | implemented, `activate: false` | Gaussian random walk, 1 px per frame |

**The off-target class is defined empirically, and deliberately not mechanistically.**
The earlier simulator modelled an aggregate as the sum of several always-on
fluorophores, which predicts aggregates *brighter* than targets. The real ones are
dimmer. What reproduces the measurements is a fainter, more intermittent, slightly
wider emitter — one mechanism that simultaneously accounts for the faint median
brightness (18.9 versus 60.6), the *higher* relative variability (CV 1.596 versus
0.791) and the shorter detector track durations (49 versus 135 frames). No claim is
made that this is the physical mechanism.

**The unannotated class deserves a paragraph in the paper.** These 500 emitters appear
in no label file, no `coord.npy` and no `class.npy`, so a detection on one is scored
as a false positive exactly as it would be on the real movie. They are not a device
for softening precision — they are the measured feature of the real recording
described in §1.1, point 1. Omitting them held the simulated blob density at 0.53× the
real value at every threshold, which was the single largest remaining mismatch;
including them brought all four gated thresholds inside tolerance and also fixed the
false-positive population, taking the reference detector's performance on simulated
movies to recall 88.3% / precision 79.6% / F1 0.837 against the real 94.6% / 76.6% /
0.847. Their count (500) is fitted, and the response is non-monotonic: because of the
per-frame normalisation in the detector (§3.3), these emitters also raise the
normalisation divisor, so more is not automatically better.

The moving class is retained in code but switched off, because the real recording
contains zero moving annotations. Emitting them would be a mismatch, not extra
realism.

### 2.5 Reproducibility and output contracts

**Seeding.** One integer seed reproduces a movie *and* its ground truth exactly,
byte for byte; distinct seeds give movies that differ in both geometry and noise.
Every random draw goes through a generator derived from the seed by splitting one seed
sequence into independent streams — placement, photophysics, background, noise — and
generators are passed explicitly. No global random state is touched anywhere, so a
caller's own random number use cannot perturb a movie, and adding a class or changing
the noise model does not reshuffle the others.

**Configuration.** All parameters live in a YAML file with a nested, per-class
`activate` / `count` / `params` structure; no module-level constant is part of the
interface. The fully resolved configuration is written into the dataset manifest, so a
dataset carries everything needed to regenerate itself.

**Three output formats, written for every movie.**

| format | shape / content | consumer |
|---|---|---|
| TIFF | `(frames, height, width)`, 16-bit, single channel, axes tagged `TYX` | the detector |
| label JSON, two files | `[{"center": {"x": …, "y": …}}, …]`, sub-pixel, `(x, y)` | the parametrizer's scorer |
| `.npy` pair | `coord.npy` shape `(N, 2)` in **`[y, x]`**, `class.npy` with 0/1/2 | the BachelorThesis benchmark pipeline |

The axis-order flip between the JSON and the `.npy` files is the most bug-prone thing
in this codebase; it is performed in exactly one function and pinned by a contract
test using a deliberately asymmetric frame. Sixteen contract tests cover axis order,
seeding, and the fact that both downstream consumers load the dataset without
modification.

### 2.6 What was calibrated against what

Every simulator knob, its final value, and the measurement it was fitted to. This
table is the honest answer to "how many free parameters does the simulator have":
roughly a dozen, all fitted to one recording (§2.9).

| knob | value | fitted against | gate it moves |
|---|---|---|---|
| `psf_sigma` | 1.55 px | measured spot width, four estimators | T1 |
| `off_target.psf_sigma` | 1.63 px | measured off-target width (1.559 vs 1.478) | T4 |
| `gain_adu_per_photon` | 5.0 | frame-difference noise, 27.41 ADU at mean 147.3 | noise realism |
| `read_noise_adu` | 3.8 | residual after the gain term | noise realism |
| background parameters | see §2.3(d) | σ = 8 field statistics | T2 |
| `min_distance` / `soft_distance` / `soft_gamma` | 3.5 / 6.0 / 3.0 | nearest-neighbour distribution | T5 |
| `peak_amplitude_median` | 153 / 110 ADU | measured brightness; linear, exact inversion | T3 |
| `peak_amplitude_sigma_log` | 0.36 / 0.047 | measured interquartile ratio | T3 (AUC) |
| `tau_on` / `tau_off` | 170/115, 120/265 | measured variability (duty) and track duration | T3, T4 |
| `bleach_shape` | 3.0 (inert) | interquartile ratio, when bleaching was active | T3 |
| `on_amplitude_cv` | 0.25 | residual variability, and blob density | T2 ↔ T4 |
| `unannotated.count` | 500 | blob density and precision | T2 |

Two methodological findings from the fitting are worth a footnote in the paper,
because both are easy to get wrong and each cost a calibration round:

1. **The per-spot brightness trace is reduced by its mean, not its median.** The
   reference photometry aggregates each spot's own time trace and then takes the
   median *across* spots. Using the median of the trace gives 51.1 / 7.2 and an AUC of
   0.786; using the mean gives 60.6 / 18.9 and AUC 0.909, which matches the recorded
   values to three digits. Calibrating against the wrong one misses the target
   variability by 27%.
2. **The *median* emitter's duty cycle is what the measurement reports, not the
   average one.** When bleaching is active, a few early-bleached emitters drag the mean
   duty far below the median (0.570 versus 0.679 at one point). A first calibration
   pass targeted the mean and did not converge.

### 2.7 Validation: the acceptance gates

The harness (`validate_against_real.py`) takes a simulated dataset and the real
reference, applies the shared estimators to both, and prints a pass/fail table plus
side-by-side distribution plots. Every tolerance is quoted from the build
specification and named in the report, so a result always states the tolerance it was
judged under.

| gate | what it tests | tolerance | result | headline number |
|---|---|---|---|---|
| T1 | spot scale / PSF width | ±0.3 px | **pass** | σ 1.453 vs real 1.478; FWHM-implied 1.527 vs 1.550 |
| T2 | blob density vs detection threshold | ±25% at thresholds 0.25, 0.32, 0.40, 0.50 | **marginal fail** | 0.91× / 0.83× / 0.73× / 0.70× of real density |
| T3 | per-spot photometry | medians ±25%, correct sign, overlapping IQRs, brightness AUC in 0.85–0.95 | **pass** | 7 of 8 medians within 10%; AUC 0.909 vs 0.909 |
| T4 | detector-mediated track duration | ±30% after length scaling, correct ordering | **marginal fail** | target 132 vs 135 ✓, spurious 29 vs 31 ✓, off-target 33 vs 49 ✗ |
| T5 | spot spacing | median ±2 px, floor ≥ 3.5 px, KS not rejected at α = 0.05 | **pass** | median 12.07 vs 12.04; KS p = 0.21 |
| T6 | end-to-end parameter transfer | real held-out F1 ≥ 0.70 | **pass** | 0.7143 (see §4.3c) |
| T7 | reproducibility | byte-identical per seed | **pass** | verified |
| T8 | consumers load the dataset unmodified | — | **pass** | 16/16 contract tests |

Four points about this table belong in the text.

**2.7.1 T2 and T4 sit within seed noise of their thresholds.** T4 passes on a single
seed (target 133 / off-target 38 / spurious 32) and fails when pooled over three
movies (33 for off-target, against the 34.3 needed); T2's threshold-0.40 ratio
measured 0.78× over two seeds and 0.73× over three. Tuning was stopped at that point
because further movement was fitting to noise. They should be reported as marginal
failures with the seed sensitivity stated, not as either passes or clean failures.

**2.7.2 One gate had to be redefined because as written it was not identifiable.**
The specification asked for a sweep of the detector's blob-scale window on both
movies, comparing the preferred window. Measured, the detector returns an *identical*
312 blobs per frame on the real movie for every upper scale bound from 0.8 to 3.0 once
the lower bound is 0.2 — every response comes from the smallest scale in the window,
so the window width carries no information. Worse, gating on it *passes* the older
simulator whose PSF the specification calls a mismatch. T1 was therefore redefined to
gate on directly measured PSF width, with the single-scale response curve kept as a
secondary check. That curve is itself informative: it collapses above σ ≈ 0.45 on both
movies, which is why the tuned window's *lower* bound is meaningful and its upper
bound is effectively inert.

**2.7.3 Two gates are in genuine tension, and both are intrinsically noisy.** The
detector rescales every frame by its own minimum and maximum before blob detection
(§3.3), so the detection threshold is a fraction of that frame's brightest
preprocessed pixel. Anything that produces bright outliers inflates the divisor and
pushes ordinary spots *below* threshold. Measured, raising the on-state amplitude
jitter from 0.25 to 0.40 *cut* the blob density at threshold 0.32 from 0.461 to 0.308;
lowering it to 0.10 raised the same density to 0.571 — the opposite of intuition. But
it cannot simply be minimised, because the detector also keeps only tracks whose
intensity variability clears a threshold, and at 0.10 the off-target class produced no
surviving events at all. 0.25 is the compromise, and no setting was found at which T2
and T4 both pass. The same mechanism makes both gates noisy: changing the unannotated
emitter count from 200 to 250 moved the density at threshold 0.32 from 0.733 to 0.559,
non-monotonically. Small movements in T2 and T4 must not be read as signal.

**2.7.4 What still does not match, stated plainly.**

| # | quantity | simulated | real | note |
|---|---|---|---|---|
| 1 | blob density at thresholds 0.40 / 0.50 | 0.73× / 0.70× | 1.00× | the high-threshold end depends on the brightest spot-frames, and everything that fattens that tail also inflates the normalisation divisor. No reachable setting was found. |
| 2 | off-target track duration | 33 frames | 49 | passes on some seeds; lengthening on-periods did not help (tested 120/265 → 170/375: 24 → 22 frames). What helped was brightness and width, and brightness is already at the edge of T3's tolerance. |
| 3 | off-target brightness spread | IQR ratio 2.50 | 1.97 | driven by per-emitter duty variance under slow switching, not by amplitude spread, and the amplitude knob is already near zero. Same cause puts the off-target variability at +22.3%, the widest of T3's eight medians. T3 still passes. |
| 4 | off-target brightness | 23.5 | 18.9 (1.24×) | a **deliberate trade**, and the one place a photometry target was knowingly given up for a detector-behaviour target. At the variability-implied 92 ADU the class was too faint to form tracks at all (10 events, 96% precision); at 110 ADU it gives 35 events and 78% precision against the real 76.6%. |
| 5 | nearest-neighbour minimum | ≈4.5 px | 3.61 px | the soft-core ramp makes sub-4 px spacings very rare. The real movie has 2 of 243 targets under 4 px (0.8%); the simulator has none. T5's floor and KS test both pass; the extreme tail is not reproduced. |
| 6 | background relative swing | 0.403 | 0.529 | autocorrelation length 24.3 vs 29.8 px. The random-field model reproduces the scale but under-reproduces the extremes. |
| 7 | signal-to-noise ratio | — | — | no agreed definition survives; see §2.8.3. Both sides measured identically agree (4.23 vs 4.19). |

### 2.8 Where measurement contradicted the build specification

Three claims in `SIMULATOR_SPEC.md` were not built to, because measurement contradicts
them. All three should be presented as findings, not as errata.

**2.8.1 The point spread function was never the mismatch — the most consequential
correction.** The specification argued the earlier simulator's σ = 1.5 was wrong
because the detector's tuned blob-scale range on the real movie is 0.45–1.27, and
predicted a calibrated value well below 1.5. Measured, the real spots are σ ≈ 1.5 —
the *same* as the earlier simulator's. The stacked radial profiles are nearly
identical (real 1.000 / 0.824 / 0.680 / 0.403 at radii 0, 1, 1.5, 2 px; earlier
simulator 1.000 / 0.808 / 0.651 / 0.369). The specification conflated the detector's
*tuned filter scale parameter* with the physical spot width. These are not the same
quantity: the filter scale interacts with the background high-pass, the per-frame
normalisation and the threshold. Building to the prediction would have made the
simulator measurably worse. The independent confirmation is that the earlier simulator
*passes* the redefined PSF gate.

**2.8.2 The blob-scale gate as written was not identifiable.** See §2.7.2.

**2.8.3 The quoted "median SNR ≈ 11" is not reproducible.** The specification directed
re-derivation using a function that takes *simulator parameters* (photon scale,
background level, read noise) and cannot be evaluated on a real movie of unknown gain,
so the instruction has no literal image-domain form. Four candidate definitions were
computed on the target class:

| definition | real movie |
|---|---|
| (peak − annulus median) / annulus standard deviation | 4.19 |
| (peak − annulus median) / √peak | 8.03 |
| peak / annulus standard deviation | 8.97 |
| (disc mean − annulus median) / annulus standard deviation | 1.56 |

None gives 11. The number traces back to a bracketed placeholder in an earlier draft
whose definition was never recorded. No acceptance gate depends on it, so it is
documented rather than resolved. **Do not quote "SNR ≈ 11" in the paper.** If a
signal-to-noise figure is wanted, state the definition and use 4.19 (real) against
4.23 (simulated), which agree because both sides go through the same estimator.

**2.8.4 The earlier simulators, and the before/after record.** The oldest simulator in
the project (`smTIRF_simulation.py`) is unusable for this purpose and is retained only
so its recorded baseline stays reproducible: it draws hard-edged filled circles and
blurs the whole summed frame with a single wide Gaussian, gives every spot an
identical amplitude, models "pulsation" as a deterministic sinusoid in spot *radius*
(so a spot vanishes when its radius rounds to zero), makes the negative class larger
and always-on (the wrong direction on every measured axis), has a flat background and
no photobleaching. The BachelorThesis simulator has sound physics — per-emitter
sub-pixel PSF with volume normalisation, hidden-Markov intensity traces, Poisson-then-
read noise, seeded throughout — and its physics was reused; only its calibration was
wrong. Running the validation harness on it reproduces that, and the same table
doubles as the before/after record:

| quantity | earlier (BachelorThesis) simulator | this simulator | real |
|---|---|---|---|
| off-target brightness (background-subtracted) | 66.41 (3.5× too bright) | 17.29 | 18.89 |
| target brightness | 89.83 | 63.06 | 60.57 |
| target duration / movie length | 0.333 | 0.263 | 0.269 |
| off-target duration / movie length | 0.180 | 0.066 | 0.098 |
| variability separates the two classes (AUC) | 0.955 | 0.709 | 0.580 |
| background relative swing | 0.162 | 0.403 | 0.529 |
| background autocorrelation length | 15.41 px | 24.29 px | 29.78 px |
| blob density at threshold 0.32 | 0.397 | 0.699 | 0.848 |
| target : off-target ratio | 2.00 | 2.670 | 2.670 |
| target nearest-neighbour minimum | 7.04 px | 4.45 px | 3.61 px |

Every row moves in the right direction. Gate totals: earlier simulator 1 of 5, this
simulator 4 of 6. The nearest-neighbour row independently confirms the placement
argument in §2.3(a): the earlier simulator's minimum is *exactly* its 7 px
configured floor, with zero pairs under 6 px where the real movie has 12.

### 2.9 Over-fitting risk

This must appear in the limitations, not only here. There is **one** annotated
recording and roughly a dozen simulator parameters were fitted to it. Every target
could be a property of this sample, this microscope and this annotator rather than of
single-molecule TIRF imaging in general. Specifically:

- the unannotated emitter population is a property of this preparation;
- the soft-core placement ramp partly describes how a person annotates crowded
  regions, not how molecules land on a surface;
- the slow switching timescales are entangled with the detector's 3-frame gap
  tolerance and its minimum track length — a different tracker would imply different
  values;
- the off-target class is whatever the annotator marked, reproduced statistically with
  no mechanistic claim.

§4.3f quantifies the part of this that can be quantified.

---

## 3. The SCP detector

### 3.1 What it measures, and what an "event" is

SCP identifies **events**: localised, approximately circular, positionally fixed
fluorescent spots whose brightness varies over time and which persist for many
frames. The output is one record per event, written as JSON:

| field | meaning |
|---|---|
| `x`, `y` | mean position over the event's lifetime, sub-pixel |
| `duration` | number of frames in which the spot was detected |
| `mean_intensity` | mean of the per-frame intensity readings |
| `mean_circularity` | mean of the per-frame shape scores |
| `max_displacement` | largest frame-to-frame movement within the event |
| `intensity_cv` | coefficient of variation of the intensity trace |

The four features beyond position are exported because they are what separates a
single molecule from an off-target object, and re-deriving them costs a full detection
pass. On the real movie the median duration is 134 frames for annotated molecules
versus 41 for annotated off-target spots — enough on its own to lift held-out
precision from 67.9% to 75.9% at a 20-frame cut. Exporting them lets a post-hoc filter
or a classifier be developed without re-running detection.

### 3.2 The processing pipeline

Applied to a `(frames, height, width)` stack, in this order.

1. **Per-frame smoothing.** A light Gaussian filter (σ = 0.5 px) to suppress
   single-pixel noise. Deliberately no global normalisation at this stage, which would
   destroy the intensity variation the classifier depends on. All subsequent
   measurements — intensity and shape — are made on this smoothed frame.
2. **Background suppression (optional, enabled throughout this work).** A broad
   Gaussian estimate of the background (σ = 8 px) is subtracted and negatives are
   clipped to zero. This is a high-pass filter: it flattens the uneven background
   described in §2.3(d). It is *necessary* on the real data. Without it, the uneven
   background dominates the per-frame normalisation in step 3, so recovering the dim
   real spots requires thresholds so low that the number of spurious candidates makes
   cross-frame linking computationally intractable. The same preprocessing is applied
   to simulated and real movies alike, so tuned parameters stay consistent with the
   pipeline used at evaluation time.
3. **Per-frame min–max normalisation.** The frame is rescaled to [0, 1] using its own
   minimum and maximum. This is the single most consequential implementation detail in
   the whole project; §3.3 explains why.
4. **Candidate detection.** Laplacian-of-Gaussian blob detection
   (`skimage.feature.blob_log`) over a scale range from `min_sigma` to `max_sigma`
   with an absolute response `threshold` and an overlap parameter of 0.5. This is a
   standard multi-scale spot detector: it responds to compact bright features whose
   size is within the scale window.
5. **Per-spot photometry.** Two modes. The default (used for every number on record)
   is the legacy measurement: the mean over a disc of radius `int(sigma × 1.5)` with no
   background subtraction. The alternative is proper aperture photometry: the mean over
   a disc of a configured radius minus the median of a surrounding annulus. The legacy
   mode is poor photometry — at the tuned scale range the truncation makes the radius
   0 or 1 px, so it is close to a single-pixel sample sitting on the full background
   pedestal — and it is kept as the default *on purpose*; §3.4 explains.
6. **Per-spot shape score.** In a small window around each candidate, pixels above
   `mean + 0.5 × standard deviation` are binarised, the largest contour is taken, and
   circularity is computed as `4π × area / perimeter²`, capped at 1. A perfect circle
   scores 1.
7. **Cross-frame linking.** Candidates are linked into tracks frame by frame. Active
   tracks are indexed in a k-d tree (a spatial data structure for fast
   nearest-neighbour queries); each new candidate is matched to its nearest unclaimed
   active track within `max_displacement` pixels, greedily; unmatched candidates start
   new tracks; a track with no match for more than 3 consecutive frames is closed.
   Note two properties: the assignment is greedy in candidate order rather than
   globally optimal, and the 3-frame gap tolerance is hard-coded, not a search
   parameter.
8. **Track merging.** Tracks whose mean positions lie within `track_merge_dist` pixels
   are merged, using a union-find structure over pairs found in a k-d tree. This
   catches one molecule that was localised slightly differently across frames and so
   split into two tracks. The radius is fixed at 1.0 px for every number reported here
   (it is not part of the search). It cannot be raised freely: nearest-neighbour
   distances between annotated spots are median 10.4 px overall and 12.0 px among
   targets, but six pairs sit closer than 5 px, so a large radius starts merging
   genuinely distinct molecules.
9. **Event classification.** A merged track becomes an event only if it satisfies all
   four criteria:
   - **length** — at least `min_pulsation_frames` detected frames;
   - **shape** — mean circularity at least `circularity_threshold`;
   - **fixedness** — largest frame-to-frame displacement at most
     `2 × max_displacement` (note the factor of two relative to the linking radius);
   - **intensity variation** — coefficient of variation at least
     `intensity_variation_threshold`, i.e. high-variability tracks are *kept*.
10. **Duplicate suppression.** A final greedy pass over classified events: where two
    surviving events lie within `event_nms_radius` pixels, the longer-lived one is
    kept. This catches what step 8 cannot — a molecule that goes dark for longer than
    the 3-frame gap tolerance has already been split into two tracks whose mean
    positions may differ by several pixels, and merging compares mean positions of
    tracks that both already exist. At 3.0 px this is provably lossless on the real
    movie, whose closest annotated pair is 3.61 px apart, so no two annotated
    molecules can be fused. Measured on the held-out region it removed 8 of 50 false
    positives with the true-positive count unchanged (precision 67.9% → 71.6%). The
    radius must stay below the minimum nearest-neighbour distance of whatever
    annotation set is in use.

The detector also reports a count of track merges alongside the event count; that is a
diagnostic and plays no part in scoring.

### 3.3 Why per-frame normalisation matters so much

Because each frame is rescaled by its own minimum and maximum *before* blob detection,
`threshold` is not an absolute brightness — it is a fraction of that frame's brightest
preprocessed pixel. Three consequences run through this entire work:

1. **Thresholds are not comparable between movies with different background structure
   or different brightness distributions.** This, not the physics, is why parameters
   tuned on the earlier simulator did not transfer: the usable threshold was about
   0.05 on those movies and at least 0.25 on the real one. Correcting the background
   model and the emitter population is what brought the two ranges into contact.
2. **Rescaling brightness cannot fix it.** Min–max normalisation is scale-invariant; a
   threefold brightness sweep leaves the measured filter response flat.
3. **Only *densities* may be compared, never raw blob counts**, across frames of
   different size — and a single bright outlier pixel can rescale a whole frame, which
   is the source of the gate noise in §2.7.3.

This is a known property of the implementation, not a bug introduced here, and it is
left in place so that every number on record remains comparable. The paper should state
it explicitly, because it is the mechanism behind the central methodological result in
§4.

### 3.4 Parameters, and two known limitations kept deliberately fixed

**Searched parameters** (seven; see §4.2 for bounds):
`min_sigma`, `max_sigma`, `threshold`, `circularity_threshold`, `max_displacement`,
`min_pulsation_frames`, `intensity_variation_threshold`.

**Pipeline switches, fixed for every run on record:** background suppression on with
σ = 8 px; duplicate-suppression radius 3.0 px; legacy photometry; track merge radius
1.0 px; gap tolerance 3 frames.

**A naming trap that must not reach the paper.** `min_pulsation_frames` is *not* a
pulsation criterion, despite the name. It is applied purely as a minimum track length.
The actual variability test is `intensity_variation_threshold`. Describe it in the
text as a minimum track duration.

That parameter is also a strong discriminator on the real movie, and the reason is
worth recording because it is a property of this configuration rather than of
photophysics. Median track length is 134 frames for annotated molecules, 41 for
annotated off-target spots and 25 for unannotated false positives, so raising the floor
strips off-target objects about five times faster than molecules. Measured on the
held-out region, precision runs 67.9% at a floor of 11, 76.7% at 20, 85.4% at 40 and
88.1% at 60, trading recall along a smooth curve. But the mechanism is detector
instability, not biology: off-target objects are bright and persistent, yet produce
short tracks here because the tuned scale range plus the 8 px high-pass detects these
larger objects only intermittently, so their tracks fragment. On simulated movies from
the earlier simulator the ordering reverses entirely (targets 241 frames, aggregates
269), which is why the search bound on this parameter is set per protocol rather than
globally (§4.2).

**Two limitations left unfixed on purpose.** Changing either would invalidate every
recorded number, so both were ruled out of scope and should be listed as future work:

1. *Legacy photometry.* Switching to proper aperture photometry improves class
   separation modestly for brightness (AUC 0.899 → 0.909) and substantially for
   variability (0.580 → 0.748), but it redefines both `mean_intensity` and
   `intensity_cv`, invalidating every tuned `intensity_variation_threshold` on record.
2. *The direction of the variability test.* With the background properly subtracted,
   annotated off-target spots show *higher* variability than molecules (1.596 versus
   0.791), partly because their near-zero mean inflates the ratio. Since the classifier
   keeps high-variability tracks, a higher threshold would then preferentially retain
   off-target objects — the opposite of what is wanted — and the searched bounds sit
   far below the observed range. Enabling proper photometry therefore requires
   revisiting the direction *and* the bounds of this criterion. It is not a drop-in
   change.

### 3.5 The reference detector configuration

Wherever "the detector" is invoked for calibration or for simulator acceptance, this
configuration is used, and it is not re-tuned as part of that work:

```
min_sigma 0.4515   max_sigma 1.2690   threshold 0.3234
circularity_threshold 0.2021   max_displacement 4.9934
min_pulsation_frames 20   intensity_variation_threshold 0.0718
background suppression on, sigma 8.0   duplicate-suppression radius 3.0
```

Held-out performance on the real movie: recall 94.6%, precision 76.6%, F1 0.847. Note
for accuracy in the text: this configuration is *not* a raw search output. The search
returned the same six shape parameters with a minimum track length of 11 and F1 0.800;
0.847 comes from raising that floor to 20 and adding 3 px duplicate suppression by
hand. Both numbers are legitimate, but they answer different questions, and §4.5
depends on keeping them apart.

---

## 4. Parameter optimization and evaluation

### 4.1 The objective: how a detection is scored

**Matching rule.** Detections are matched to annotated positives by greedy one-to-one
nearest-neighbour assignment. All detection–annotation pairs within a fixed margin are
formed — the margin is applied per axis (|Δx| ≤ 3 px and |Δy| ≤ 3 px) — and then
assigned in order of increasing Euclidean distance, each detection and each annotation
used at most once.

**Counting.**
- **True positives (TP)** — matched pairs.
- **False negatives (FN)** — annotated positives left unmatched.
- **False positives (FP)** — *every* detection not matched to a positive. This
  includes detections on annotated off-target spots and detections on unmarked
  background features, so over-detection is penalised directly.
- A secondary count, how many annotated negatives were fired on, is reported but never
  optimised.

**Metrics.** recall = TP / (annotated positives); precision = TP / (TP + FP);
Fβ = (1 + β²) · precision · recall / (β² · precision + recall), with β = 1 (F1) used
throughout as both the search objective and the selection metric.

**Why F1 and not a weighted sum of recall and precision.** An earlier version of the
scorer used a linear weighted sum (0.6 recall + 0.4 precision), which is retained for
reporting only so that old runs remain comparable. It is unsuitable as an objective
for two reasons: it is fully compensatory, so high recall can mask poor precision, and
its weights carry no interpretable trade-off. F1 penalises imbalance, is standard for
spot-detection tasks, and — being non-convex in the precision/recall plane — does not
collapse whole regions of the trade-off into ties.

**One assumption to state.** Treating every unmatched detection as a false positive
assumes the annotations are exhaustive within the scored region. They are not
(§1.1), so reported precision is a *lower bound* on true precision. This is the
conservative direction, it applies equally to every protocol compared, and the
simulator reproduces the same situation on purpose (§2.4).

**One scorer only.** The same function computes whole-movie scores, region-restricted
scores, and the simulator's acceptance measurements. Nothing re-implements it.

### 4.2 The search

**Algorithm.** Bayesian optimization with a Tree-structured Parzen Estimator (TPE)
sampler (`optuna`), maximizing F1. TPE builds a probabilistic model of which parameter
regions produce good scores and samples from it. Unlike one-parameter-at-a-time
(coordinate-descent) tuning, which the project used previously, it samples the full
seven-dimensional space jointly and so accounts for interactions — the blob scale
range and the detection threshold are strongly coupled, and tuning either alone is
misleading.

**Search bounds.**

| parameter | role | range | justification for the range |
|---|---|---|---|
| `min_sigma` | smallest blob scale (px) | 0.0 – 1.5 | real spots are diffraction-limited; keep the window tight |
| `max_sigma` | largest blob scale (px) | 0.5 – 3.5, and constrained to exceed `min_sigma` by 0.1 | a degenerate window wastes an expensive trial |
| `threshold` | relative detection threshold | 0.25 – 0.5 | floored at 0.25 by measurement: with background suppression on, about 0.30 recovers the real spots (~95% recall) in ~130 s, while below about 0.20 the real movie becomes intractable — linking explodes and the run times out. Keeping the search inside the tractable band is what lets every candidate actually be judged on real data. |
| `circularity_threshold` | minimum shape score | 0.1 – 0.6 | narrowed toward the recall-preserving region: a lenient probe reached ~95% recall, while aggressive shape filtering rejects the irregularly blinking real spots |
| `max_displacement` | linking radius (px) | 0.5 – 5.0 | — |
| `min_pulsation_frames` | minimum track length (frames), integer | 3 – 15 by default; 3 – 60 for the direct-on-real protocol | see below |
| `intensity_variation_threshold` | minimum coefficient of variation | 0.05 – 0.25 | same recall-preserving argument as shape |

**Why the track-length bound is set per protocol.** On movies from the earlier
simulator the parameter is a dead dimension — aggregates there last *longer* than
targets (269 versus 241 frames) and precision is flat for every cut from 5 to 100 — so
a raised ceiling would only give TPE a useless dimension to spend trials on, and a
high value would strip targets before aggregates. On the real movie the ordering is the
opposite and the parameter is one of the strongest available (§3.4). The ceiling is
therefore 15 for searches performed on simulated movies and 60 for searches performed
directly on the real movie. Both are absolute frame counts and so do not transfer
between movies of different length; this must be revisited if movie length changes.

The current simulator reproduces the real ordering (target 132 > off-target 33 >
spurious 29 frames), so the original justification for the low ceiling no longer
applies to it. That the ceiling was nevertheless left in place for comparability is
what §4.5 is about.

**Warm starts.** The direct-on-real search is seeded with four manually identified
configurations before TPE begins: three historical recall-preserving probes, plus one
that is a known-good configuration with the track-length floor raised to 20. Without
the fourth, all seeds would sit at the very bottom of the newly opened range and the
search would spend its early trials re-exploring the region already known to yield
67.9% precision. Warm starts are recorded in the results file so two runs can be told
apart by what seeded them.

**Robustness measures, all of which matter for reproducibility.**
- The sampler is seeded (42).
- Each detector run is a subprocess with a wall-clock timeout (600 s by default, 900 s
  in the transfer experiment), so one pathological parameter set cannot stall the
  search.
- Each run writes to an explicit output path, which is deleted beforehand; a missing
  output is scored as a failure. There is no "newest file in the directory" fallback,
  which would silently score against the label files sitting in the same folder.
- A failed run yields a zeroed metrics record rather than an exception, so the search
  continues.
- The child process is forced to UTF-8. This is not cosmetic: the detector prints a
  non-ASCII character, and on a Windows console with a captured pipe that raised an
  encoding error *before* the output JSON was written, while the process still exited
  successfully — silently poisoning the score.
- Trials run one at a time, because parallel trials would need per-trial rather than
  per-movie output paths.

### 4.3 Evaluation designs

This is the part of the methods most likely to be questioned, so it is set out in
full. Every protocol below reports a held-out number that was used for no decision.

**(a) Spatial cross-validation on the real movie ("direct on real").**
With only one annotated recording, an independent train/validation/test split has to
come from *within* the field of view. The movie is partitioned into three disjoint
vertical bands by column index: train `[0, 170)`, validation `[170, 340)`, test
`[340, 512)` px. These partition the annotated spots into disjoint sets.

The detector processes the whole movie in one pass, so each trial produces detections
across the entire field and the three band scores are obtained by restricting the
matching to spots and detections within each band. Every trial therefore costs one
detection while still yielding three scores. Parameters are optimized on the train
band; the final configuration is chosen from the top-K train trials by validation-band
F1; the test band is reported and was used for no decision at any point.

Result (40 trials): test F1 **0.800**, recall 94.9%, precision 69.2%, with train
0.839 and validation 0.806 — closely matching scores across the three bands, which
indicates the configuration is not tuned to one region. Raising the track-length floor
from 11 to 20 and adding 3 px duplicate suppression by hand gives the reference
configuration of §3.5 at F1 0.847 (recall 94.6%, precision 76.6%).

*The limitation to state.* Bands of one movie are not independent samples in the sense
that separate recordings would be. They share the illumination profile, the background
structure, the preparation and the annotator. Spatial cross-validation controls for
over-fitting to a *region*; it cannot control for over-fitting to the recording.

**(b) Train and validate on simulation, test once on real ("sim only").**
Simulated movies are split by seed into disjoint training and validation groups. The
objective is the mean per-movie F1 across the training movies — each movie weighted
equally, the standard cross-validation aggregation. The final configuration is the
top-K training trial re-scored on the validation movies, and the real movie is scored
exactly once at the end. This is the classic design; it is included for completeness
and because the results file format supports it.

**(c) Search on simulation, select on real ("simulation-guided") — the headline
protocol.**
The two halves of parameter tuning are separated: *generating* candidate
configurations, which is cheap on simulation and needs no annotation, and *choosing*
among them, which needs real labels.

1. TPE search on the simulated training movies, objective = mean per-movie F1.
2. The top-K configurations (K = 6) are each run **once** on the real movie. The
   detections are sliced by column into a validation band `[0, 256)` and a held-out
   test band `[256, 512)`.
3. The winner is chosen by validation-band F1. The test band is reported.

Nothing in the scorer, the matching margin, the detector switches or the band
boundaries differs from (a); only the source of the candidates does. That is what
makes the comparison meaningful.

Result (30 trials, 2 simulated training movies, 3.3 h): real test F1 **0.7143**,
recall 90.1%, precision 59.2%, against **0.256** for the same protocol run with the
earlier simulator — a 2.8-fold improvement, and above the pre-registered target of
0.70.

**(d) Transfer the shape parameters, recalibrate the threshold on real data.**
A variant motivated directly by §3.3: because the detection threshold is a fraction of
each frame's brightest pixel, its usable range is a property of the movie, and the two
datasets' ranges did not overlap. The other six parameters describe the physical size
and behaviour of a spot and are not tied to a frame's intensity scaling, so they should
transfer. The protocol searches all seven on simulation, then for each top-K
configuration sweeps the threshold over a small grid (0.20, 0.26, 0.32) on the real
movie, selects the single (configuration, threshold) pair by validation-band F1, and
reports its test band. The grid is bounded by measurement: on the real movie 0.20 runs
in 184 s at F1 0.699, 0.15 takes 320 s and *degrades* to F1 0.213 because spurious
blobs hijack track association, and 0.12 exceeds a 420 s timeout entirely.

**(e) The random-search control — this is what makes the headline result
interpretable.**
Every configuration with a real-movie score on record was first chosen by its
*simulated* score, so protocol (c) alone cannot separate two contributions: the
simulator concentrating the search, and the real-labelled selection step picking the
winner. The control removes the simulator and changes nothing else — six configurations
drawn uniformly from the same bounds (matching K), same scorer, same detector
switches, same bands, same selection rule.

| method | real labels used | real detector runs | real test F1 |
|---|---|---|---|
| 6 random configurations + real-validation selection | validation band | 6 | 0.4505 |
| picking by simulated score alone, no real labels | none | 1 | 0.5591 |
| simulation search + real-validation selection (c) | validation band | 6 | **0.7143** |
| direct-on-real search (a) | train + validation bands | 100+ | 0.800 |

The simulator's candidate pool is worth **+0.26 F1** over random draws at the same
annotation budget and the same number of real-movie detections. Two of the six random
configurations detected nothing at all; the simulation-guided search never produced a
candidate that bad. The margin is far outside the ±0.05 ambiguity a six-draw
single-seed estimate warrants. Runtime: 16 min.

*One comparison remains open and should be acknowledged.* The simulation-guided
protocol screens 30 candidates on simulation, using no real labels and no real
detector time, and spends 6 real detections on the survivors — the simulator functions
as a cheap screening stage. Whether 30 *random* draws, all evaluated on the real movie,
would also reach about 0.71 is untested; it would cost five times the real-movie
detector time, which is the expensive resource. If it did, the simulator would be
saving compute rather than improving quality.

**(f) The leakage control: recalibrating the simulator without the test band.**
Protocol (c) has a subtle leak. Detector parameters are selected on the validation
band only, so it is not circular in the usual sense — but the *simulator itself* was
calibrated against statistics of the whole movie, test band included. A second arm
therefore recalibrates the simulator using only annotations from columns `[0, 256)`,
so the test band influenced nothing. Pixel-derived quantities (background, noise, blob
density) are still taken from the whole frame, deliberately: a new recording would
supply unlabelled pixels too, so those are not a leak.

The two bands genuinely differ — validation-band target variability 0.583 against the
whole movie's 0.791, off-target brightness 16.0 against 18.9, spot density 2.76 against
2.55 per 1000 px — so this is a real recalibration rather than a formality.

| arm | simulator calibrated on | real validation F1 | real test F1 | recall | precision |
|---|---|---|---|---|---|
| full | whole movie | 0.7524 | **0.7143** | 90.1% | 59.2% |
| leak-free | validation band only | 0.5377 | **0.6852** | 66.7% | 70.5% |

**Test-band gap: 0.029 F1** (4% relative). Both arms are about 2.7× the 0.256
baseline, so the headline result is not mostly leakage: removing every trace of
test-band annotation costs about 0.03 F1 and lands at 0.685 — marginally under the
0.70 bar rather than collapsing toward the baseline.

The comparison is sound in a checkable way: both arms use the same sampler seed, so
their early random trials evaluate identical parameter sets, and the real-movie
evaluation is deterministic. One trial appears in both candidate pools and scores
identically (test 0.5591, validation 0.4663) — a direct check that the two runs measure
the same thing.

**0.029 is an upper bound on leakage, not an estimate of it.** Three effects are
confounded and cannot be separated with one recording: leakage itself; the fact that
the validation band is not representative of the whole movie (higher spot density,
higher duty cycle — on identical shared parameter sets the leak-free arm scored
consistently lower, confirming that dataset is genuinely harder); and search variance
(the leak-free search never found as strong a region, best simulated F1 0.675 against
0.758, so its candidate pool was drawn from weaker configurations). Because the latter
two both push the leak-free arm *down*, the true leakage cost lies somewhere between 0
and 0.029. Separating them requires a second annotated recording — which is, per §2.1,
the deeper reason the simulator exists.

Also worth noting: the two arms reach similar F1 by different routes — the full arm is
recall-heavy (90.1% / 59.2%), the leak-free arm precision-heavy (66.7% / 70.5%).
Neither matches the direct-tuning balance (94.6% / 76.6%), and precision remains the
weak side of simulation-guided tuning.

### 4.4 What the simulator does and does not do

The three roles a simulator might play in parameter tuning are now separated and each
is measured. This table is the honest summary and should appear close to the headline
result.

| role | verdict | evidence |
|---|---|---|
| concentrates the search on a useful region | **yes**, worth +0.26 F1 | random draws at the same budget give 0.4505 (§4.3e) |
| ranks candidates by their real performance | **no** | Spearman ρ ≈ −0.54 among the finalists; trusting the ranking costs 0.16 |
| removes the need for real annotations | **no** | a labelled region is still required to select |
| beats tuning directly on real labels | **no** | 0.7143 against 0.800 |

The ranking result deserves its own numbers, because it constrains how the headline
should be phrased. The six configurations carried from the simulated search to the
real movie:

| trial | simulated F1 | real validation F1 | real test F1 |
|---|---|---|---|
| 22 | 0.6230 | **0.7524** | **0.7143** |
| 23 | 0.6124 | 0.7075 | 0.7046 |
| 24 | 0.5161 | 0.6667 | 0.6447 |
| 13 | **0.7582** | 0.4663 | 0.5591 |
| 27 | 0.6826 | 0.2844 | 0.4306 |
| 26 | 0.6798 | 0.2011 | 0.3810 |

Simulated and real scores are *anti-correlated* among the finalists. The best
configuration on the simulation is the fourth best on the real movie, and picking by
simulated score alone would have given 0.559 instead of 0.714. So the "select on the
real validation band" half of the protocol is doing real work, and the result must
**not** be reported as "the simulation now predicts real performance". The supported
claim is weaker and still useful: *the simulator generates a candidate pool that
contains configurations which work on the real movie, where the previous one did not.*
Any protocol without a real-labelled selection step lands near 0.56.

This also bounds what a better simulator could buy. Beating direct tuning would
require the simulated score itself to be a trustworthy ranking signal, and at
ρ ≈ −0.54 it is not close.

**Practical framing for the discussion.** On a new recording, this saves the
*parameter search*, not the *annotation*. A labelled region is still needed — about 180
spots here — but a supervised search against those labels is not.

### 4.5 Two claims that must be handled carefully in the write-up

**4.5.1 The superseded conclusion.** The earlier draft concluded that the simulator
"is not an adequate surrogate for parameter selection on this real data." That
conclusion held for the simulator available at the time and no longer holds for the
current one: 0.256 → 0.7143 under an unchanged protocol, and 0.6852 under the stricter
leak-free variant. Present this as a corrected conclusion with both numbers, not as a
quiet revision.

**4.5.2 A post-hoc improvement that does not survive scrutiny — do not quote 0.8276.**
Because the current simulator reproduces the real track-duration ordering, the low
ceiling on the track-length parameter (§4.2) is no longer justified for it. Sweeping
that one parameter after the fact on the winning configuration, selecting on the real
validation band and reporting the held-out test band, gives:

| track-length floor | real validation F1 (selects) | real test F1 | recall | precision |
|---|---|---|---|---|
| 8 (chosen under the ceiling) | 0.7524 | 0.7143 | 90.1% | 59.2% |
| 20 | 0.7811 | 0.7656 | 88.3% | 67.6% |
| 60 | 0.8444 | 0.8251 | 82.9% | 82.1% |
| **100** ← best on validation | **0.8594** | **0.8276** | 75.7% | 91.3% |
| 150 | 0.8103 | 0.7473 | | |

0.7143 → 0.8276, above the direct-on-real *search* result and just below the
hand-tuned reference, with an interior rather than boundary optimum. It is tempting,
and it does not hold up. Running the identical sweep on the leak-free arm:

| arm | capped search | + track-length sweep | gain | selected floor |
|---|---|---|---|---|
| full (simulator saw whole movie) | 0.7143 | 0.8276 | **+0.113** | 100 |
| leak-free | 0.6852 | 0.7129 | **+0.028** | 20 |

The gain is four times larger on the arm whose simulator was calibrated against the
whole movie, the gap between arms widens from 0.029 to 0.115, and the leak-free sweep
lands on a far more conservative floor (20, inside the sanctioned range) rather than an
extrapolated 100. That pattern is what a post-hoc adjustment exploiting information the
strict protocol would not have looks like. The confound of §4.3f cuts the other way —
the leak-free arm also had a weaker search and a harder dataset — but the asymmetry is
large enough that the honest reading is "the gain is not established", not "the gain is
real and worth 0.11".

**The defensible figures are 0.7143 (protocol as specified) and 0.7129 (leak-free,
with the sweep), which agree closely, and neither reaches the direct-tuning result.**

Three further caveats on the 0.8276, independent of the leakage argument: it adjusts
one parameter post hoc on a configuration whose other six were optimized under the
ceiling, so a full re-search may land elsewhere; demanding a 100-frame track in a
501-frame movie is aggressive and may be specific to this recording; and it reaches
that score by the *opposite* route to the reference configuration (75.7% recall /
91.3% precision against 94.6% / 76.6%) — the same number, a very different detector.

One methodological note in the sweep's favour, which should be stated so the sweep is
not mistaken for a shortcut: it was performed by filtering saved detections on their
exported duration, which is exactly equivalent to re-running the detector. This was
verified against a real run at a floor of 100, which produced an identical 209-event
set. The equivalence holds because the length filter is applied before classification
and the duplicate-suppression pass keeps the longest-lived track in a cluster, so a
shorter track can never displace a longer one.

**4.5.3 Which baseline the headline is compared against.** Read 0.256 as the thing
being beaten, not 0.847. The 0.847 comes from tuning on the real movie's own labels and
then hand-improving the result — an oracle, and not a search output. The like-for-like
search-versus-search comparison is 0.7143 against 0.800.

### 4.6 Reproducibility and implementation

- **Seeding.** Simulator movies are fully seeded (§2.5); the TPE sampler is seeded;
  real-movie evaluation is deterministic.
- **No implicit file discovery.** Every detector run writes to an explicit path that is
  deleted first; missing output is a scored failure.
- **Timeouts** on every detector run, so the search cannot stall.
- **Software.** Python 3.12.8 verified for the simulator; the detector pins Python
  3.11 on the original machine. `numpy` ≥ 1.24 (seed-sequence splitting), `scipy`
  ≥ 1.10, `scikit-image` ≥ 0.20 (blob detection), `tifffile`, `PyYAML`, `matplotlib`,
  `opencv-python` (shape scoring), `optuna` ≥ 3.4, `pandas`. **The author must record
  the exact installed versions**, and one caution belongs in the text: the blob
  detector's version should be held fixed, because a change in its behaviour would
  look like a simulator regression on the density gate.
- **Runtime.** A 501 × 256 × 512 movie takes roughly 1–3 minutes to detect, depending
  on how many candidates the parameters produce. The headline transfer run (30 trials
  × 2 training movies, plus 6 real-movie detections) took 3.3 h; the random control
  took 16 min; the direct-on-real search ran 40 trials. **Hardware must be filled in
  by the author.**
- **A known rough edge to fix before release, not to write about.** Several scripts
  carry absolute paths from a different machine (`C:\Users\PC\Desktop\…`). The
  experiment drivers in `experiments/` override them programmatically, which is why
  those runs work; a direct invocation of the parametrizer needs its paths adjusted
  first.

### 4.7 Commands that reproduce everything

```bash
# --- simulator: measure, build, validate -------------------------------------
cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/Simulation
python measure_real_movie.py                                  # -> config/real_movie_reference.json
python measure_real_movie.py --band 0 256                     # leak-free reference
python generate_dataset.py                                    # -> dataset/{train,val}/ + manifest.json
python generate_dataset.py --out-dir dataset_valband --config config/real_calibrated_valband.yaml
python validate_against_real.py --manifest dataset/manifest.json          # gates T1-T5, T7
python validate_against_real.py --manifest dataset_bt/manifest.json --label BT-sim
python tests/test_contracts.py                                # 16 contract tests
python calibrate.py                                           # report the current fit

# --- detector: one movie -----------------------------------------------------
cd ../SCP
python SCP_KTree_JSON.py <movie.tif> --output_json out.json --no_plot \
    --min_sigma 0.4515 --max_sigma 1.2690 --threshold 0.3234 \
    --circularity_threshold 0.2021 --max_displacement 4.9934 \
    --min_pulsation_frames 20 --intensity_variation_threshold 0.0718 \
    --bandpass --bandpass_bg_sigma 8.0 --event_nms_radius 3.0

# --- tuning and evaluation ---------------------------------------------------
cd ../experiments
python real_calibrated_sim_transfer/run_transfer_t6.py --arm full    --trials 30 --top-k 6 --train-movies 2
python real_calibrated_sim_transfer/run_transfer_t6.py --arm valband --trials 30 --top-k 6 --train-movies 2
python real_calibrated_sim_transfer/run_random_control.py --candidates 6 --seed 0
python direct_real_spatial_crossval/run_direct.py
```

Note: the detector's own command-line defaults (blob scales 30–40 px) are historical
and should never be used; always pass parameters explicitly.

---

## 5. Results table for the paper

Every number below has a named source. All F1 values are computed by the same scorer
at a 3 px matching margin. Rows marked *held-out band* are on columns 256–512 of the
real movie; the direct-on-real search splits the movie three ways and so reports on
columns 340–512, which is not directly comparable and is labelled accordingly.

| protocol | region | real F1 | recall | precision | source |
|---|---|---|---|---|---|
| tuned on the earlier simulator, applied to real | 256–512 | 0.256 | 18.0% | 44.4% | `experiments/sim_guided_real_selection/` |
| 6 random configurations + real-validation selection | 256–512 | 0.4505 | 36.9% | 57.7% | `experiments/real_calibrated_sim_transfer/random_control_results.json` |
| current simulator, picking by simulated score alone | 256–512 | 0.5591 | — | — | `optimization_results_full.json`, trial 13 |
| **current simulator, search + real-validation selection** | 256–512 | **0.7143** | 90.1% | 59.2% | `optimization_results_full.json` |
| same, leak-free simulator calibration | 256–512 | 0.6852 | 66.7% | 70.5% | `optimization_results_valband.json` |
| direct search on the real movie | *340–512* | 0.800 | 94.9% | 69.2% | `experiments/direct_real_spatial_crossval/` |
| the same configuration, hand-improved (reference) | 256–512 | 0.847 | 94.6% | 76.6% | §3.5 |
| reference detector on a simulated movie | whole frame | 0.837 | 88.3% | 79.6% | `Simulation/CALIBRATION.md` §4.1 |

**Do not include** the 0.8276 post-hoc figure (§4.5.2) or the "SNR ≈ 11" figure
(§2.8.3).

Suggested figures, all already produced by the harness: side-by-side distributions of
nearest-neighbour distance, per-spot brightness, intensity variability, track duration,
and blob density versus threshold (six plots from `validate_against_real.py`); the
gate pass/fail table (§2.7); the simulated-versus-real scatter of the six finalist
configurations (§4.4); and the method comparison bar chart (§5).

---

## 6. Limitations, in the order they should be conceded

1. **N = 1 annotated recording.** Every performance estimate comes from partitioning
   one field of view. Bands of one movie share illumination, background, preparation
   and annotator, so spatial cross-validation controls for over-fitting to a region,
   not to a recording. No estimate of between-recording variability is possible.
2. **Roughly a dozen simulator parameters were fitted to that one recording** (§2.9).
   Some of them plausibly describe the preparation or the annotator rather than the
   physics.
3. **The annotations are not exhaustive and their off-target criterion is
   undocumented** (§1.1). Reported precision is a lower bound; the off-target class is
   reproduced statistically with no mechanistic claim. Labels are one person's
   judgement, and no inter-annotator agreement was measured.
4. **Two acceptance gates fail marginally** (§2.7.1) and six quantities remain
   unmatched (§2.7.4), including one place where a photometry target was knowingly
   traded for a detector-behaviour target.
5. **The simulator does not rank candidates** (§4.4). It is a screening device, not a
   predictor of real performance, and any protocol built on it still needs a labelled
   region for selection.
6. **Several calibration targets are detector-mediated** and therefore entangled with
   this particular detector's gap tolerance and minimum track length. A different
   tracker would imply different simulator parameters.
7. **Two detector limitations were deliberately left in place** to keep every recorded
   number comparable (§3.4): the legacy per-spot photometry and the direction of the
   intensity-variation test. Fixing them is a coupled change requiring its own
   held-out measurement.
8. **Absolute frame counts do not transfer** between movies of different length; the
   track-length bounds must be revisited if movie length changes materially.
9. **One comparison remains open**: whether the same number of *random* candidates,
   all evaluated on the real movie, would match the simulation-guided result (§4.3e).

---

## 7. Writing checklist

**State these explicitly; each is load-bearing and each is easy to omit.**
- The detector normalises each frame by its own extremes, so its threshold is relative,
  not absolute — this is the mechanism behind the transfer result (§3.3).
- `min_pulsation_frames` is a minimum track duration, not a pulsation test (§3.4).
- Every unmatched detection counts as a false positive, so precision is a lower bound
  (§4.1).
- The 500 unannotated simulated emitters are in no ground-truth file, deliberately
  (§2.4).
- Which baseline the headline beats: 0.256, not 0.847 (§4.5.3).
- The simulator supplies candidates; real labels still select (§4.4).

**Never quote:** "median SNR ≈ 11" (§2.8.3); the post-hoc 0.8276 (§4.5.2); the earlier
draft's conclusion that the simulator is not an adequate surrogate (§4.5.1); the
detector's built-in command-line defaults as if they were tuned values (§4.7).

**Placeholders only the authors can fill:** acquisition metadata and pixel size (§1.1);
annotator identity, criterion and any agreement measure (§1.2); exact installed package
versions (§4.6); hardware and total compute (§4.6).
