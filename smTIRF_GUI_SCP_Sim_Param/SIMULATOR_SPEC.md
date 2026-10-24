# Spec — real-calibrated smTIRF simulator (replaces `Simulation/`)

**Status:** IMPLEMENTED — see `Simulation/CALIBRATION.md` for measured-vs-target,
the gate table, and what remains unmatched. Headline: gate T6 passes at real test
F1 **0.7143** (target ≥ 0.70) against 0.256 for the previous simulator. T1, T3, T5,
T7 and T8 pass; T2 and T4 fail marginally, within seed noise of their thresholds.
A leak-free rerun, with the simulator calibrated only on the validation band, gives
0.6852–0.7129 — so the headline is not an artifact of calibrating against the whole
movie. Do not quote the 0.8276 that appears in CALIBRATION.md §1.2; it does not
survive that check.

⚠️ **Three claims in this document are contradicted by measurement** and were not
built to. Read `CALIBRATION.md` §3 before trusting §3b row 1, §7-T1 or §4b:
the real PSF is σ ≈ 1.5 — the *same* as the BachelorThesis simulator's, not "well
below" it (§3b row 1 and R1 confuse the detector's tuned LoG σ parameter with the
physical spot width); T1's sigma-window sweep is not identifiable and passes the
very simulator §3b calls wrong; and §4b's "median SNR ≈ 11" is not reproducible
under any definition.

**Audience:** a coding agent (or person) building the replacement end to end.
**Written:** 2026-07-30. **Implemented:** 2026-07-31 / 2026-08-01.

---

## 1. Mission

Build a simulator whose movies are close enough to `realdata/vid_ch1.tiff` that
**detector parameters tuned on the simulation transfer to the real movie**. That
single sentence is the whole acceptance test. Everything in §7 exists to measure it.

This replaces `Simulation/` (`smTIRF_simulation.py`, `generate_dataset.py`,
`generate_dataset_bt.py`). It borrows its physics from the BachelorThesis
simulator and its calibration from measurements of the real movie.

### Why this is worth doing

Parameters tuned on the current simulations do not transfer. Measured, held-out:

| protocol | real test-band F1 | recall | precision |
|---|---|---|---|
| tuned on BT sim, applied to real | 0.256 | 18.0% | 44.4% |
| tuned directly on the real movie | 0.847 | 94.6% | 76.6% |

Source: `experiments/sim_guided_real_selection/run_log.txt` and
`experiments/sim_vs_real_transfer_comparison/`. `METHODS_parameter_optimization.md`
§"Summary of the main methodological finding" concludes the simulator "is not an
adequate surrogate for parameter selection on this real data." The goal is to
make that conclusion obsolete.

Only one annotated real movie exists, so the simulator is also the only route to
a second, independent evaluation set. That is the deeper motivation: §8 of this
spec matters as much as §7.

---

## 2. Read these first

| what | where | why |
|---|---|---|
| **Physics to copy** | `../../BachelorThesis-FRETAlgorithms/scripts/data_analysis/data_simulation.py` | Per-emitter PSF, HMM traces, Poisson+read noise. The good parts. |
| Config shape to copy | `../../BachelorThesis-FRETAlgorithms/config/realdata.yaml` → `simulation:` | Nested, per-class `activate`/`count`/`params`. Adopt this style. |
| Consumer #1 (must not break) | `Parameter_opt/SCP_parametrizer_002.py` → `load_movies_from_manifest`, `extract_center_coordinates` | Defines the manifest + label contract. See §6. |
| Consumer #2 | `SCP/SCP_KTree_JSON.py` → `load_tif_video`, `detect_spots` | The detector that must behave the same on sim and real. |
| What is being replaced | `Simulation/*.py` | Read before deleting; `generate_dataset_bt.py` has useful driver structure. |
| Prior art on scoring | `Parameter_opt/SCP_parametrizer_002.py` → `compute_metrics` | Do not write a second scorer. Reuse this one. |
| Ground truth | `../../BachelorThesis-FRETAlgorithms/realdata/` — `vid_ch1.tiff`, `coord.npy`, `class.npy` | 501×256×512, 16-bit. `coord.npy` is `(N,2)` in **`[y, x]`**; `class.npy` is 0=stationary/target, 1=aggregate/off-target, 2=moving. |

---

## 3. Why both current simulators fail

Diagnose before rebuilding. These are measured, not guessed.

### 3a. `Simulation/smTIRF_simulation.py` — unusable, delete

| defect | code | consequence |
|---|---|---|
| Spots are **hard-edged filled circles** (`cv2.circle`, radius 3–5 px; 10 px for distractors) blurred by **one whole-frame Gaussian at σ=5.0** | `draw_clean_frame`, `BLUR_SIGMA` | Real spots are best detected at LoG σ **0.45–1.27**. Sim spots are several times too broad. Optics is also applied *after* summing emitters, so overlapping spots fuse before the PSF — physically wrong. |
| **Every spot has identical amplitude** — `np.clip(frame, 0, 1)` before blur | `draw_clean_frame` | No brightness distribution at all. Real molecules span a wide range (§4d). |
| "Pulsation" modulates the **radius**, via `sin(frame*0.2 + phase)` | `Dot.get_size` | A deterministic ~31-frame sinusoid in *size*. Real blinking is stochastic on/off in *intensity*. When size rounds to 0 the spot vanishes entirely. |
| Negative class = **bigger and always-on** (`MAX_DOT_SIZE * 2`) | `Dot.get_size` | Inverted: real off-target spots are **dimmer** and **shorter-lived** (§4d, §4e). |
| Flat background | `apply_noise` | Real movie needs a σ=8 Gaussian high-pass to be tractable at all. |
| No photobleaching; no minimum spot separation; 512×512 not 256×512 | — | See §4c, §4e. |

### 3b. BachelorThesis `data_simulation.py` — good physics, wrong calibration

**Keep:** per-emitter sub-pixel Gaussian PSF with volume normalisation
(`render_frames.add_particles`); two-state HMM intensity traces
(`generate_stationary_intensities_hmm`); aggregates as a sum of N independent
fluorophore traces (`generate_aggregate_intensities`); Poisson-then-read-noise
order; minimum-separation rejection sampling (`generate_valid_simulation_tracks`,
`min_distance_stat = 7`, `min_distance_agg = 4`); RNG seeding throughout.

**Fix — every one of these is a measured mismatch:**

| # | mismatch | sim | real |
|---|---|---|---|
| 1 | PSF width | `gauss_sigma = 1.5`; detector prefers LoG σ 1.29–2.29 | detector prefers LoG σ **0.45–1.27** |
| 2 | Aggregate brightness | **brighter** than targets (sum of 3–10 fluorophores) | **dimmer**: median 18.9 vs 60.6 (bg-subtracted) |
| 3 | Persistence | both classes span the movie — target median **241**/300 frames, aggregate **269**/300 | molecule **134**/501, aggregate **41**/501 |
| 4 | Intensity CV | separates classes: target 0.194 vs aggregate 0.123 | **does not separate**: 0.305 vs 0.258, AUC 0.580 |
| 5 | Background | flat `background_level` | broad, uneven; requires σ=8 high-pass |
| 6 | Usable threshold | optimum ≈ **0.05** | tractable only ≥ **0.25** |
| 7 | Photobleaching | none | present (drives #3) |
| 8 | Class ratio | 90 : 45 : 25 (target : agg : moving) | 243 : 91 : **0** |

Mismatch #6 is the transfer killer and #1/#5 cause it: `SCP.detect_spots`
min–max normalises **each frame**, so `threshold` is a fraction of that frame's
brightest pixel. Different background structure and PSF width ⇒ different
normalisation ⇒ thresholds that cannot be compared. Fixing #1 and #5 should
largely fix #6; §7-T2 verifies it directly.

---

## 4. Calibration targets (measured on the real movie)

These are the numbers to hit. Every one is reproducible — the recipe is given so
the agent can re-derive rather than trust this document.

### 4a. Geometry and format
- 501 frames, **256 (H) × 512 (W)**, 16-bit, single channel.
- Output `(T, H, W)`, readable by `SCP.load_tif_video`.
- 334 annotated spots: **243 target (class 0), 91 off-target (class 1), 0 moving (class 2)**.
- Density **2.55 spots per 1000 px**; ratio target:off-target ≈ **2.67 : 1**.
- **Do not emit a moving class by default** — the real movie has none. Keep the
  code path, `activate: false`.

### 4b. Signal-to-noise
Median SNR ≈ **11** in this channel (`METHODS_parameter_optimization.md` §Datasets).
Re-derive with the peak-signal-over-noise definition already documented in
`Simulation/smTIRF_simulation.py:calculate_snr`; do not silently change the definition.

### 4c. Spot spacing — nearest-neighbour distance
Between annotated **targets** (px): min **3.61**, p1 5.00, p2 5.39, p5 6.32,
p10 7.07, p25 9.22, **median 12.04**. Pairs closer than 4 px: 2; than 6 px: 12;
than 8 px: 33. Over *all* annotations: min 3.61, p1 4.24, p5 5.83.

⚠️ BT's `min_distance_stat = 7` is **wrong** — it forbids spacings that occur in
the real data (12 target pairs sit under 6 px). Set the floor at **3.5 px** and
match the whole distribution, not just its floor.

> Recipe: `coord.npy` → `(x, y)`, split by `class.npy`, `scipy.spatial.cKDTree`,
> `query(k=2)`, take column 1.

### 4d. Per-spot photometry
Aperture radius 2 px, local background = median of the 4–7 px annulus, sampled
every 5th frame (101 frames):

| | target (class 0) | off-target (class 1) | separability |
|---|---|---|---|
| median bg-subtracted intensity | **60.6** | **18.9** | AUC 0.909 |
| CV of that trace | 0.791 | 1.596 | AUC 0.748 |
| raw 1-px value (no bg subtraction) | 220.5 | 164.9 | AUC 0.899 |
| CV of raw 1-px trace | 0.305 | 0.258 | AUC 0.580 |

Match the **medians and the spreads**, and match the **direction** — off-target
spots are *fainter*. Reproduce distributions, not just medians; report quartiles.

> ⚠️ The mechanism is unknown. "Aggregate" is whatever the human annotator marked
> in `class.npy`; there is no documentation of the criterion, and a physical
> aggregate would normally be *brighter*. Treat class 1 as an empirical
> population to reproduce, and **do not** encode "aggregate = N summed
> fluorophores" as BT does — that model predicts the wrong sign.

> Recipe: aperture photometry directly at the annotated coordinates, no detector
> involved. AUC = Mann–Whitney; report `max(auc, 1-auc)`.

### 4e. Detector-mediated track duration
Run SCP with the tuned real-movie parameters (§4f) and bin the resulting events
by which annotation class they land on (3 px box):

| | median duration | IQR |
|---|---|---|
| on a target | **134** frames | 77–241 |
| on an off-target | **41** | 25–84 |
| on nothing annotated | **25** | 18–34 |

**This is a detector-mediated quantity, not a ground-truth on-time.** The target
is that SCP *run identically* yields this distribution on simulated movies.
Do not implement it as "molecules are on for 134 frames" — get there through
photobleaching and blinking, then measure the same way and compare.

Scale by movie length if you generate movies shorter than 501 frames
(134/501 ≈ 27% of the movie for targets, ≈ 8% for off-target).

### 4f. Reference detector configuration
Use this everywhere a "the detector" is invoked for calibration or acceptance.
Do not tune it as part of this work.

```
min_sigma 0.4515457399879006   max_sigma 1.2689848283173364
threshold 0.32336429282837625  circularity_threshold 0.20209149318019778
max_displacement 4.9934159843646295
min_pulsation_frames 20        intensity_variation_threshold 0.07179385564931576
use_bandpass true              bandpass_bg_sigma 8.0
event_nms_radius 3.0
```

Held-out performance on the real movie: R 94.6% / P 76.6% / F1 0.847.
Note `min_pulsation_frames` is a **minimum track length**, not a pulsation test
(see `SCP._analyze_tracks`).

### 4g. Background structure — MEASURE THIS, it is not yet quantified
The one calibration target this spec cannot hand over. The real movie needs a
σ=8 Gaussian high-pass; the simulators have flat backgrounds. Before building,
measure and record:

1. `bg = scipy.ndimage.gaussian_filter(frame, 8.0)` per sampled frame.
2. Report `bg` mean, std, min, max, and `(max-min)/mean` across the field.
3. Its spatial autocorrelation length, and whether it is static or drifts over time.
4. The same statistics for a BT-sim frame, for contrast.

Then reproduce it — a low-spatial-frequency field (a few broad Gaussians plus a
gradient is usually enough). Put the measured numbers in the module docstring so
the next person does not have to re-derive them.

---

## 5. Design requirements

**R1 — Per-emitter PSF.** Each emitter gets its own sub-pixel, volume-normalised
Gaussian added to the frame. Never blur the summed frame. Port
`render_frames.add_particles`; make PSF σ a config parameter and calibrate it
against §7-T1, expecting a value well below BT's 1.5.

**R2 — Intensity from photophysics, not geometry.** Amplitude varies per frame;
spot *size* never does. Each emitter needs: a per-molecule brightness draw
(→ §4d medians and spreads), stochastic blinking with on- and off-time
distributions, and photobleaching to a permanent dark state (→ §4e). Off-times
must sometimes exceed the tracker's 3-frame gap, since that is what fragments
tracks and produces the observed duration distribution.

**R3 — Two annotated classes, empirically defined.** Target and off-target,
distinguished by the §4d statistics: off-target **fainter**, noisier in relative
terms, shorter-lived. Do not model off-target as summed fluorophores. Keep the
moving class implemented but `activate: false`.

**R4 — Structured background.** Per §4g, plus Poisson-then-read-noise in that
order (copy BT). Preserve the documented SNR definition.

**R5 — Placement.** Rejection sampling with a **3.5 px** floor, tuned to
reproduce the §4c distribution. Include an edge margin; record it, since spots
near the edge affect the background annulus (see `_aperture_intensity`).

**R6 — Config-driven, seeded, reproducible.** YAML in the style of
`realdata.yaml`'s `simulation:` block — no module-level constants as the
interface (the current `smTIRF_simulation.py` fails this). One seed reproduces a
movie *and* its ground truth exactly; distinct seeds give independent movies.
Seed every RNG and pass generators explicitly; never rely on global state.
Write the resolved config into the manifest `meta`.

**R7 — Emit ground truth in both formats.** See §6.

**R8 — Performance.** A 501×256×512 movie should generate in well under a
minute. Vectorise over emitters per frame; do not loop pixels in Python.

---

## 6. Output contract — do not break these consumers

### 6a. Label JSON (read by `extract_center_coordinates`)
Two files per movie, positives and negatives:
```json
[ {"center": {"x": 243.39, "y": 60.93}}, ... ]
```
`(x, y)` — x is the column. Sub-pixel floats. The reader ignores extra keys, so
per-spot metadata may be added alongside `center` (useful for §8) but `center`
must be present and correctly named.

### 6b. Manifest (read by `load_movies_from_manifest`)
```json
{
  "meta": { "...": "free-form provenance; record the full resolved config here" },
  "train": [
    { "seed": 0,
      "tif": "<path>/rep0.tif",
      "positive": "<path>/rep0_positive.json",
      "negative": "<path>/rep0_negative.json",
      "n_positive": 243, "n_negative": 91 }
  ],
  "val": [ "...same shape..." ]
}
```
`tif`, `positive`, `negative` are required per record; the reader falls back to
`<manifest_dir>/<split>/<basename>` if a path has moved, so relative layout must
stay stable. Movie stem becomes the movie name — keep stems unique.

### 6c. BachelorThesis `.npy` format (second, parallel output)
`coord.npy` `(N,2)` **`[y, x]`**, `class.npy` `(N,)` with 0=target, 1=off-target,
2=moving. Enables `matching_and_benchmarking.py` and lets simulated movies feed
the benchmark pipeline as if they were `realdata/`. Note the axis-order flip
against §6a — get it right and unit-test it.

### 6d. TIFF
`(T, H, W)`, single channel, loadable by `SCP.load_tif_video`. Do **not** write
`(T, C, H, W)` — that path is a known trap (see `scp_wrapper.py:load_frames`).
Match the real movie's dtype/scale (16-bit) so intensity thresholds are comparable.

---

## 7. Acceptance criteria

Quantitative gates. Report every one as a pass/fail table; a build that cannot
report this is not finished.

**T1 — Spot scale.** Sweep SCP's `min_sigma`/`max_sigma` on simulated movies and
on the real movie; the detector's preferred window must agree. Real prefers
**0.45–1.27**. Fail if sim's optimum sits outside ±0.3 of each bound.

**T2 — Threshold transfer (the headline test).** Blob-count-vs-threshold curves
(§4f sigmas, bandpass on) must agree in **density per 1000 px** within **±25%**
at thresholds 0.25, 0.32, 0.40, 0.50. The real movie's density at threshold 0.05
with σ 1.29–2.29 is ~1.45–1.52 per 1000 px; BT sim gives ~1.57–1.69 there but
diverges completely in the usable range. Reuse the density helper in
`experiments/sim_vs_real_transfer_comparison/run_transfer.py:_mean_density`.

**T3 — Photometry.** Reproduce §4d medians within **±25%**, with the correct
sign (off-target fainter) and overlapping IQRs. AUC of bg-subtracted brightness
must land in **0.85–0.95**.

**T4 — Track duration.** Run §4f's config on sim; per-class median durations
within **±30%** of §4e after scaling for movie length, and the ordering
target > off-target > spurious must hold.

**T5 — Spacing.** NN distribution per §4c: median within ±2 px, minimum ≥ 3.5 px,
and a comparable count of sub-6 px pairs. A two-sample KS test against the real
distribution should not reject at α=0.05.

**T6 — End-to-end transfer (the real one).** Run the existing protocol
unchanged: tune on the new simulated movies, select on the real validation band,
report the real **test** band. Compare against the two rows in §1. Target: real
test F1 **≥ 0.70**, versus 0.256 for the current simulator and 0.847 for direct
tuning. Use `Parameter_opt/SCP_parametrizer_002.py` with
`METHOD = "sim_guided_real"`; change **nothing** in the scorer or the bands.

**T7 — Reproducibility.** Same seed ⇒ byte-identical TIFF and identical labels.
Different seeds ⇒ different geometry *and* noise. Assert this in a test.

**T8 — No regression in the consumers.** `load_movies_from_manifest` and
`extract_center_coordinates` load the new dataset without modification;
`SCP.load_tif_video` reads the TIFF to shape `(T, H, W)`.

---

## 8. Deliverable: the validation harness

Ship this **with** the simulator, not after. It is what makes §7 checkable and
what makes the next calibration round cheap.

`Simulation/validate_against_real.py` — takes a simulated dataset and the real
movie, runs T1–T5 and T7, prints the pass/fail table, and writes side-by-side
distribution plots (NN distance, per-spot brightness, CV, track duration,
blob-count-vs-threshold). It must read its targets from a machine-readable
`real_movie_reference.json` that a companion script derives from
`realdata/` — so re-deriving beats trusting §4, and a new real movie can be
dropped in.

Also produce a short `CALIBRATION.md` recording measured-vs-target for each
number, which knobs moved which metric, and what remains unmatched. Honest gaps
are worth more than silent ones.

---

## 9. Traps

1. **Axis order.** §6a is `(x, y)`; §6c is `[y, x]`; SCP internals are `(x, y)`;
   numpy indexing is `[row, col]` = `[y, x]`. Every historical bug in this
   codebase lives here. Unit-test with a deliberately asymmetric frame.
2. **Non-ASCII in stdout on Windows.** SCP prints a `→`; captured pipes default
   to cp1252 and raise `UnicodeEncodeError` mid-run. Set `PYTHONUTF8=1` /
   `PYTHONIOENCODING=utf-8`, as `run_detection` and `scp_wrapper.py` both do.
3. **Per-frame min–max normalisation.** `SCP.detect_spots` normalises each frame
   before `blob_log`, so one bright outlier rescales everything and shifts the
   effective threshold. This is exactly why thresholds do not transfer. Always
   compare *density*, never raw blob counts, across differently sized frames.
4. **Do not write a second scorer.** Reuse `compute_metrics` at
   `MATCH_MARGIN = 3`. Every number on record comes from it.
5. **Do not "fix" the detector here.** Two known issues are deliberately out of
   scope: legacy intensity photometry (`intensity_aperture = 0` truncates the
   aperture to 0–1 px) and the CV test's direction. Changing them invalidates
   §4f. If the simulator seems to need them changed, say so — don't do it.
6. **Aggregate semantics are undocumented.** Match the measured statistics; do
   not invent a mechanism (§4d).
7. **Duration is not an on-time.** §4e is detector-mediated. Do not hard-code it.
8. **`min_pulsation_frames` is a track-length filter**, despite the name.
9. **Overlapping emitters.** Real spots come within 3.61 px. Adding
   volume-normalised PSFs handles this correctly; clipping the frame to [0,1]
   (as the old simulator does) does not.
10. **Don't over-fit to one movie.** N=1 real recording. Every target here could
    be a property of this sample rather than of smTIRF. Keep the config general
    and note in `CALIBRATION.md` which knobs were set from this single movie.

---

## 10. Suggested layout

```
Simulation/
  simulate.py                  # core: emitters, PSF, photophysics, noise, render
  photophysics.py              # blinking + bleaching + brightness models (R2)
  background.py                # structured background (R4, §4g)
  placement.py                 # rejection sampling (R5, §4c)
  generate_dataset.py          # driver -> movies + labels + manifest (§6)
  config/
    real_calibrated.yaml       # default profile, calibrated per §4
    real_movie_reference.json  # machine-readable targets, derived from realdata/
  measure_real_movie.py        # regenerates the reference file (§4g, §8)
  validate_against_real.py     # T1-T5, T7 pass/fail table + plots (§8)
  tests/                       # axis order, seeding (T7), contract (T8)
  CALIBRATION.md               # measured vs target, what moved what
  requirements.txt
```

Delete `smTIRF_simulation.py` and `generate_dataset_bt.py` only once T1–T6 pass.
Until then keep them so the old numbers stay reproducible. `generate_dataset_bt.py`
stays useful indefinitely as the BT-simulator baseline for comparison.

## 11. Order of work

1. `measure_real_movie.py` + `real_movie_reference.json` — reproduce §4 and fill §4g. **Nothing else starts until this passes review.**
2. `validate_against_real.py` against the *existing* BT dataset. It should reproduce §3b's mismatch table. This proves the harness works before it is used to steer.
3. Core simulator: PSF + placement + noise. Iterate on T1, T2, T5.
4. Photophysics. Iterate on T3, T4.
5. Background. Re-check T2 — it is coupled to §4g.
6. Dataset driver + contract tests. T7, T8.
7. T6, the full transfer experiment. Expect to revisit step 4.
8. `CALIBRATION.md`.

Report after step 2 and after step 5 — those are the points where a wrong turn
is cheapest to correct.
