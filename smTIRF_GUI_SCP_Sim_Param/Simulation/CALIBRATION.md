# CALIBRATION — measured vs target, and what remains unmatched

Companion to `SIMULATOR_SPEC.md`. Records what was measured, which knob moved which
metric, and — per spec §8, *"honest gaps are worth more than silent ones"* — what
still does not match.

Everything here is reproducible:

```bash
cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/Simulation
python measure_real_movie.py                                   # derive the targets
python generate_dataset.py                                     # build the movies
python validate_against_real.py --manifest dataset/manifest.json
python tests/test_contracts.py                                 # 16 contract tests
```

---

## 1. Headline result

Gate **T6** — tune on the simulator, select on the real validation band, report the
held-out real test band, with the scorer and bands untouched:

Every row is scored by the same `compute_metrics` at `MATCH_MARGIN = 3`. All are on
the held-out band **x 256–512** except where noted — `direct_real` splits the movie
three ways because it needs a training band of its own, so its search output is
reported on x 340–512 and is *not* directly comparable.

| protocol | band | real test F1 | recall | precision |
|---|---|---|---|---|
| tuned on BT sim, applied to real | 256–512 | 0.256 | 18.0% | 44.4% |
| **tuned on this simulator (gate T6 as specified)** | 256–512 | **0.7143** | 90.1% | 59.2% |
| — with the test band excluded from calibration too (§8) | 256–512 | 0.6852 | 66.7% | 70.5% |
| — T6 config + post-hoc track-length sweep (§1.2) | 256–512 | 0.8276 | 75.7% | 91.3% |
| — **same sweep, leak-free simulator** | 256–512 | **0.7129** | 64.9% | 79.1% |
| tuned directly on the real movie: hand-improved | 256–512 | 0.8468 | 94.6% | 76.6% |
| tuned directly on the real movie: *search* output | *340–512* | 0.800 | 94.9% | 69.2% |

**The number to quote is 0.7143** — gate T6 run exactly as specified. The 0.8276 in
row four does not survive the leak-free check (row five) and should not be quoted;
see §1.2.

Target was ≥ 0.70. **T6 passes**, at 2.8× the previous simulator. The stricter
leak-free variant lands at 0.685 — marginally under the bar, and 0.029 below the
headline, which is the measured **upper bound** on how much the result owes to the
simulator having been calibrated against the whole movie. See §8.

`METHODS_parameter_optimization.md` concluded the simulator *"is not an adequate
surrogate for parameter selection on this real data."* That conclusion no longer
holds for this simulator.

**Read 0.256 as the thing being beaten, not 0.847.** The 0.847 comes from tuning on
the real movie's own labels and then hand-improving the result — an oracle, and not
a search output.

What the simulator is and is not, measured (§1.1, §1.2):

| | verdict | evidence |
|---|---|---|
| concentrates the search on a useful region | **yes**, worth +0.26 F1 | random draws at the same budget give 0.4505 |
| ranks candidates by real performance | **no** | ρ ≈ −0.54; trusting it costs 0.16 |
| removes the need for real annotations | **no** | still needs a labelled band to select |
| beats tuning directly on real labels | **no** | 0.7143 against 0.800 |

Practically: on a new recording this saves the *parameter search*, not the
*annotation*. You would still need to label a band — about 180 spots here — but not
run a supervised search against those labels.

Run: `experiments/real_calibrated_sim_transfer/run_transfer_t6.py --arm full`,
30 trials, 2 simulated train movies, 3.3 h.

### The simulator supplies good candidates; it does not rank them

The six configs carried from the simulated search to the real movie:

| trial | sim F1 | real VAL F1 | real TEST F1 |
|---|---|---|---|
| 22 | 0.6230 | **0.7524** | **0.7143** |
| 23 | 0.6124 | 0.7075 | 0.7046 |
| 24 | 0.5161 | 0.6667 | 0.6447 |
| 13 | **0.7582** | 0.4663 | 0.5591 |
| 27 | 0.6826 | 0.2844 | 0.4306 |
| 26 | 0.6798 | 0.2011 | 0.3810 |

Sim score and real score are **anti-correlated** among the finalists (Spearman
ρ ≈ −0.54). The best config *on the simulation* (trial 13) is the fourth-best on the
real movie, and picking by sim score alone would have given a real test F1 of 0.559
instead of 0.714.

So the "select on the real validation band" half of the protocol is doing real work,
and this result should **not** be read as "the simulation now predicts real
performance". What it supports is the weaker, and still useful, claim: the simulator
generates a candidate pool that *contains* configs which work on the real movie,
where the previous one did not. Any protocol without a real-labelled selection step
would land near 0.56.

This also bounds what a better simulator could buy. Beating the direct-tuning oracle
(0.847) would require the simulated score itself to be a trustworthy ranking signal,
and at ρ ≈ −0.54 it is not close.

## 1.1 Does the simulator actually help? The random-search control

Every configuration with a real-movie score on record was first chosen by its
*simulated* score, so the T6 result alone cannot separate two contributions: the
simulator concentrating the search, and the real-labelled selection step picking the
winner. The control that separates them draws configurations at random from the same
bounds — no simulator — and runs the identical second half of the protocol.

`experiments/real_calibrated_sim_transfer/run_random_control.py`, 6 uniform draws
(matching `TOP_K`), same bounds, same scorer, same bands, 16 min:

| method | real labels used | real detector runs | real TEST F1 |
|---|---|---|---|
| 6 random configs + real-val selection | val band | 6 | **0.4505** |
| picking by simulated score alone | none | 1 | 0.5591 |
| **sim search + real-val selection (T6)** | val band | 6 | **0.7143** |
| direct-on-real search | train + val bands | 100+ | 0.800 |

**The simulator's candidate pool is worth +0.26 F1** over random draws at the same
annotation budget and the same number of real-movie detections. Two of the six random
configs detected nothing at all (F1 0.000); the simulated search never produced a
candidate that bad. The margin is far outside the ±0.05 ambiguity band that a
6-draw, single-seed estimate warrants.

So the three roles are now separated and measured:

* **concentrating the search** — the simulator does this, and it is worth 0.26;
* **ranking candidates** — it does NOT do this (ρ ≈ −0.54, §1); real labels are
  required, and trusting the simulated ranking costs 0.16;
* **replacing real labels entirely** — it does not do this either. Direct tuning on
  real annotations still wins, 0.800 against 0.7143.

One open comparison. Sim-guided screens 30 candidates on simulation (no real labels,
no real detector time) and spends 6 real detections on the survivors, so the
simulator functions as a cheap **screening** stage. Whether 30 *random* draws all
evaluated on the real movie would also reach ~0.71 is untested — it costs 5x the
real-movie detector time, which is the expensive resource. If it did, the simulator
would be saving compute rather than improving quality.

---

## 1.2 The search handicap, and why lifting it does NOT hold up

`min_pulsation_frames` (a minimum track length, despite the name) is bounded at
**(3, 15)** for `METHOD="sim_guided_real"` but **(3, 60)** for `direct_real`. The
cap exists for a reason recorded in `SCP_parametrizer_002.py`: on **BT-simulator**
movies track length was a dead dimension, because aggregates there lasted *longer*
than targets (269 vs 241 frames), so a raised ceiling would only waste trials.

This simulator reproduces the real ordering — target 132 > off-target 33 >
spurious 29 frames — so that justification no longer applies to it.

Sweeping the parameter post-hoc on the `full` arm's winning config, selecting on the
real validation band and reporting the held-out test band:

| `min_pulsation_frames` | real VAL F1 (selects) | real TEST F1 | R | P |
|---|---|---|---|---|
| 8 (chosen under the cap) | 0.7524 | 0.7143 | 90.1% | 59.2% |
| 20 | 0.7811 | 0.7656 | 88.3% | 67.6% |
| 60 | 0.8444 | 0.8251 | 82.9% | 82.1% |
| **100** ← best on VAL | **0.8594** | **0.8276** | 75.7% | 91.3% |
| 150 | 0.8103 | 0.7473 | | |

**0.7143 → 0.8276** on this arm, above the direct-on-real *search* result (0.800)
and 0.019 below the hand-tuned reference (0.8468). The VAL curve peaks at 100 rather
than at the edge of the sweep, so it is an interior optimum, not a boundary artifact.

### …but it does not survive the leak-free check

Running the **identical** sweep on the `valband` arm — same procedure, same
val-band selection, on a simulator that never saw the test band's annotations:

| arm | capped search | + track-length sweep | gain | selected mpf |
|---|---|---|---|---|
| `full` (simulator saw whole movie) | 0.7143 | 0.8276 | **+0.113** | 100 |
| `valband` (leak-free) | 0.6852 | 0.7129 | **+0.028** | 20 |

The gain is four times larger on the arm whose simulator was calibrated against the
whole movie, and the gap between arms widens from 0.029 to **0.115**. The leak-free
sweep also lands on a far more conservative `min_pulsation_frames` (20, inside
`direct_real`'s sanctioned range) rather than the extrapolated 100.

That pattern is what it looks like when a post-hoc adjustment is exploiting
information the strict protocol would not have. **0.8276 should not be quoted.**
The defensible figures are 0.7143 (T6 as specified) and 0.7129 (leak-free, with the
sweep) — which agree closely, and neither reaches the direct-tuning oracle.

The confound of §8 still applies and cuts the other way: the `valband` arm also had a
weaker search and a harder dataset, so some of the 0.115 is not leakage. But the
asymmetry is large enough that the honest reading is "the cap-lifting gain is not
established", not "the cap-lifting gain is real and worth 0.11".

The sweep is done by filtering saved detections on their exported `duration`, which
is **exactly** equivalent to re-running the detector: verified against a real run at
`min_pulsation_frames=100`, which produced an identical 209-event set. (Equivalent
because SCP applies the length filter before classification, and its event NMS keeps
the longest-lived track in a cluster, so a shorter track can never displace a longer
one.)

Three further caveats on that 0.8276, independent of the leak-free result above:

1. It adjusts **one** parameter post-hoc on a config whose other six were optimised
   under the cap. A full re-search with the bound raised may land elsewhere.
2. `min_pulsation_frames = 100` is far outside even `direct_real`'s sanctioned
   (3, 60). Demanding a 100-frame track in a 501-frame movie is aggressive and may
   be a property of this recording.
3. It reaches that F1 by the **opposite** route to the reference config — 75.7% R /
   91.3% P against 94.6% R / 76.6% P. Same score, very different detector.

Note also that 0.847 is itself not a search output: `direct_real`'s search returned
0.800 with `min_pulsation_frames = 11`, and 0.847 came from manually raising it to
20 and adding 3 px event NMS. The like-for-like search-vs-search comparison is
therefore 0.7143 (capped) or 0.8276 (uncapped) against 0.800.

---

## 2. Acceptance gates

| gate | result | headline number |
|---|---|---|
| T1 spot scale | **PASS** | PSF σ 1.453 vs real 1.478; FWHM-implied 1.527 vs 1.550 |
| T2 threshold transfer | **marginal fail** | 0.91× / 0.83× / 0.73× / 0.70× of real density (needs ≥0.75) |
| T3 photometry | **PASS** | 7 of 8 medians within 10%; worst is off-target CV at +22.3% (tolerance ±25%); brightness AUC 0.909 vs 0.909 |
| T4 track duration | **marginal fail** | target 132 vs 135 ✓, spurious 29 vs 31 ✓, off-target 33 vs 49 ✗ |
| T5 spacing | **PASS** | NN median 12.07 vs 12.04; KS p = 0.21 |
| T6 transfer | **PASS** | real test F1 0.7143 (target ≥ 0.70) |
| T7 reproducibility | **PASS** | byte-identical TIFF and labels per seed |
| T8 consumers | **PASS** | 16/16 contract tests |

T2 and T4 both sit **within seed noise of their thresholds**: T4 passes on seed 0
(target 133 / off-target 38 / spurious 32) and fails when pooled over three movies
(33 for off-target, against the 34.3 needed); T2's threshold-0.40 ratio measured
0.78× over two seeds and 0.73× over three. Tuning was stopped there because further
movement was fitting to noise. §6 explains why that noise is intrinsic.

---

## 3. Three spec claims that measurement contradicts

### 3.1 The PSF is **not** a mismatch (§3b row 1, R1) — the most consequential

§3b says the simulator's `gauss_sigma = 1.5` is wrong because the detector prefers
LoG σ 0.45–1.27 on the real movie, and R1 says to expect a calibrated value "well
below BT's 1.5".

Measured, the real spots are **σ ≈ 1.5 — the same as BT's**:

| estimator | real target | BT sim (configured 1.5) |
|---|---|---|
| stacked second moment | 1.478 | 1.424 |
| per-spot median | 1.473 | 1.423 |
| stacked Gaussian fit | 1.718 | 1.517 |
| radial-profile FWHM (fit-free) | 3.65 px → σ 1.550 | 3.54 px → σ 1.502 |

The stacked radial profiles are nearly identical (real 1.000 / 0.824 / 0.680 / 0.403
at r = 0, 1, 1.5, 2 px; BT 1.000 / 0.808 / 0.651 / 0.369).

**§3b conflated the detector's tuned LoG σ *parameter* with the physical spot
width.** They are not the same quantity: LoG σ is a filter scale interacting with the
σ=8 high-pass, the per-frame normalisation and the threshold. Building to R1's
prediction would have made the simulator measurably *worse*.

`psf_sigma` is therefore **1.55** (feeding 1.55 in reproduces a measured 1.45–1.53,
since the estimators are slightly biased low).

### 3.2 Gate T1 as written is not identifiable

§7-T1 says to sweep `min_sigma`/`max_sigma` and compare the detector's preferred
window. Measured, `blob_log` returns an **identical** 312 blobs/frame on the real
movie for *every* `max_sigma` from 0.8 to 3.0 once `min_sigma` is 0.2 — every
response comes from the smallest scale in the window, so window width is not
identifiable. Worse, gating on it **passes the BT simulator**, whose PSF §3b calls a
mismatch.

T1 now gates on measured PSF width, with the single-scale response curve kept as a
secondary check. The response curve is itself informative: it collapses above σ≈0.45
on both movies, which is why the tuned window's *lower* bound (0.45) is meaningful
while its upper bound (1.27) is inert.

### 3.3 §4b's "median SNR ≈ 11" is not reproducible

§4b says to re-derive with `smTIRF_simulation.py:calculate_snr`'s definition. That
function takes *simulator parameters* (`photon_scale`, `background_level`,
`read_noise`) and cannot be evaluated on a real movie of unknown gain, so the
instruction has no literal image-domain form. Four candidates, on the target class:

| definition | real |
|---|---|
| (peak − annulus median) / annulus std | 4.19 |
| (peak − annulus median) / √peak | 8.03 |
| peak / annulus std | 8.97 |
| (disc mean − annulus median) / annulus std | 1.56 |

None gives 11. That number traces to a bracketed placeholder in
`METHODS_parameter_optimization.md` whose definition was never recorded. **No §7 gate
depends on SNR**, so this is documented rather than resolved. The same estimator is
applied to both sides, and agrees: simulated 4.23 vs real 4.19.

---

## 4. Two things the spec did not anticipate

### 4.1 The real movie contains unannotated emitters

`coord.npy` holds exactly 334 spots, but the movie contains more. In regions ≥6 px
from every annotation, with noise matched to 0.4%:

| | real | simulated (without them) |
|---|---|---|
| high-passed p99.9 in spot-free regions | 85.6 ADU | 57.2 ADU |
| detector blobs/frame unexplained by any annotation | 26.5 | 6.4 |

Those are localised bright features at unmarked positions — the dim non-specific
emitters any real preparation has. **Omitting them was the entire T2 gap**: the
simulated blob density sat at 0.53× the real value at every threshold.

They are now simulated (`unannotated`, count 500) and appear in **no** ground-truth
output — not the label JSONs, not `coord.npy`, not `class.npy`. A detection on one is
scored as a false positive by `compute_metrics`, exactly as on the real movie. This
also fixed the false-positive population: the reference detector on simulated movies
now scores R 88.3% / P 79.6% / F1 0.837, against the real R 94.6% / P 76.6% /
F1 0.847.

### 4.2 The photophysics is slow switching, not fast blinking

R2 suggests off-times that "sometimes exceed the tracker's 3-frame gap". Fast
blinking (τ_on ≈ 8 frames) cannot reproduce §4e at all: SCP bridges ≤3-frame gaps and
then requires 20 detections, so a track is essentially one contiguous on-period seen
through an imperfect detector. With τ_on = 8 the off-target class produced **one**
surviving event per movie against the real 46, across six configurations.

Moving to on/off periods of order 100 frames:

| | before (fast) | after (slow) | real |
|---|---|---|---|
| target track median | 264 frames | 132 | 135 |
| spurious track median | none | 29 | 31 |
| off-target events per movie | 1 | 35 | 46 |

---

## 5. Which knob moved what

| knob | value | fitted against | effect |
|---|---|---|---|
| `psf_sigma` | 1.55 | measured spot width | T1 |
| `gain_adu_per_photon` | 5.0 | frame-difference noise 27.4 ADU at mean 147 | noise realism |
| `read_noise_adu` | 3.8 | residual after `gain × mean` | noise realism |
| `background.*` | see `background.py` | σ=8 field statistics | T2, background |
| `min_distance` / `soft_distance` / `soft_gamma` | 3.5 / 6.0 / 3.0 | NN distribution | T5 |
| `peak_amplitude_median` | 153 / 110 | §4d brightness (linear, exact inversion) | T3 |
| `peak_amplitude_sigma_log` | 0.36 / 0.047 | §4d IQR ratio | T3 AUC |
| `tau_on` / `tau_off` | 170/115, 120/265 | §4d CV (duty) and §4e duration | T3, T4 |
| `bleach_shape` | 3.0 (inert) | §4d IQR ratio | T3 |
| `on_amplitude_cv` | 0.25 | §4d CV, and T2 (see §6) | T2 ↔ T4 |
| `unannotated.count` | 500 | blob density | T2, precision |
| `off_target.psf_sigma` | 1.63 | measured (1.559 vs 1.478) | T4 |

Two non-obvious findings worth keeping:

**The trace is reduced by its MEAN, not its median.** §4d aggregates each spot's own
trace and then takes the median across spots. Using the median of the trace gives
target 51.1 / off-target 7.2 and AUC 0.786; the mean gives 60.6 / 18.9 and AUC 0.909,
matching the recorded values to three digits. Calibrating against the wrong one
misses the target CV by 27%.

**The MEDIAN emitter's duty is what §4d reports, not the average one.** With
bleaching, a handful of early-bleached emitters drag the mean duty far below the
median (0.570 vs 0.679 at one point). A first calibration pass targeted the mean and
would not converge.

---

## 6. The T2 ↔ T4 tension, and why both gates are noisy

`SCP.detect_spots` min–max normalises **every frame** before blob detection (spec §9,
trap 3). The detection threshold is therefore a fraction of that frame's brightest
high-passed pixel, and anything producing bright outliers inflates the divisor and
pushes ordinary spots *below* threshold.

Measured, `on_amplitude_cv` moves T2 in the **opposite** direction to intuition:

| `on_amplitude_cv` | blob density at threshold 0.32 |
|---|---|
| 0.10 | 0.571 |
| 0.25 | 0.461 |
| 0.40 | 0.308 |

But it cannot simply be minimised: SCP also keeps only tracks whose intensity CV
clears `intensity_variation_threshold`, so at 0.10 the off-target class produced **no
surviving events at all** and T4 collapsed. 0.25 is the compromise, and no setting
was found where T2 and T4 both pass.

The same mechanism makes both gates **intrinsically noisy**: a single bright outlier
pixel rescales the whole frame. Measured, changing the unannotated count from 200 to
250 moved the density at threshold 0.32 from 0.733 to 0.559 — non-monotonically. Do
not read small T2/T4 movements as signal.

---

## 7. What still does not match

| # | quantity | sim | real | note |
|---|---|---|---|---|
| 1 | blob density at threshold 0.40 / 0.50 | 0.73× / 0.70× | 1.00× | gate needs ≥0.75×; the high-threshold end depends on the brightest spot-frames, and everything that fattens that tail also inflates the normalisation divisor. No reachable setting was found. |
| 2 | off-target track duration | 33 frames | 49 | passes on some seeds (38), fails pooled. Lengthening on-periods did **not** help (tested 120/265 → 170/375: 24 → 22 frames); what helped was brightness and width, and brightness is already at the edge of T3's tolerance. |
| 3 | off-target brightness spread | IQR ratio 2.50 | 1.97 | driven by per-emitter duty variance under slow switching, not by amplitude spread — `peak_amplitude_sigma_log` is already ≈0 and cannot narrow it further. The same cause puts the off-target CV at +22.3%, the widest of T3's eight medians (tolerance ±25%). T3 still passes: IQRs overlap and the AUC lands. |
| 4 | off-target brightness | 23.5 | 18.9 (1.24×) | a **deliberate trade**, the one place a photometry target was knowingly given up for a detector-behaviour target. At the CV-implied 92 ADU the class was too faint to form tracks (10 events, 96% precision); at 110 ADU it gives 35 events and 78% precision against the real 76.6%. |
| 5 | NN minimum | ≈4.5 px | 3.61 px | the soft-core ramp makes sub-4 px spacings very rare. The real movie has 2 of 243 targets under 4 px (0.8%); the simulator has none. T5's floor (≥3.5 px) and KS test both pass. |
| 6 | background relative swing | 0.403 | 0.529 | and autocorrelation length 24.3 px vs 29.8. The Gaussian-random-field model reproduces the scale but under-reproduces the extremes. |
| 7 | SNR | — | — | see §3.3: no agreed definition. Both sides measured identically agree (4.23 vs 4.19). |

---

## 8. Over-fitting risk — read this before trusting a number

**N = 1 real recording**, and roughly a dozen simulator parameters were fitted to it.
Every target in `config/real_movie_reference.json` could be a property of this
sample, this microscope and this annotator rather than of smTIRF. Specifically:

- The **unannotated emitter population** (§4.1) is a property of this preparation.
- The **soft-core placement ramp** partly describes how a human annotates crowded
  regions, not how molecules land.
- The **slow-switching timescales** are entangled with SCP's 3-frame gap and 20-frame
  minimum; a different tracker would imply different values.
- The **off-target class** is whatever the annotator marked (spec trap 6). The model
  here — fainter, more intermittent, wider — reproduces the statistics and makes no
  claim about mechanism.

### The T6 leakage caveat

The `full` arm's simulator was calibrated against statistics of the **whole** real
movie, test band included. Detector parameters are still selected on the validation
band only, so it is not circular in the usual sense — but its test number is not
fully held out.

A second arm exists for exactly this: `--arm valband` calibrates the simulator
against `real_movie_reference_valband.json`, derived from the annotations in
x ∈ [0, 256) only, so the test band influenced nothing. Pixel-derived values
(background, noise, blob density) still come from the whole frame, deliberately: a
new recording would supply unlabelled pixels too, so those are not a leak.

**The two bands genuinely differ** — val-band target CV 0.583 against the whole
movie's 0.791, off-target brightness 16.0 against 18.9, spot density 2.76 against
2.55 per 1000 px — so this is a real recalibration, not a formality. The gap between
the two arms is the honest measure of how much of T6 is transfer rather than leakage.

### Result: the leak is small

| arm | simulator calibrated on | real VAL F1 | real TEST F1 | recall | precision |
|---|---|---|---|---|---|
| `full` | whole movie (spec T6 as written) | 0.7524 | **0.7143** | 90.1% | 59.2% |
| `valband` | validation band only — **no test-band labels** | 0.5377 | **0.6852** | 66.7% | 70.5% |

**Test-band gap: 0.029 F1** (4% relative). Both arms are ~2.7× the 0.256 baseline.

So the headline result is **not** mostly leakage. Removing every trace of test-band
annotation from the simulator's calibration costs about 0.03 F1 — the leak-free
number lands at 0.685, marginally under the ≥0.70 bar rather than collapsing toward
the 0.256 baseline.

**The comparison is sound.** Both arms use the same TPE seed, so their early random
trials evaluate identical parameter sets, and the real-movie evaluation is
deterministic. Trial 13 appears in both candidate pools and scores identically
(test 0.5591, val 0.4663) — a direct check that the two runs are measuring the same
thing. Later trial numbers diverge because TPE's proposals depend on the simulated
scores, which is expected.

**0.029 is an upper bound on leakage, not an estimate of it.** Three effects are
confounded and cannot be separated with one recording:

1. **Leakage** — the thing being measured.
2. **The val band is not representative of the whole movie.** It has a higher spot
   density (2.76 vs 2.55 per 1000 px) and a higher duty cycle (target CV 0.583 vs
   0.791), so the `valband` simulator is a different — and harder — simulator, not
   merely a less-informed one. On identical parameter sets from the shared random
   trials it scored consistently lower (0.222 vs 0.245, 0.062 vs 0.101, 0.030 vs
   0.040), confirming the dataset itself is harder.
3. **Search variance.** The `valband` search never found as strong a region: best
   simulated F1 0.6750 against the `full` arm's 0.7582, so its top-6 pool was drawn
   from weaker candidates.

Because (2) and (3) both push the `valband` arm *down*, the true leakage cost is
somewhere between 0 and 0.029. Separating them would need a second annotated
recording — which is, per §1, the deeper reason this simulator exists.

Worth noting the two arms reach similar F1 by different routes: `full` is
recall-heavy (90.1% / 59.2%), `valband` precision-heavy (66.7% / 70.5%). Neither
matches the direct-tuning oracle's balance (94.6% / 76.6%), and precision remains
the weak side of sim-guided tuning.

Reproduce with:
```bash
python run_transfer_t6.py --arm full     --trials 30 --top-k 6 --train-movies 2
python run_transfer_t6.py --arm valband  --trials 30 --top-k 6 --train-movies 2
```

---

## 9. Reproduction status of the spec's own numbers

| section | status |
|---|---|
| §4a geometry, counts, density | **exact** (501×256×512, 243/91/0, 2.548, 2.67:1) |
| §4c all NN percentiles and close-pair counts | **exact** (3.61, 5.00, 5.39, 6.32, 7.07, 9.22, 12.04; 2/12/33) |
| §4d all twelve photometry numbers | **exact** (60.568, 18.892, 0.909; 0.791, 1.596, 0.748; 220.483, 164.875, 0.899; 0.305, 0.258, 0.580) |
| §4e track durations | target 135 vs 134; off-target 49 vs 41; spurious 31 vs 25 |
| §4f held-out detector performance | **exact** (R 94.6% / P 76.6% / F1 0.847) |
| §7-T2 historical density at threshold 0.05 | 1.516, inside the quoted 1.45–1.52 |
| §4b SNR | **not reproducible** — see §3.3 |
| §4g background | **newly measured** — the spec left it open |
| §3b mismatch table | reproduced on the BT dataset — see §10 |

Reproducing §4d required recovering two undocumented recipe details by experiment:
edge windows are **clipped, not dropped** (dropping the 22 edge spots moves target CV
from 0.791 to 0.813), and "raw 1-px" means a **radius-1 aperture**, not a single
pixel (the single pixel gives 232.7 / 167.4 with CV 0.379 / 0.329 instead of 220.5 /
164.9 with CV 0.305 / 0.258).

---

## 10. The harness was validated before it was trusted

Per spec §11 step 2, `validate_against_real.py` was run on the **BT simulator** first
— before it was used to steer anything. It reproduced §3b's mismatch table, and the
same table now doubles as the before/after record:

| §3b row | quantity | BT sim | **this simulator** | real |
|---|---|---|---|---|
| 2 | off-target brightness (bg-sub) | 66.41 (3.5× too bright) | **17.29** | 18.89 |
| 2 | target brightness (bg-sub) | 89.83 | **63.06** | 60.57 |
| 3 | target duration / movie length | 0.333 | **0.263** | 0.269 |
| 3 | off-target duration / movie length | 0.180 | **0.066** | 0.098 |
| 4 | intensity CV separates classes (AUC) | 0.955 | **0.709** | 0.580 |
| 5 | background relative swing | 0.162 | **0.403** | 0.529 |
| 5 | background autocorrelation length (px) | 15.41 | **24.29** | 29.78 |
| 6 | blob density at threshold 0.32 | 0.397 | **0.699** | 0.848 |
| 8 | target : off-target ratio | 2.00 | **2.670** | 2.670 |
| 4c | target NN minimum (px) | 7.04 | **4.45** | 3.61 |

Every row moves the right way. Gate totals: BT sim **1 of 5** (T1 only, T7 not
applicable), this simulator **4 of 6**.

The NN row independently confirms §4c's warning: BT's target NN minimum is *exactly*
7.04 px — its `min_distance_stat = 7` floor — with zero pairs under 6 px where the
real movie has 12.

### The BT control is also what disproves §3b row 1

BT sim **passes T1**: its measured PSF σ is 1.424 (stacked second moment) / 1.501
(radial-profile FWHM), against the real 1.478 / 1.550 — inside the ±0.3 px tolerance
on every estimator. A simulator that §3b calls wrong on PSF width measures the same
width as the real movie, because §3b row 1 was not about width. See §3.1.

`dataset_bt/` was regenerated for this (the on-disk copy predated the current
`generate_dataset_bt.py` — 60 frames and summed channels, against §3b's 300-frame
single-channel measurements).

---

## 11. Files

| file | role |
|---|---|
| `measurements.py` | every estimator; both sides of every comparison go through it |
| `measure_real_movie.py` | derives `config/real_movie_reference.json` (`--band` for the leak-free variant) |
| `validate_against_real.py` | gates T1–T5, T7; pass/fail table, 6 plots, JSON report |
| `simulate.py` | emitters, PSF, render, noise |
| `photophysics.py` | brightness, blinking, bleaching |
| `background.py` | structured background field |
| `placement.py` | soft-core rejection sampling |
| `calibrate.py` | fits photometry knobs — **searches**, where the harness **judges** |
| `generate_dataset.py` | driver; the §6 output contract |
| `tests/test_contracts.py` | 16 tests: axis order, seeding, consumers |
| `config/real_calibrated.yaml` | the profile; every value carries its justification |
| `config/real_calibrated_valband.yaml` | leak-free variant |
| `legacy_generate_dataset_smtirf.py` | the old driver, preserved so the 0.256 baseline stays reproducible |

`smTIRF_simulation.py` and `generate_dataset_bt.py` are **kept**. Spec §10 permits
deleting the first only once T1–T6 pass; T2 and T4 have not, and
`generate_dataset_bt.py` remains useful indefinitely as the BT baseline.
