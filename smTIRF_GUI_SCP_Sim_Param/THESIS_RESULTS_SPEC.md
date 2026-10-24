# Spec — Thesis Results chapter

**For:** a writing agent drafting the complete Results chapter.
**Covers two halves:**
- **Part A** — parameter tuning *with* versus *without* the simulator.
- **Part B** — the resulting detector benchmarked against DeepFRET and Papylio on real data.

**Status of the numbers:** every figure quoted here is measured and on record. Do not
recompute, do not round differently, do not introduce a number that is not in §4, §6 or §7.

---

## 1. Source documents and precedence

| document | role |
|---|---|
| `METHODS_SPEC.md` | authoritative for Part A. Reconciles all earlier drafts; where anything disagrees with it, it wins. |
| `RESULTS_SECTION_SPEC.md` | authoritative for Part B. This spec absorbs it; that file remains the detail reference for the benchmark. |
| `Simulation/CALIBRATION.md` | measured-vs-target record for every simulator number. |
| `METHODS_parameter_optimization.md` | **superseded draft.** Two of its claims are wrong (see §9). Do not read it as a source. |
| `SIMULATOR_SPEC.md` | build spec. Three of its claims were contradicted by measurement. Do not quote it as fact. |

---

## 2. What the chapter must establish

Four claims, in this order. Everything else is support.

1. **A simulator calibrated against the real recording makes simulation-guided parameter
   tuning work where the previous simulator did not** — real held-out F1 0.256 → 0.7143,
   a 2.8-fold improvement, above the pre-registered 0.70 bar.
2. **It still does not beat tuning directly on real labels** (0.7143 against 0.847). The
   simulator supplies a *candidate pool*; a labelled region is still required to *select*
   among candidates. It is a screening device, not a predictor.
3. **The configuration carried forward to the benchmark is therefore the real-tuned one**,
   with no simulation in its provenance. State this explicitly — it is the chapter's hinge.
4. **On real annotated data that configuration detects substantially more target molecules
   than either published tool, and its one clear deficit is rejecting off-target spots.**

Do not write Part A as a simulator success story or Part B as a victory lap. Claim 2 and
the deficit in claim 4 are load-bearing; a reader who finishes the chapter without both is
a reader the chapter misled.

---

## 3. Structure and word budget

| § | content | words |
|---|---|---|
| R.1 | Simulator fidelity — the acceptance gates | 250–350 |
| R.2 | Parameter transfer: with simulation | 350–450 |
| R.3 | What the simulator does and does not contribute (controls) | 300–400 |
| R.4 | Parameter tuning without simulation, direct on real data | 200–300 |
| R.5 | *Bridge paragraph* — which configuration goes forward, and why | 80–120 |
| R.6 | Benchmark: target recovery | 200–250 |
| R.7 | Benchmark: off-target retention | 150–200 |
| R.8 | Benchmark: detection-to-selection trajectories | 200–250 |

**Total 1750–2350 words**, excluding tables and captions. **No limitations subsection** —
every caveat in §8 goes to the Discussion chapter.

Lead every paragraph with the finding, then the evidence, then the qualification. Never
open a paragraph with a method restatement.

---

## 4. Part A — the numbers

### 4a. Simulator acceptance gates (§R.1)

Reproduce as a table. Report T2 and T4 as **marginal failures**, not as passes and not as
clean failures.

| gate | tests | tolerance | result | headline |
|---|---|---|---|---|
| T1 | PSF / spot scale | ±0.3 px | pass | σ 1.453 vs real 1.478; FWHM-implied 1.527 vs 1.550 |
| T2 | blob density vs threshold | ±25% at 0.25/0.32/0.40/0.50 | **marginal fail** | 0.91× / 0.83× / 0.73× / 0.70× of real |
| T3 | per-spot photometry | medians ±25%, AUC 0.85–0.95 | pass | 7 of 8 medians within 10%; AUC 0.909 vs 0.909 |
| T4 | track duration | ±30%, correct ordering | **marginal fail** | target 132 vs 135 ✓; spurious 29 vs 31 ✓; off-target 33 vs 49 ✗ |
| T5 | spot spacing | median ±2 px, KS not rejected | pass | median 12.07 vs 12.04; KS p = 0.21 |
| T6 | end-to-end transfer | real held-out F1 ≥ 0.70 | pass | 0.7143 |
| T7 | reproducibility | byte-identical per seed | pass | verified |
| T8 | consumers load unmodified | — | pass | 16/16 contract tests |

**Mandatory qualification, one sentence:** T2 and T4 sit within seed noise of their
thresholds — T4 passes on a single seed and fails pooled over three; T2's threshold-0.40
ratio measured 0.78× over two seeds and 0.73× over three. Small movements in either must
not be read as signal.

**Optional before/after table** (use if space allows; it is the strongest single piece of
evidence that calibration, not physics, was the problem):

| quantity | earlier simulator | this simulator | real |
|---|---|---|---|
| off-target brightness | 66.41 | 17.29 | 18.89 |
| target brightness | 89.83 | 63.06 | 60.57 |
| off-target duration / movie length | 0.180 | 0.066 | 0.098 |
| background relative swing | 0.162 | 0.403 | 0.529 |
| background autocorrelation length | 15.41 px | 24.29 px | 29.78 px |
| blob density at threshold 0.32 | 0.397 | 0.699 | 0.848 |
| target nearest-neighbour minimum | 7.04 px | 4.45 px | 3.61 px |

Gate totals: earlier simulator 1 of 5, this simulator 4 of 6. Every row moves in the right
direction.

### 4b. The transfer result and its controls (§R.2, §R.3)

The chapter's central table. All rows are real-movie held-out F1 at a 3 px matching margin,
scored by one shared scorer.

| protocol | region | real F1 | recall | precision |
|---|---|---|---|---|
| tuned on the earlier simulator, applied to real | 256–512 | 0.256 | 18.0% | 44.4% |
| 6 random configurations + real-validation selection | 256–512 | 0.4505 | 36.9% | 57.7% |
| current simulator, picking by simulated score alone | 256–512 | 0.5591 | — | — |
| **current simulator, search + real-validation selection** | 256–512 | **0.7143** | 90.1% | 59.2% |
| same, leak-free simulator calibration | 256–512 | 0.6852 | 66.7% | 70.5% |
| raw direct-on-real search output | *340–512* | 0.800 | 94.9% | 69.2% |
| **direct-on-real, final configuration** | 256–512 | **0.847** | 94.6% | 76.6% |

\*the raw search row is on columns 340–512 from a three-way split; every other row is on
columns 256–512.

**Use 0.847 as the direct-on-real headline** and draw the comparison as 0.7143 against
0.847, a gap of 0.13. Two reasons this is the right row: it is the performance of the
configuration actually carried into the benchmark (§5), and it is measured on the same
256–512 band as every simulation-guided figure, so the comparison is band-for-band.

⚠️ **One clause is mandatory wherever 0.847 appears:** it is not a raw search output. Six
of its seven parameters come from the 40-trial search; the minimum track length was raised
from 11 to 20 and 3 px duplicate suppression added by hand. Keep the 0.800 row in the table
so the reader can see the difference, and note that it sits on a narrower band.

**The random-search control is what makes 0.7143 interpretable.** Six configurations drawn
uniformly from the same bounds, same scorer, same detector switches, same bands, same
selection rule, reach 0.4505. The simulator's candidate pool is therefore worth **+0.26 F1**
at the same annotation budget and the same number of real-movie detections. Two of the six
random draws detected nothing at all; the simulation-guided search never produced a
candidate that bad.

**The leakage control.** The simulator was calibrated against statistics of the whole movie,
test band included. Recalibrating it using only annotations from columns [0, 256) costs
**0.029 F1** on the test band (0.7143 → 0.6852). Both arms are ≈2.7× the 0.256 baseline, so
the headline is not mostly leakage. **Write 0.029 as an upper bound, not an estimate** —
three effects are confounded (leakage; the validation band being genuinely harder; the
leak-free search finding a weaker region, best simulated F1 0.675 against 0.758) and the
latter two both push the leak-free arm down. The true leakage cost lies between 0 and 0.029.

### 4c. The simulator does not rank candidates (§R.3)

This constrains how the headline may be phrased and must not be omitted.

| trial | simulated F1 | real validation F1 | real test F1 |
|---|---|---|---|
| 22 | 0.6230 | **0.7524** | **0.7143** |
| 23 | 0.6124 | 0.7075 | 0.7046 |
| 24 | 0.5161 | 0.6667 | 0.6447 |
| 13 | **0.7582** | 0.4663 | 0.5591 |
| 27 | 0.6826 | 0.2844 | 0.4306 |
| 26 | 0.6798 | 0.2011 | 0.3810 |

Simulated and real scores are **anti-correlated** among the finalists, Spearman ρ ≈ −0.54.
The best configuration on simulation is the fourth best on the real movie; picking by
simulated score alone gives 0.559 instead of 0.714.

Summarise with this table:

| role a simulator might play | verdict | evidence |
|---|---|---|
| concentrates the search on a useful region | **yes**, +0.26 F1 | random draws give 0.4505 |
| ranks candidates by real performance | **no** | ρ ≈ −0.54; trusting the ranking costs 0.16 |
| removes the need for real annotations | **no** | a labelled region is still required to select |
| beats tuning directly on real labels | **no** | 0.7143 against 0.847 |

The supported claim, in these words or close to them: *the simulator generates a candidate
pool that contains configurations which work on the real movie, where the previous one did
not.* **Never** write that the simulation predicts real performance.

### 4d. Direct tuning on real data (§R.4)

40 TPE trials, spatial cross-validation on one movie: train [0, 170), validation
[170, 340), test [340, 512) px, with the track-length ceiling at 15. The search returned a
minimum track length of 11 at train F1 0.839, validation 0.806, test 0.800 (recall 94.9%,
precision 69.2%). The three band scores agreeing closely is the evidence that the
configuration is not tuned to one region.

Raising that floor from 11 to 20 and adding 3 px duplicate suppression gives the final
configuration: **F1 0.847**, recall 94.6%, precision 76.6% on columns 256–512. This is the
number to lead with — it is the configuration that goes into the benchmark.

⚠️ **State that the floor and the suppression radius were set by hand**, not returned by
the search. Two supporting facts, either of which may be used to show the choice was not
arbitrary: the floor is a strong discriminator on this movie (median track length 134
frames for annotated molecules against 41 for off-target spots and 25 for unannotated false
positives), and 3 px duplicate suppression is provably lossless here because the closest
annotated pair is 3.61 px apart, so no two annotated molecules can be fused. Measured, the
suppression pass removed 8 of 50 false positives with the true-positive count unchanged.

⚠️ **Do not claim a search was run with the ceiling at 60.** The parametrizer defines a
(3, 60) ceiling for the direct-on-real protocol and a fourth warm start at track length 20,
but no recorded run used them — every trial in `direct_real_spatial_crossval/` is at 15 or
below. The 60 ceiling is a provision for future searches, not a result.

---

## 5. §R.5 — the bridge paragraph

Short, explicit, and load-bearing. It must say all four of these:

1. The direct-on-real configuration outperformed every simulation-guided one, so **it** is
   what was benchmarked against the published tools.
2. Its parameters: `min_sigma` 0.4515, `max_sigma` 1.2690, `threshold` 0.3234,
   `circularity` 0.2021, `max_displacement` 4.9934, minimum track length 20,
   `intensity_variation_threshold` 0.0718, background suppression on (σ = 8.0), duplicate
   suppression 3.0 px.
3. Six of those seven came from the 40-trial search; the track-length floor and the
   duplicate-suppression radius were set by hand (§4d).
4. The consistency check: the benchmark's selection-stage F1 for SCP, 0.850, falls within
   0.003 of the 0.847 obtained during parameter selection. **Phrase this carefully** — the
   two numbers come from different scorers (Hungarian at 2 px vs greedy per-axis ±3 px) and
   different regions (whole movie vs held-out band). It is a reassuring consistency check
   between two independent evaluations, *not* a replication and not evidence of anything to
   three decimal places.

---

## 6. Part B — the benchmark numbers

Setup, for the opening sentence: three pipelines (SCP, Papylio, DeepFRET) on one
dual-channel smFRET recording — 501 frames, 2 × 256 × 512 px — against 334 manually
annotated spots (243 target, 91 off-target, 0 moving). Detections matched by globally
optimal Hungarian assignment at a 2 px Euclidean threshold, scored at two stages:
**detection** (raw candidates) and **selection** (retained after each pipeline's own filter).

### 6a. Headline — noise excluded (§R.6)

Primary framing. Justify the exclusion in one clause: pure-noise detections match nothing
in the annotation and are not the discrimination task under study.

| algorithm | targets retained | off-target retained | recall | precision | F1 |
|---|---|---|---|---|---|
| **SCP** | **224** / 243 | 43 | **92.2%** | 83.9% | **0.878** |
| Papylio | 173 / 243 | 9 | 71.2% | **95.1%** | 0.814 |
| DeepFRET | 168 / 243 | 37 | 69.1% | 82.0% | 0.750 |

### 6b. Off-target retention (§R.7)

Of the off-target spots each pipeline detected, the fraction correctly discarded:
**Papylio 59 of 68 (86.8%)**, SCP 38 of 81 (46.9%), DeepFRET 13 of 50 (26.0%). SCP both
detected the most off-target spots and retained the largest absolute number, 43.

Per-class counts, detected → retained (retention %):

| algorithm | target | off-target | pure noise |
|---|---|---|---|
| SCP | 242 → 224 (92.6%) | 81 → 43 (53.1%) | 1544 → 17 (**1.1%**) |
| Papylio | 221 → 173 (78.3%) | 68 → 9 (**13.2%**) | 16 → 6 (37.5%) |
| DeepFRET | 193 → 168 (87.0%) | 50 → 37 (74.0%) | 157 → 43 (27.4%) |

Worth one sentence: SCP's filter discards 98.9% of noise, which is why a detection
precision of 13.0% still yields a selection precision of 78.9%.

*Optional, and only if marked unambiguously as a projection rather than a measurement:* if
SCP's off-target rejection matched Papylio's 86.8%, it would retain ≈11 rather than 43,
giving precision ≈95% at unchanged recall and F1 ≈0.94.

### 6c. Detection-to-selection trajectories (§R.8)

Noise counted as false positives. Do not mix this framing with §6a inside one table.

| algorithm | stage | recall | precision | F1 |
|---|---|---|---|---|
| SCP | detection | 99.6% | 13.0% | 0.229 |
| SCP | selection | 92.2% | 78.9% | 0.850 |
| Papylio | detection | 90.9% | 72.5% | 0.807 |
| Papylio | selection | 71.2% | 92.0% | 0.803 |
| DeepFRET | detection | 79.4% | 48.2% | 0.600 |
| DeepFRET | selection | 69.1% | 67.7% | 0.684 |

The three pipelines move in genuinely different directions between stages: SCP
near-vertically (detect everything, then filter hard), Papylio the other way (trading
recall for precision), DeepFRET a shorter move in the same direction as Papylio. Different
philosophies, similar endpoints. This is the passage where the "not a simple ranking" claim
is made.

Note when writing prose: SCP's detection recall is **99.6%**, not 1.00. Rounding it to 1.00
is acceptable in a figure description, not in the text.

---

## 7. Figures and tables

### Already produced — cite these

Paths relative to `BachelorThesis-FRETAlgorithms/` unless stated.

| ref | file | use in |
|---|---|---|
| Fig. 1 | `results/detection_selection_bars.png` | §R.6, §R.7 |
| Fig. 2 | `results/recall_precision_plot.png` | §R.8 |
| Table 1 | `results/metrics_table.png` | reference table; do not retype into prose |
| Fig. S* | `Simulation/validation/real-calibrated/t1_spot_scale.png`, `t2_blob_density.png`, `t3_photometry.png`, `t4_track_duration.png`, `t5_nn_distance.png`, `background_structure.png` | §R.1 |
| Fig. S* | same six under `validation/BT-sim/` (earlier simulator) and `validation/valband/` (leak-free) | §R.1 comparison, §R.3 |

Figure 1 has **three** panels — A targets, B off-target, C pure noise — on **independent
y-scales** (≈250, ≈80, ≈1600). Say so in the caption or the reader will misread panel C.
Panel B is the single most important panel in Part B.

### Must be created — no file exists

- **A method-comparison bar chart for §4b** (real held-out F1 by protocol). The chapter
  needs it; the harness never produced it. Either produce it from the table in §4b or state
  in the draft that the comparison is presented as a table only.
- **A simulated-vs-real scatter of the six finalists (§4c).** Same situation. The
  anti-correlation is much more legible as a scatter than as the table.

### ⚠️ Do not cite

`results/plots/retention_stationary.png` and `retention_aggregate.png` are **blank 2.4 KB
placeholders** — the plotting script writes an empty figure to satisfy Snakemake's output
contract and the plots were never implemented. The retention data exists (Table 1, and the
per-class CSVs); the plots do not.

`results/plots/counts_stationary.png` and `counts_aggregate.png` duplicate panels A and B of
Figure 1. Prefer Figure 1.

---

## 8. Caveats — these belong in the Discussion, not here

**The Results section carries no limitations paragraph.** Every item below is handed to the
Discussion chapter. Do not write them into Results, and do not append a limitations
paragraph to any subsection.

Two things stay in Results regardless, because they are descriptions of what was measured
rather than concessions about it, and a number is unreadable without them:

- the scoring convention (every unmatched detection counts as a false positive);
- the region and stage each figure refers to (which band, detection or selection).

The list below is the hand-off to the Discussion author.

**On Part A:**

1. **N = 1 annotated recording.** Every estimate comes from partitioning one field of view.
   Bands share illumination, background, preparation and annotator, so spatial
   cross-validation controls for over-fitting to a *region*, not to a *recording*.
2. **Roughly a dozen simulator parameters were fitted to that one recording.** Some
   plausibly describe the preparation or the annotator rather than the physics.
3. **Two acceptance gates fail marginally** and six quantities remain unmatched, including
   one place where a photometry target was knowingly traded for a detector-behaviour target
   (off-target brightness 23.5 simulated against 18.9 real).
4. **The final configuration's floor and suppression radius were set by hand** (§4d), and
   the raw search output (0.800) sits on a narrower band than every other figure.
5. **One comparison is open:** whether 30 *random* candidates all evaluated on the real
   movie would also reach ≈0.71. It is untested and would cost five times the real-movie
   detector time. If it did, the simulator would be saving compute rather than improving
   quality.

**On Part B:**

6. **SCP was tuned on this movie; the others were not.** Its parameters come from a 40-trial
   search with spatial cross-validation on this exact recording. Papylio and DeepFRET ran
   with the pipeline config's shipped parameters. This is a real home advantage and must be
   stated where the comparison is drawn, not buried in a limitations section.
7. **The benchmark scores the whole movie, including the band SCP was trained on.** The
   tuning split is train [0, 170) / validation [170, 340) / test [340, 512); the benchmark's
   243 targets span all 512 columns. The benchmark evaluation set is therefore a superset of
   SCP's training set, and its numbers are not held-out for SCP. This is the sharper form of
   caveat 6 and it must appear.
8. **SCP received one channel; the others received both.** SCP is a single-channel detector
   and channel 1 is what it was tuned on; DeepFRET and Papylio are genuinely two-colour.
   Defensible, but state it. Measured: on summed channels SCP drops to F1 0.755, below
   Papylio.
9. **N = 1, single repeat.** No error bars, no significance tests, no "significantly
   better". The `p025`/`p975` columns in `statistics_*.csv` are degenerate — one repeat, so
   they equal the mean — and must never be presented as confidence intervals.

**Cross-cutting, state once each:**

- Every unmatched detection counts as a false positive, and the annotations are not
  exhaustive, so **reported precision is a lower bound**. This applies equally to every
  protocol and every tool compared.
- **Detection-stage numbers are not comparable across tools.** "Detection" means something
  different in each pipeline (SCP: merged tracks pre-filter; the others: their own candidate
  stages). Compare selection stages; use detection only to describe each tool's internal
  strategy.
- **The off-target class semantics are undocumented.** Class 1 is whatever the annotator
  marked. Measured, these spots are *fainter* than targets, which is atypical for
  fluorescent aggregates. Do not assert a physical mechanism.

---

## 9. Do not write

- **Never quote `0.8276`** (the post-hoc track-length sweep). The gain is four times larger
  on the arm whose simulator saw the whole movie (+0.113) than on the leak-free arm
  (+0.028), which is what an adjustment exploiting information the strict protocol would not
  have looks like. The defensible pair is 0.7143 and 0.7129.
- **Never quote "median SNR ≈ 11".** It is not reproducible under any of four candidate
  definitions and traces to an unrecorded placeholder. If a signal-to-noise figure is
  wanted, state the definition and use 4.19 real against 4.23 simulated.
- **Never repeat the superseded conclusion** that the simulator "is not an adequate
  surrogate for parameter selection". It held for the simulator available at the time. If
  the chapter mentions it at all, present it as a corrected conclusion with both numbers.
- **The simulation-guided headline beats 0.256, not 0.847.** 0.7143 is what the simulator
  route achieves; 0.847 is what direct tuning achieves and is the thing it *falls short of*.
  Never let a sentence imply the simulator route reached 0.847.
  (Note: METHODS_SPEC §4.5.3 prefers 0.800 for this comparison on the grounds that 0.847 is
  not a pure search output. This chapter uses 0.847 instead, because it is band-matched to
  0.7143 and is the configuration actually benchmarked — with the hand-set floor disclosed
  each time, per §4d. That is a deliberate departure, not an oversight.)
- No significance claim, p-value, or "consistently / reliably outperforms".
- No citation of the blank retention PNGs (§7).
- No suggestion that the three tools were tuned equally (§8.6).
- No claim that the simulation predicts real performance (§4c).
- Do not quote the detector's built-in command-line defaults (blob scales 30–40 px) as if
  they were tuned values. They are historical and were never used.

---

## 10. Terminology and style

- **Not "model", not "trained".** Nothing here is learned. Write "parameter configuration",
  "tuned detector", "parameter search", "tuning protocol". Reserve "trained" for DeepFRET's
  own neural network where that is what is meant.
- **`min_pulsation_frames` is a minimum track length, not a pulsation criterion**, despite
  the name. If the parameter is named in the text, describe it as a minimum track duration.
  The actual variability test is `intensity_variation_threshold`.
- Call the two tuning routes **"simulation-guided"** and **"direct-on-real"** consistently.
  Avoid "with/without simulation" after the first use — it is ambiguous about whether the
  simulation supplied data or supplied candidates.
- Past tense, third person. Match the surrounding thesis on first person plural.
- Report every number to the precision given here.
- Reference figures explicitly and by panel ("Figure 1B").

---

## 11. Acceptance checklist

The draft is not finished until every line is true.

- [ ] All four claims in §2 appear, in a form a skimming reader would extract.
- [ ] The 0.256 → 0.7143 improvement is stated **with** the 0.847 direct-tuning comparison
      in the same passage.
- [ ] Every appearance of 0.847 discloses that the track-length floor and suppression
      radius were hand-set.
- [ ] No sentence claims a search was run at a track-length ceiling of 60.
- [ ] The random control (0.4505) appears; without it 0.7143 is uninterpretable.
- [ ] The leakage control (0.6852, gap 0.029) appears, described as an upper bound.
- [ ] The anti-correlation (ρ ≈ −0.54) appears and no sentence claims predictive ranking.
- [ ] The bridge paragraph (§5) names the benchmarked configuration and its provenance.
- [ ] No limitations subsection and no conceding paragraph anywhere; every §8 caveat has
      been handed to the Discussion author instead.
- [ ] Panel C's independent y-scale is flagged in the Figure 1 caption.
- [ ] No forbidden number from §9 appears anywhere, including in a caption or a footnote.
- [ ] No occurrence of "model" or "trained" referring to SCP.
- [ ] Word count within the §3 budget.
