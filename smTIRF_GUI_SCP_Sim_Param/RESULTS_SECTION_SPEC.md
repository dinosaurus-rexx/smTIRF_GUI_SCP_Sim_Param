# Spec — Results section: SCP vs DeepFRET vs Papylio on real smTIRF data

**For:** a writing agent drafting the Results section.
**Status:** all numbers below are measured and verified. Do not recompute; do not
round differently; do not invent any figure not listed in §2.
**Run date:** 2026-08-02. **Pipeline:** `BachelorThesis-FRETAlgorithms`, Snakemake,
`config/realdata.yaml`.

---

## 1. What this section must establish

In one sentence: **on real annotated smTIRF data, SCP detects substantially more
target molecules than either published tool, and its remaining weakness is
rejecting off-target spots.**

Three claims, in order of importance:

1. SCP achieves the highest F1 of the three, driven by recall.
2. The three tools occupy genuinely different points on the precision/recall
   trade-off — this is not a simple ranking.
3. SCP's one clear deficit is off-target rejection, where Papylio is far better.
   State this plainly; it is the honest limitation and it motivates future work.

Do **not** write this as a victory lap. Claim 3 and the caveats in §5 are load-bearing.

---

## 2. Figures — use these, in this order

All paths relative to `BachelorThesis-FRETAlgorithms/`.

### Figure 1 (primary) — `results/detection_selection_bars.png`
Three panels, stacked bars, pale = total detected, solid = retained after each
tool's own filtering. **A: Target Interactions. B: Off-Target Interactions.
C: Pure Noise.** Panels are on independent y-scales (≈250, ≈80, ≈1600) — say so
in the caption or the reader will misread panel C.

**Panel B is the most important single panel in the section.** It shows SCP
detecting the most off-target spots (81) and retaining half (43), against
Papylio retaining 9 of 68.

### Figure 2 — `results/recall_precision_plot.png`
Precision–recall space with F1 isolines. Each tool is an arrow from its
detection stage (hollow marker) to its selection stage (filled). Colours:
DeepFRET blue circle, Papylio orange square, SCP pink triangle.

Use this to make claim 2 visually. SCP's arrow is near-vertical — from
(1.00, 0.13) up to (0.92, 0.79), i.e. detect everything then filter hard.
Papylio's runs the other way, trading recall for precision. Different
philosophies, similar endpoints.

### Figure 3 — `results/metrics_table.png`
"Table 1. Comprehensive Benchmarking Statistics." Per-class counts and retention
on the left, global Detection/Selection precision, recall and F1 on the right.
Use as the reference table; do not retype it into prose.

### Optional, only if a single-class figure is wanted
- `results/plots/counts_stationary.png` — targets alone
- `results/plots/counts_aggregate.png` — off-target alone

Both duplicate panels of Figure 1. Prefer Figure 1.

### ⚠️ Do NOT cite
`results/plots/retention_stationary.png` and `retention_aggregate.png` are
**blank 2.4 KB placeholders** — the upstream plotting script saves an empty
figure to satisfy Snakemake's output contract and never implemented them. The
retention data exists (Figure 3, and the CSVs in §3); the plots do not.

---

## 3. Source data

Per algorithm `{scp, papylio, deepfret}`, under `results/{alg}/`:
`class_benchmarks/rep1.csv`, `global_benchmarks/rep1.csv`,
`statistics_class.csv`, `statistics_global.csv`.

`statistics_*.csv` carry `p025`/`p975` columns. **They are degenerate** — one
repeat, so they equal the mean. Do not present them as confidence intervals.

---

## 4. The numbers

### 4a. Headline — noise excluded (use this as the primary framing)

Target vs off-target only, out of 243 annotated targets. Justify the exclusion
in one clause: pure-noise detections match nothing in the annotation and are not
the discrimination task under study.

| algorithm | targets retained | off-target retained | recall | precision | F1 |
|---|---|---|---|---|---|
| **SCP** | **224** / 243 | 43 | **92.2%** | 83.9% | **0.8784** |
| Papylio | 173 / 243 | 9 | 71.2% | **95.1%** | 0.8141 |
| DeepFRET | 168 / 243 | 37 | 69.1% | 82.0% | 0.7500 |

**Off-target spots correctly rejected** (of those each tool detected):
Papylio **86.8%**, SCP **46.9%**, DeepFRET **26.0%**.

### 4b. Global metrics, noise counted as false positives

Matches Figure 3. Report alongside 4a; do not mix the two framings in one table.

| algorithm | stage | recall | precision | F1 |
|---|---|---|---|---|
| SCP | detection | 99.6% | 13.0% | 0.229 |
| SCP | selection | 92.2% | 78.9% | 0.850 |
| Papylio | detection | 90.9% | 72.5% | 0.807 |
| Papylio | selection | 71.2% | 92.0% | 0.803 |
| DeepFRET | detection | 79.4% | 48.2% | 0.600 |
| DeepFRET | selection | 69.1% | 67.7% | 0.684 |

### 4c. Per-class counts (detected → retained, retention %)

| algorithm | target | off-target | pure noise |
|---|---|---|---|
| SCP | 242 → 224 (92.6%) | 81 → 43 (53.1%) | 1544 → 17 (**1.1%**) |
| Papylio | 221 → 173 (78.3%) | 68 → 9 (**13.2%**) | 16 → 6 (37.5%) |
| DeepFRET | 193 → 168 (87.0%) | 50 → 37 (74.0%) | 157 → 43 (27.4%) |

Worth a sentence: SCP's filtering stage discards 98.9% of noise, which is why a
detection precision of 13.0% still yields a selection precision of 78.9%.

### 4d. Optional quantified headroom
If off-target rejection matched Papylio's 86.8%, SCP would retain ≈11 rather
than 43, giving precision ≈95% at unchanged recall and F1 ≈0.94. Mark clearly as
a projection, not a measurement.

---

## 5. Caveats — all five are mandatory

1. **SCP was tuned on this movie; the others were not.** Its parameters come from
   a 40-trial Optuna search with spatial cross-validation on this exact
   recording. Papylio and DeepFRET ran with the config's shipped parameters.
   This is a real home advantage and must be stated where the comparison is
   drawn, not buried.
2. **N = 1.** One annotated movie, `repeats: 1`. No error bars, no significance
   tests, no "significantly better".
3. **SCP received one channel; the others received both.** SCP is a
   single-channel detector and channel 1 is what it was tuned on; DeepFRET and
   Papylio are genuinely two-colour. Defensible, but state it. (Measured: on
   summed channels SCP drops to F1 0.755, below Papylio.)
4. **Off-target class semantics are undocumented.** `class.npy` class 1 is
   whatever the human annotator marked as off-target; no written criterion
   survives. Measured, these spots are *fainter* than targets, which is atypical
   for fluorescent aggregates. Do not assert a physical mechanism.
5. **Detection-stage numbers are not comparable across tools.** "Detection"
   means something different in each pipeline (SCP: merged tracks pre-filter;
   the others: their own candidate stages). Compare selection stages; use
   detection only to describe each tool's internal strategy.

---

## 6. Methods facts the section may need

- **Data:** one dual-channel smFRET movie, 501 frames, 2 × 256 × 512, 16-bit.
- **Ground truth:** 334 manually annotated spots — 243 target (class 0), 91
  off-target (class 1), 0 moving.
- **Matching:** Hungarian (globally optimal) assignment, 2 px Euclidean threshold.
- **SCP parameters:** min_sigma 0.452, max_sigma 1.269, threshold 0.323,
  circularity 0.202, max_displacement 4.993, min_pulsation_frames 20,
  intensity_variation_threshold 0.072, bandpass on (σ = 8.0), event NMS 3.0.
- `min_pulsation_frames` is a **minimum track length**, not a pulsation
  criterion, despite the name. If the parameter is named in the text, say so.

Useful consistency check, worth one sentence: SCP's selection F1 here (0.850) is
within 0.003 of the 0.847 obtained during parameter selection on a held-out
spatial band with a different scorer. Two independent evaluations agreeing is
evidence the tuning did not overfit the band.

---

## 7. Do not write

- Any significance claim, p-value, or "consistently/reliably outperforms".
- Any citation of the retention PNGs (§2).
- Any mechanism for why off-target spots are faint (§5.4).
- Any comparison of detection-stage numbers across tools (§5.5).
- Any suggestion the three tools were tuned equally (§5.1).
- The `p025`/`p975` columns as uncertainty (§3).

---

## 8. Two upstream bugs fixed for this run

Mention only if the section must explain why numbers differ from earlier runs;
otherwise these belong in Methods or a changelog.

1. **Global metrics were always zero.** In
   `scripts/data_analysis/matching_and_benchmarking.py`, `class_names` is keyed
   by integer class id but was looked up with the string `"Stationary"`,
   silently falling through to a zero default. Every `Global_Precision`,
   `Global_Recall` and `Global_F1_Score` this pipeline produced before this fix
   was `0.0`. Per-class numbers were unaffected.
2. **Plotting crashed natively.** The conda-forge matplotlib Windows build
   aborts in the Agg renderer (`0xC0000409`, no traceback); reproduced across
   matplotlib 3.8.4/3.10.9/3.11.1, numpy 1.26/2.2/2.4, pandas 2.2/2.3/3.0 and
   freetype 2.12.1/2.14.3. The pip wheel of the identical version works. Fixed
   by giving the plotting rule a pip-only conda env (`envs/plotting.yaml`).

---

## 9. Style

Past tense, third person, no first person plural unless the surrounding thesis
uses it. Report numbers to the precision given here. Lead each paragraph with
the finding, then the evidence, then the qualification. Reference figures
explicitly ("Figure 1B"). Target 600–900 words excluding the table.
