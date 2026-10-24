# Unified benchmark re-scoring

**Problem.** The same SCP configuration had two F1 values on record — 0.850 from the
Snakemake benchmark and 0.847 from parameter tuning — because the two used different
scorers and different regions. A reader cannot tell whether the difference means anything.
It does not.

**Fix.** Score all three algorithms with the *tuning* scorer, over one region, so a single
number describes each pipeline everywhere in the thesis.

---

## The two scorers

| | Snakemake benchmark (old) | parameter tuning (used here) |
|---|---|---|
| assignment | Hungarian, globally optimal | greedy 1:1, closest pair first |
| gate | Euclidean ≤ 2 px | per-axis, \|dx\| ≤ 3 **and** \|dy\| ≤ 3 px |
| region | whole movie | held-out band |
| implementation | `scripts/data_analysis/matching_and_benchmarking.py` | `Parameter_opt/SCP_parametrizer_002.py` |

Both count every detection not matched to a **target** annotation as a false positive,
including detections on annotated off-target spots, and both compute
recall = TP / 243, precision = TP / (all detections), F1 = harmonic mean.

`unified_benchmark.py` imports `compute_metrics` and `match_detections_to_positives` from
the parametrizer rather than reimplementing them, so the numbers come from the same code
path that produced every tuning figure in the thesis.

---

## Result

SCP's selection-stage F1 is **0.8539 on the whole movie** and **0.8538 on the held-out band
x[340,512)**. They agree to four digits, so one figure — **0.854** — can be quoted for both
the tuning result and the benchmark result with no region asterisk. The recorded 0.847
retires; it differs by one true positive because the benchmark's detection array came from
a separate detector run through the Snakemake wrapper.

Whole movie, 243 target and 91 off-target annotations:

| algorithm | stage | detections | targets found | off-target | pure noise | recall | precision | F1 |
|---|---|---|---|---|---|---|---|---|
| DeepFRET | detection | 400 | 197 | 56 | 147 | 81.1% | 49.2% | 0.613 |
| DeepFRET | selection | 248 | 170 | 39 | 39 | 70.0% | 68.5% | **0.692** |
| Papylio | detection | 305 | 221 | 69 | 15 | 90.9% | 72.5% | 0.807 |
| Papylio | selection | 188 | 173 | 10 | 5 | 71.2% | 92.0% | **0.803** |
| SCP | detection | 1867 | 243 | 86 | 1538 | 100.0% | 13.0% | 0.230 |
| SCP | selection | 284 | 225 | 45 | 14 | 92.6% | 79.2% | **0.854** |

**The ranking and every qualitative conclusion are unchanged** from the Hungarian scoring.
Only the third decimal moves, plus small shifts in the per-class counts (SCP off-target
retained 43 → 45, DeepFRET targets found 168 → 170).

---

---

## The same benchmark on the 0.847 band, x[256,512)

The 0.847 was reported on `REAL_TEST_XBAND` = x[256,512) — the reference detector's
held-out band, tagged `test_x256_512` in `measure_real_movie.py` and carrying
§3.5's R 94.6% / P 76.6% / F1 0.847. `unified_benchmark.py --band 256 512` restricts
both the annotations and the detections to that band and writes to
`output_x256_512/`, leaving the whole-movie output untouched.

111 target and 42 off-target annotations fall in the band (of 334 on the movie):

| algorithm | stage | detections | targets found | off-target | pure noise | recall | precision | F1 |
|---|---|---|---|---|---|---|---|---|
| DeepFRET | detection | 188 | 85 | 25 | 78 | 76.6% | 45.2% | 0.569 |
| DeepFRET | selection | 115 | 72 | 17 | 26 | 64.9% | 62.6% | **0.637** |
| Papylio | detection | 145 | 102 | 32 | 11 | 91.9% | 70.3% | 0.797 |
| Papylio | selection | 79 | 70 | 6 | 3 | 63.1% | 88.6% | **0.737** |
| SCP | detection | 912 | 111 | 38 | 763 | 100.0% | 12.2% | 0.217 |
| SCP | selection | 135 | 104 | 22 | 9 | 93.7% | 77.0% | **0.846** |

Two things to read off this:

**SCP is band-insensitive; its competitors are not.** SCP moves 0.854 → 0.846 between
the whole movie and this band, while Papylio drops 0.803 → 0.737 and DeepFRET
0.692 → 0.637. So restricting to the 0.847 band *widens* SCP's margin (+0.051 → +0.109
over Papylio) rather than narrowing it. Neither competitor was tuned on this movie, so
that swing is region heterogeneity, not a like-for-like disadvantage — which is the
argument for quoting the whole movie in the head-to-head and this band only when the
point is comparability with a tuning number.

**SCP's 0.846 here is not the same measurement as the recorded 0.847.** Same band and
same scorer, but the detections come from the Snakemake wrapper's detector run, not the
tuning run, and the whole-movie reconciliation already showed those two arrays differ by
one true positive. The agreement to three digits is a cross-check, not an identity.

---

## Files

| file | what it is |
|---|---|
| `unified_benchmark.py` | computes per-class and global metrics, writes `output/*.csv`; `--band XMIN XMAX` restricts to a half-open x band and writes `output_x{XMIN}_{XMAX}/` instead |
| `plot_unified.py` | regenerates the three thesis figures from those CSVs; `--out DIR` reads a band directory |
| `rescore_all.py` | diagnostic: the same scoring repeated over five spatial bands, used to show the whole-movie and held-out-band figures agree |
| `output/summary.csv` | one row per algorithm × stage — the table above |
| `output/class_benchmarks_{alg}.csv` | per-class detected/retained/retention, same schema as the pipeline's |
| `output/global_benchmarks_{alg}.csv` | precision/recall/F1 at detection and selection |
| `output/band_scores.txt` | `rescore_all.py` output, all five bands |
| `output/run_metadata.json` | scorer settings, region, input paths |
| `output/figures/*.png` | regenerated Figure 1, Figure 2, Table 1 |

## Reproduce

```bash
cd smTIRF_GUI_SCP_Sim_Param/smTIRF_GUI_SCP_Sim_Param/experiments/unified_benchmark_rescoring
python unified_benchmark.py                  # metrics -> output/
python plot_unified.py                       # figures -> output/figures/
python rescore_all.py > output/band_scores.txt   # per-band diagnostic

# the same benchmark on the 0.847 band
python unified_benchmark.py --band 256 512           # -> output_x256_512/
python plot_unified.py --out output_x256_512         # -> output_x256_512/figures/
```

Inputs are the detection and selection arrays the Snakemake pipeline already wrote
(`BachelorThesis-FRETAlgorithms/results/{alg}/{detections,selections}/rep1.npy`, shape
(N, 2) in `[y, x]`) and the real labels (`realdata/coord.npy`, `realdata/class.npy`).
**No detector is re-run** — only the scoring changes, so the comparison isolates the scorer.

---

## Two things to carry into the write-up

**One framing, not two.** Under this scorer the Results section needs only a single metrics
table. The old section had two — one excluding pure-noise detections, one counting them —
because the Hungarian pipeline reported both. Here every unmatched detection is a false
positive throughout, so the "noise excluded" variant is gone.

**The figures are regenerated, so use these.** `output/figures/*_unified.png` replace
`BachelorThesis-FRETAlgorithms/results/detection_selection_bars.png`,
`recall_precision_plot.png` and `metrics_table.png`. The originals are left untouched and
are still consistent with each other under Hungarian scoring — do not mix the two sets.

## Band diagnostic, for reference

Selection-stage F1 by region (`output/band_scores.txt`):

| band | SCP | Papylio | DeepFRET |
|---|---|---|---|
| whole movie x[0,512) | 0.854 | 0.803 | 0.693 |
| direct-CV train x[0,170) | 0.867 | 0.843 | 0.721 |
| direct-CV val x[170,340) | 0.841 | 0.814 | 0.688 |
| direct-CV test x[340,512) | 0.854 | 0.740 | 0.642 |
| sim-guided test x[256,512) | 0.846 | 0.737 | 0.637 |

Papylio's swing across bands (0.843 → 0.740) is region heterogeneity, not tuning — it was
never tuned on this movie. That is the argument for reporting the whole movie in the
head-to-head: a band choice moves Papylio more than it moves SCP.
