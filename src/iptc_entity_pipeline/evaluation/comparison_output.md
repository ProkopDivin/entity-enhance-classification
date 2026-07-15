# Evaluation Comparison Output

`comparison.py` compares two saved evaluation runs (**current** vs **base**) and writes results to a timestamped folder:

```
results/comparisons/{config_name}_{YYYYMMDD_HHMMSS}/
├── evaluation_comparison_{config_name}.xlsx   # all tables as sheets
├── *.csv                                    # one CSV per table (same data)
└── brier_mean_error_diff_*.png              # optional diagnostic plots
```

The Excel workbook and CSV files contain the same tables. Use the workbook when you need row labels for `current_*` / `base_*` tables (see below).

**Column convention:** comparison tables use `*_current`, `*_base`, and `*_diff` (current minus base). Positive `*_diff` means the current run is better for metrics like F1, Precision, Recall, and PR-AUC.

---

## Quick start — where to look first

| Question | Table |
|----------|-------|
| Did the model improve overall? | `summary_comparison` |
| Which corpora / languages changed? | `corpora_comparison`, `language_comparison` |
| Which IPTC classes changed most? | `top_improved`, `top_degraded` |
| Is the F1 change statistically significant? | `top_improved` / `top_degraded` (`mcnemar_*` columns) |
| How good are raw scores (threshold-free)? | `pr_auc_summary`, `pr_auc_per_class` |
| Which entities correlate with gains? | `entity_impact_all` |

---

## Tables

### `summary_comparison`

Compact headline metrics for the whole test set.

| Column | Meaning |
|--------|---------|
| `summary_key` | Aggregation level: `micro`, `macro_head` (classes with ≥15 test articles), `macro_tail` (<15), or `macro_support_{bucket}` (0-15, 15-100, 100-1000, 1000+) |
| `precision_*`, `recall_*`, `f1_*` | Precision, recall, F1 for current, base, and diff |

**Start here** for thesis-level “did we win?” numbers.

---

### `corpora_comparison`

Per-corpus metrics (micro-averaged over labels within each corpus).

| Column | Meaning |
|--------|---------|
| `Corpus Name` | Test corpus id (e.g. `en_bbc_iptc`). Summary rows: `All_micro`, `All_macro_corpora`, `All_datapoint` |
| `Data Count` | Number of test articles |
| `Docs No Labels` | Share of articles with no gold labels |
| `Decent Labels` | Count of “decent” gold labels (evaluation metadata) |
| `Precision_*`, `Recall_*`, `F1_*` | Standard metrics |
| `False Positive Rate_*` | Micro false-positive rate over gold-negative label slots |

Shows **where** (which publisher/language corpus) the model improved or regressed.

---

### `corpora_comparison_macro_head`

Same layout as `corpora_comparison`, but metrics are **macro-averaged only over head classes** (support ≥ 15) within each corpus. Summary row: `All_macro_head_corpora`.

Useful when tail classes would dominate or distort per-corpus macro scores.

---

### `language_comparison`

Metrics aggregated across corpora by language prefix (`en`, `de`, `cs`, …). Same columns as `corpora_comparison`, keyed by `Language` instead of `Corpus Name`.

---

### `language_comparison_macro_head`

Language-level version of `corpora_comparison_macro_head` (macro head classes only).

---

### `classes_comparison`

Per-IPTC-category comparison — the main class-level table.

| Column | Meaning |
|--------|---------|
| `IPTC Category` | Human-readable class name (quoted long label) |
| `class_id` | IPTC category id |
| `Data Count_*` | Number of test articles with this label |
| `Precision_*`, `Recall_*`, `F1_*` | Per-class metrics |
| `False Positive Count_current` | Absolute false positives in the current run (base/diff FP columns omitted to keep the table compact) |

Aggregate rows: `All_micro`, `All_macro_head`, `All_macro_tail`, `All_datapoint`.

---

### `class_confusion_counts`

Per-class confusion counts (summed over all articles), not rates.

| Column | Meaning |
|--------|---------|
| `Fp_*`, `Fn_*`, `Tp_*`, `Tn_*` | False positive, false negative, true positive, true negative counts for current, base, and diff |

Helps explain *why* F1 changed (more FP vs fewer FN, etc.).

---

### `class_thresholds`

Per-class decision thresholds used for each run (auto-loaded from `custom_thresholds.json` / `thresholds.json`, or `--threshold-eval` fallback).

| Column | Meaning |
|--------|---------|
| `count` | Gold-label support (positive articles) |
| `threshold_current`, `threshold_base`, `threshold_diff` | Effective threshold per class |

---

### `top_improved` / `top_degraded`

All classes with `F1_diff > 0` (improved) or `F1_diff < 0` (degraded), sorted by `F1_diff`.

| Column | Meaning |
|--------|---------|
| `article_frequency` | Test-set support for the class |
| `Precision_*`, `Recall_*`, `F1_*` | Metrics and deltas |
| `mcnemar_pass` | `1` if McNemar test passes FDR correction in the expected direction (current better for improved; base better for degraded) |
| `mcnemar_p_value`, `mcnemar_p_value_fdr` | Raw and Benjamini–Hochberg-adjusted p-values |
| `mcnemar_n10_current_only_correct` | Articles correct only under current |
| `mcnemar_n01_base_only_correct` | Articles correct only under base |

McNemar requires ≥25 disagreeing articles; otherwise p-values are empty.

---

### `top_improved_stats` / `top_degraded_stats`

Key–value summary of the ranked change lists (`metric`, `value`).

Examples: `count_improved`, `count_improved_f1_diff_gt_0.1`, `avg_f1_diff_top_100`, `top_level_top_100::sport` (how many top-100 changes fall under each IPTC top-level topic).

---

### `pr_auc_per_class`

Threshold-free ranking quality per class (average precision / PR-AUC).

| Column | Meaning |
|--------|---------|
| `pr_auc_current`, `pr_auc_base`, `pr_auc_diff` | PR-AUC per class |
| `positive_support`, `article_frequency` | Number of positive test articles |

---

### `pr_auc_summary`

Aggregated PR-AUC across classes and corpora.

| Column | Meaning |
|--------|---------|
| `aggregation` | Level: `micro`, `macro_all`, `macro_tail`, `macro_support_{bucket}`, `macro_over_corpora_prefix_{lang}`, `macro_over_corpora_eurosport`, `macro_over_corpora` |
| `current`, `base`, `diff` | PR-AUC at that aggregation |

---

### `entity_impact_all`

Links extracted entities to per-article F1 gains (sum of `f1_diff` over articles mentioning the entity).

| Column | Meaning |
|--------|---------|
| `gkbid`, `stdform`, `entity_type` | Entity identifier and label |
| `avg_relevance`, `avg_mentions_count` | Mean entity signals across mentions |
| `entity_score` | Sum of `f1_diff` over articles containing the entity |
| `article_count` | Distinct articles with the entity |
| `normalized` | `entity_score / article_count` |

Sorted by `entity_score` descending. Last row `AVG` is the mean of numeric columns.

---

### `article_confusion_diff`

Per-article change in total TP/TN/FP/FN counts (current minus base, summed over all classes).

| Column | Meaning |
|--------|---------|
| `article_id`, `corpus_name` | Article identity |
| `tp_diff`, `tn_diff`, `fp_diff`, `fn_diff` | Confusion count deltas |

Positive `tp_diff` / negative `fn_diff` generally indicate improvement on that article.

---

### `current_corpora` / `base_corpora`

Standalone evaluation tables for one run (not a comparison). Metrics per corpus.

| Column | Meaning |
|--------|---------|
| Row index (Excel only) | `Corpus Name` |
| `Data Count`, `Precision`, `Recall`, `F1`, `False Positive Rate`, `Docs No Labels`, `Decent Labels` | Single-run metrics |

**Note:** In CSV exports the corpus name is only in the Excel sheet index; use the `.xlsx` sheet or align rows by order with `corpora_comparison`.

---

### `current_classes` / `base_classes`

Standalone per-class evaluation for one run.

| Column | Meaning |
|--------|---------|
| Row index (Excel only) | `IPTC Category` |
| `class_id`, `Data Count`, `Precision`, `Recall`, `F1`, `False Positive Count` | Single-run metrics |

---
