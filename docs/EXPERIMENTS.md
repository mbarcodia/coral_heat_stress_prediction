# Experiment Log

This file documents every named experiment run stored under `output/`. Each entry
records what was run, why, what changed relative to the prior experiment, and the
key results. This log is required for reproducibility and for the public GitHub
release accompanying the ERC publication.

**Convention:** Experiment folders are named numerically (e.g. `exp402`). All
associated outputs — models, plots, metrics JSON — are written to
`output/<EXP_NAME>/<site>/`. Do not delete or overwrite a named experiment folder;
create a new one instead.

---

## exp402

**Path:** `output/exp402/`  
**Status:** Complete — all three sites  
**Purpose:** Adapted from the original `bleaching_predict_clean_repo` pipeline for
the ERC publication. First full three-site run in the new `coral_heat_stress_prediction`
repository using corrected input data.

### What distinguishes this experiment

This is the baseline experiment for the ERC publication. It uses corrected versions
of three features that contained errors in the old repository:

| Feature | Old repo (bleaching_predict_clean_repo) | exp402 (this repo) |
|---|---|---|
| `airTempInter` | Stored in Kelvin (~300 K) | Stored in Celsius (~25 °C) — correct; must remain in °C |
| `oniInter` | Running-sum bug for 2024 weeks 38–40 (values: −20, −40, −60) | Correct values (~−0.18) |
| `zonalWindInter` | Different interpolation method (corr ≈ 0.52 with new) | Corrected interpolation |

`meanDhw`, `meanHs`, `meanSsta`, and `loop_curr_sst` are identical between old and
new CSVs. Labels (`weeks_until_onset`) are derived solely from `meanDhw` and are
therefore identical in both pipelines.

Two features present in the new pipeline were **not present** in the old repo:
- `loop_curr_sst` — Loop Current SST index (built by `compute_loop_curr_sst.ipynb`)
- `dswrf` — downwelling shortwave radiation / solar flux (built by `compute_dswrf.ipynb`)

### Configuration

| Parameter | Value |
|---|---|
| Sites | sombrero, sk (Sand Key), molasses |
| Data file | `{site}_preds.csv` |
| Date cutoff | Row index 2114 (through 2024) |
| Years per site | 40 (1985–2024) |
| Cross-validation | 5-fold blocked temporal (8 years/fold) |
| Stress threshold (DHW) | 4 °C-weeks |
| Prediction span | 6 weeks |
| `val_no_onset` label | 7 |
| Summer window | Weeks 21–40 (20 weeks/year) |
| Sample weights | No-onset: 0.02, Onset: 1.0 |
| Onset tolerance | ±1 week |
| Post-onset masking | Applied — predictions after first onset in a season are set to NaN and excluded from all metrics |

### Features used

```
week, meanDhw, meanHs, weeks_until_onset,
meanDhw_lag_1, meanDhw_lag_2, meanSsta,
airTempInter, zonalWindInter, oniInter,
loop_curr_sst, dswrf
```

### Hyperparameters

Tuned via one-at-a-time sweep in `hyperparameter_sweep.ipynb`. Best parameters
saved to `output/site_xgb_params.json` and loaded automatically by
`stress_predictions.ipynb`.

| Parameter | Sombrero | Sand Key | Molasses |
|---|---|---|---|
| `n_estimators` | 5000 | 5000 | 5000 |
| `max_depth` | 10 | 10 | 10 |
| `learning_rate` | 0.1 | 0.5 | 0.1 |
| `gamma` | 0.0 | 0.0 | 0.0 |
| `subsample` | 0.7 | 0.7 | 0.5 |
| `reg_lambda` | 1.0 | 0.0 | 0.0 |
| `reg_alpha` | 1.0 | 1.0 | 1.0 |
| `random_state` | 13 | 13 | 13 |

Shared fixed parameters: `objective=multi:softmax`, `booster=gbtree`,
`tree_method=auto`, `min_child_weight=1`, `max_delta_step=0`,
`colsample_bytree=colsample_bylevel=colsample_bynode=1.0`.

### Results (XGBoost, ±1 week tolerance, post-onset masked)

| Site | Correct Onset | Missed Onset | False Alarm | True No-Onset | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Sombrero | 59 / 70 (84.3%) | 11 / 70 (15.7%) | 74 / 667 (11.1%) | 593 / 667 (88.9%) | 0.444 | 0.843 | 0.581 |
| Sand Key | 54 / 77 (70.1%) | 23 / 77 (29.9%) | 66 / 650 (10.2%) | 584 / 650 (89.9%) | 0.450 | 0.701 | 0.548 |
| Molasses | 30 / 42 (71.4%) | 12 / 42 (28.6%) | 55 / 724 (7.6%) | 669 / 724 (92.4%) | 0.353 | 0.714 | 0.472 |

**Note on denominators:** Onset denominator < total onset-class labels because
post-onset weeks are masked. No-onset denominator < (800 − onset labels) for the
same reason. See session notes for exact masked-row counts per site (sombrero: 63,
sk: 73, molasses: 34).

### Comparison to old repo (Sombrero only — old repo ran sombrero by default)

| Metric | Old repo XGB | exp402 XGB | Change |
|---|---|---|---|
| Correct Onset | 60 / 70 (85.7%) | 59 / 70 (84.3%) | −1 |
| False Alarm | 105 / 667 (15.7%) | 74 / 667 (11.1%) | **−31** |
| Precision | 0.364 | 0.444 | +0.080 |
| Recall | 0.857 | 0.843 | −0.014 |
| **F1** | 0.511 | **0.581** | **+0.070** |

Improvement is driven by 31 fewer false alarms, attributable to the corrected
feature data (primarily the ONI running-sum bug fix).

### Outputs

```
output/exp402/
├── site_xgb_params.json          ← best hyperparameters per site
├── sombrero/
│   ├── models/                   ← XGB fold 1–5 models + linear coefficients
│   └── plots/                    ← metrics table, performance plot, SHAP summary,
│                                    per-fold per-class SHAP beeswarms
├── sk/
│   └── (same structure)
└── molasses/
    └── (same structure)
```

---

## exp001

**Path:** `output/exp001/`  
**Status:** Complete — all three sites  
**Purpose:** First officially numbered experiment in this repository. Establishes
the clean baseline for the ERC publication using the fully corrected input data,
per-site tuned hyperparameters, and the fixed `dswrf` feature label.

### What distinguishes this experiment from exp402

exp402 was a transitional run adapted directly from the old `bleaching_predict_clean_repo`
folder. exp001 is the first run made under the new numbered experiment convention
with the following confirmed fixes applied:

| Change | exp402 | exp001 |
|---|---|---|
| `rename_features` map | `dswrf` missing — SHAP labels showed raw column name | `dswrf` → `"solar flux"` added |
| Experiment numbering | Legacy number carried over from old repo | Starts fresh at 001 |

Everything else is identical to exp402: same corrected CSVs, same features, same
hyperparameters (loaded from `output/site_xgb_params.json`), same pipeline version.

### Configuration

Identical to exp402 — see that entry for full parameter table.  
`EXP_NAME = 'exp001'`, `DATE_CUTOFF = 2114`.

### Features used

```
week, meanDhw, meanHs, weeks_until_onset,
meanDhw_lag_1, meanDhw_lag_2, meanSsta,
airTempInter, zonalWindInter, oniInter,
loop_curr_sst, dswrf
```

### Hyperparameters

Loaded dynamically from `output/site_xgb_params.json`. Same values as exp402 —
see that entry for the full table.

### Results (XGBoost, ±1 week tolerance, post-onset masked)

Results are identical to exp402, as expected — the only change between the two
experiments was the SHAP axis label for `dswrf`; model training is unaffected.

**XGBoost**

| Site | Correct Onset | Missed Onset | False Alarm | True No-Onset | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Sombrero | 59 / 70 (84.3%) | 11 / 70 (15.7%) | 74 / 667 (11.1%) | 593 / 667 (88.9%) | 0.444 | 0.843 | 0.581 |
| Sand Key | 54 / 77 (70.1%) | 23 / 77 (29.9%) | 66 / 650 (10.2%) | 584 / 650 (89.9%) | 0.450 | 0.701 | 0.548 |
| Molasses | 30 / 42 (71.4%) | 12 / 42 (28.6%) | 55 / 724 (7.6%) | 669 / 724 (92.4%) | 0.353 | 0.714 | 0.472 |

**Linear**

| Site | Correct Onset | Missed Onset | False Alarm | True No-Onset | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Sombrero | 52 / 70 (74.3%) | 18 / 70 (25.7%) | 291 / 667 (43.6%) | 376 / 667 (56.4%) | 0.152 | 0.743 | 0.252 |
| Sand Key | 50 / 77 (64.9%) | 27 / 77 (35.1%) | 252 / 650 (38.8%) | 398 / 650 (61.2%) | 0.166 | 0.649 | 0.264 |
| Molasses | 25 / 42 (59.5%) | 17 / 42 (40.5%) | 148 / 724 (20.4%) | 576 / 724 (79.6%) | 0.145 | 0.595 | 0.233 |

**Frequency baseline**

| Site | Correct Onset | Missed Onset | False Alarm | True No-Onset | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Sombrero | 17 / 70 (24.3%) | 53 / 70 (75.7%) | 108 / 667 (16.2%) | 559 / 667 (83.8%) | 0.136 | 0.243 | 0.174 |
| Sand Key | 18 / 77 (23.4%) | 59 / 77 (76.6%) | 129 / 650 (19.8%) | 521 / 650 (80.2%) | 0.122 | 0.234 | 0.161 |
| Molasses | 8 / 42 (19.0%) | 34 / 42 (81.0%) | 58 / 724 (8.0%) | 666 / 724 (92.0%) | 0.121 | 0.190 | 0.148 |

### Outputs

```
output/exp001/
├── sombrero/models/    ← XGB fold 1–5 + linear coefficients
├── sombrero/plots/     ← metrics table, performance, SHAP summary, per-fold SHAP
├── sk/
└── molasses/
```

---

## Template for future experiments

Copy this block when creating a new experiment.

```
## expXXX

**Path:** `output/expXXX/`
**Status:** [In progress / Complete / Abandoned]
**Purpose:** [Why this experiment was run; what question it answers]

### What distinguishes this experiment

[What changed relative to the previous experiment — features, hyperparameters,
data range, code fixes, etc. Be specific.]

### Configuration

[Table of key parameters — copy from exp402 template and update changed values only]

### Features used

[List features]

### Hyperparameters

[Table or reference to site_xgb_params.json if loaded dynamically]

### Results

[Copy results table format from exp402]

### Outputs

[List what was generated]
```
