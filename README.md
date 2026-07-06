# Coral Heat Stress Prediction

XGBoost model that predicts coral bleaching heat stress onset up to 6 weeks in advance for three Florida Keys reef sites. Given the current week's environmental conditions, the model outputs *weeks until onset* — how many weeks until Degree Heating Weeks (DHW) exceed the 4 °C-week bleaching threshold — or predicts that no onset will occur this season.

**Sites:** Sombrero Reef · Sand Key · Molasses Reef  
**Prediction target:** weeks until DHW ≥ 4 °C-weeks (classes 0–6) or no onset (class 7)  
**Summer window:** ISO weeks 21–40 (May–October)  
**Training period:** 1985–present

---

## Repository structure

```
coral_heat_stress_prediction/
├── Data/
│   ├── sombrero_preds.csv        ← per-site feature/label CSVs (not in git)
│   ├── sk_preds.csv
│   └── molasses_preds.csv
├── notebooks/
│   ├── compute_all_variables.ipynb     ← update all input data (run first)
│   ├── stress_predictions_original.ipynb   ← train fold-based models
│   ├── stress_predictions_nofolds.ipynb    ← train no-fold model
│   ├── predict_2025_2026_folds.ipynb       ← generate current-season predictions
│   ├── predict_2025_original.ipynb         ← predictions using no-fold model
│   └── hyperparameter_sweep.ipynb          ← tune XGBoost hyperparameters
├── src/
│   ├── preprocess.py             ← CoralBleachingPreprocessor class
│   ├── data_split.py             ← blocked temporal cross-validation splits
│   └── train_test_model.py       ← model training, masking, metrics
├── output/                       ← saved models and plots (not in git)
│   └── <EXP_NAME>/
│       ├── site_xgb_params.json
│       └── <site>/models/ + plots/
├── docs/
│   └── EXPERIMENTS.md            ← log of all named experiment runs
└── environment.yml
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate env-coral-stress-predict
```

The environment includes xgboost, xarray, netCDF4, pandas, matplotlib, and all OPeNDAP dependencies.

---

## Input data

Each site CSV (`sombrero_preds.csv`, `sk_preds.csv`, `molasses_preds.csv`) contains one row per ISO week from 1985 onward with the following columns used in predictions:

| Column | Source | Description |
|---|---|---|
| `meanDhw` | CRW | Degree Heating Weeks |
| `meanHs` | CRW | Coral Bleaching HotSpot |
| `meanSsta` | CRW | SST anomaly |
| `meanDhw_lag_1/2` | derived | DHW from prior 1–2 weeks |
| `airTempInter` | NCEP | Surface air temperature (°C) |
| `zonalWindInter` | NCEP | 925 mb zonal wind (m/s) |
| `oniInter` | NOAA PSL | Niño 3.4 SST anomaly index |
| `loop_curr_sst` | OISST | Loop Current / Caribbean SST anomaly index |

---

## Workflow

### Step 1 — Update input data

Run **`compute_all_variables.ipynb`** to pull the latest observations from all remote sources and refresh the site CSVs.

The notebook runs four sections in order:

| Section | Data source | Columns updated |
|---|---|---|
| 1 · CRW | NOAA Coral Reef Watch daily virtual-station files | `meanSst`, `meanSsta`, `meanHs`, `meanDhw` |
| 2 · NCEP | NOAA PSL NCEP Reanalysis via OPeNDAP | `airTempInter`, `zonalWindInter` |
| 3 · ONI | `nina34.anom.data` from NOAA PSL | `oniInter` |
| 4 · OISST | NOAA OISST weekly SST via OPeNDAP | `loop_curr_sst` |

No configuration changes are needed — the notebook automatically detects the latest available week from each source and appends new rows. If the OISST product has not yet been published for the most recent week, `loop_curr_sst` is automatically in-filled from the prior week.

After running, verify the final output in the **Final Verification** cell — it prints the last 5 rows and NaN counts for all updated columns.

---

### Step 2 — Train models (if needed)

Models only need to be re-trained when the training dataset is updated (e.g., after a new season completes) or when starting a new experiment.

There are two training notebooks depending on the experiment type:

**Fold-based (exp001–003):** `stress_predictions_original.ipynb`
- Uses 5-fold blocked temporal cross-validation (~8 years per fold)
- Set `EXP_NAME` in the Configuration cell, then run all cells
- Saves one model per fold per site to `output/<EXP_NAME>/<site>/models/`

**No-fold (exp004+):** `stress_predictions_nofolds.ipynb`
- Trains a single model on all available data up to a cutoff date
- Set `EXP_NAME` and `DATE_CUTOFF` in the Configuration cell, then run all cells
- Saves one model per site to `output/<EXP_NAME>/<site>/models/`

Hyperparameters are loaded automatically from `output/site_xgb_params.json`. To retune, run `hyperparameter_sweep.ipynb` first.

---

### Step 3 — Generate predictions

Run **`predict_2025_2026_folds.ipynb`** to evaluate the trained model on current-season data. Update the Configuration cell before running:

```python
EXP_NAME        = 'exp003'   # experiment to load models from
RUN_THROUGH_WEEK = 'wk27'    # ISO week of most recent CSV data — used to name the output folder
MODEL_FOLD      = 5           # fold model to use (None for no-fold experiments)
EVAL_YEARS      = [2025, 2026]  # years to predict
```

**Which experiment/model to use:**

| Experiment | Model type | Training period | Evaluation years |
|---|---|---|---|
| exp003 | Fold 5 (fold-based) | 1985–2016 | 2025, 2026 |
| exp004 | No-fold | 1985–2023 | 2024, 2025, 2026 |

The notebook produces three outputs for each evaluation year:

1. **Printed predictions** — truth (if the season is complete) and XGB predicted class for each summer week
2. **Plots** — saved to `output/<EXP_NAME>/predictions_2025_2026_<RUN_THROUGH_WEEK>/`
   - *Complete years (verified):* side-by-side timeline and outcome bar chart
   - *In-progress years:* prediction-only plot with colored markers (red = onset imminent, gray = no onset) and ISO week start dates on the x-axis
3. **Summary table** — hit/miss/false alarm counts and F1 score per site per year

Each run is saved to its own output folder (e.g., `predictions_2025_2026_wk27/`) so previous runs are never overwritten. Update `RUN_THROUGH_WEEK` each time you re-run with fresh data.

---

## Experiment naming

All experiment runs are documented in `docs/EXPERIMENTS.md`. Follow the numbered convention (`exp001`, `exp002`, …) and add an entry whenever you run a new experiment. Never delete or overwrite an existing experiment folder — create a new numbered folder instead.

---

## Key concepts

**Post-onset masking:** Once the model predicts onset (class 0) in a season, all subsequent predictions for that season are masked to NaN and excluded from all metrics and plots. This prevents a single early prediction from inflating performance statistics.

**Weeks until onset (prediction classes):**
- `0` — onset predicted this week (DHW ≥ 4 imminent)
- `1–6` — onset predicted in N weeks
- `7` — no onset predicted this season

**Tolerance:** A prediction is scored as a hit if it falls within ±1 week of the true onset week.

**`airTempInter` is in degrees Celsius** — not Kelvin. Conversion from NCEP's raw Kelvin output (−273.15) is applied in `compute_all_variables.ipynb`.
