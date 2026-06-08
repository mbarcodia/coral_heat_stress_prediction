# Data

Place your reef site CSV files here. Expected filenames (one per site):

- `sombrero_preds2024.csv`
- `molasses_preds2024.csv`
- `sk_preds2024.csv`

When new data files are available (e.g., through 2025), update two settings in  
`notebooks/stress_predictions.ipynb`:

1. `DATA_FILENAME_TEMPLATE` — e.g., change `preds2024` to `preds2025`
2. `DATE_CUTOFF` — increase to include additional rows from the new file

CSV files are excluded from version control via `.gitignore`.
