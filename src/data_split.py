import numpy as np


def create_labels_and_features(processed_data):
    """Extract labels (weeks_until_onset) and feature array from the processed DataFrame."""
    labels = np.array(processed_data["weeks_until_onset"])
    features = np.array(processed_data.drop(columns=["weeks_until_onset"]))
    return labels, features


def create_splits(n_folds, n_years, weeks_per_year):
    """Create temporal cross-validation splits (blocked by year, not shuffled).

    Each fold uses a contiguous block of years as the test set and the
    remaining years as training. This preserves temporal structure.

    Returns:
        list of (train_indices, test_indices) tuples (index into flattened array)
    """
    splits = []
    fold_size = n_years // n_folds
    total_weeks = n_years * weeks_per_year

    for i in range(n_folds):
        start_test_year = i * fold_size
        end_test_year = (i + 1) * fold_size if i < n_folds - 1 else n_years

        test_indices = np.arange(
            start_test_year * weeks_per_year, end_test_year * weeks_per_year
        )
        train_indices = np.setdiff1d(np.arange(total_weeks), test_indices)
        splits.append((train_indices, test_indices))

    return splits
