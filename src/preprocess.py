import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CoralBleachingPreprocessor:
    """Handles preprocessing of coral bleaching prediction data."""

    def __init__(
        self,
        filepath,
        reef_location,
        stress_threshold,
        val_no_onset,
        prediction_span,
        weeks_per_year,
    ):
        self.filepath = filepath
        self.reef_location = reef_location
        self.stress_threshold = stress_threshold
        self.val_no_onset = val_no_onset
        self.prediction_span = prediction_span
        self.weeks_per_year = weeks_per_year
        self.data = None

    def load_data(self, filename):
        full_path = f"{self.filepath}{filename}"
        self.data = pd.read_csv(full_path)
        print(f"Data loaded from {full_path}")
        self.data = self.data.drop(columns=self.data.filter(like="Unnamed").columns)

    def preprocess_time_variables(self):
        self.data["year"] = self.data["year"].astype(int)
        self.data["week"] = self.data["week"].astype(int)
        self.data["year_week"] = (
            self.data["year"].astype(str)
            + "-"
            + self.data["week"].astype(str).str.zfill(2)
        )
        self.data["time_variable"] = pd.to_datetime(
            self.data["year_week"] + "-1", format="%Y-%W-%w"
        )

    def filter_data(self, max_index=None):
        if max_index is not None:
            self.data = self.data.iloc[:max_index]
        print("Data filtered.")

    def calculate_weeks_until_onset(self, prediction_span):
        """Labels each row with weeks until the first DHW onset in that year.
        Values > prediction_span are set to val_no_onset. Post-onset rows are labeled 0."""

        def calculate_group(group):
            onset = group[group["meanDhw"] >= self.stress_threshold]
            if onset.empty:
                return pd.Series([self.val_no_onset] * len(group), index=group.index)
            onset_week = onset["week"].iloc[0]
            weeks_until_onset = []
            for week in group["week"]:
                if week < onset_week:
                    diff = onset_week - week
                    weeks_until_onset.append(
                        self.val_no_onset if diff > prediction_span else diff
                    )
                else:
                    weeks_until_onset.append(0)
            return pd.Series(weeks_until_onset, index=group.index)

        self.data["weeks_until_onset"] = (
            self.data.groupby("year").apply(calculate_group).reset_index(drop=True)
        )

    def add_lag_features(self, lag_features):
        """Adds lagged versions of specified features (e.g., {"meanDhw": [1, 2]})."""
        for feature, lags in lag_features.items():
            for lag in lags:
                self.data[f"{feature}_lag_{lag}"] = self.data[feature].shift(lag)

    def filter_summer_weeks(self, summer_start, summer_end):
        """Retain only summer weeks (inclusive) and drop all others."""
        if self.data is None:
            raise ValueError("Data not loaded.")
        self.data = self.data.drop(
            self.data[
                (self.data["week"] < summer_start) | (self.data["week"] > summer_end)
            ].index
        )
        print(f"Data filtered to weeks {summer_start}–{summer_end}.")

    def select_columns(self, columns_to_keep):
        """Reduce the DataFrame to only the specified columns."""
        self.data = self.data[columns_to_keep]

    def drop_missing_values(self):
        self.data = self.data.dropna()

    def identify_most_frequent_onset_week(self, verbose=True):
        """Return the most frequent calendar week of DHW onset across all years."""
        onset_weeks = (
            self.data[self.data["weeks_until_onset"] == 0]
            .groupby("year")["week"]
            .min()
            .dropna()
        )
        most_frequent = onset_weeks.mode().iloc[0]
        if verbose:
            print(f"Most Frequent Onset Week: {most_frequent}")
        return most_frequent

    def create_baseline_preds(self, summer_start, n_years, prediction_span, val_no_onset):
        """Build a frequency-baseline prediction array.

        Assigns a countdown to the most-frequent onset week for a random subset
        of years equal to the observed number of onset years. All other years
        predict val_no_onset (no onset).
        """
        most_frequent_week = self.identify_most_frequent_onset_week(verbose=False)
        index = most_frequent_week - summer_start + 1
        if index < 0:
            raise ValueError("Calculated index is negative. Check summer_start.")

        baseline_range = self._generate_countdown(index, prediction_span, val_no_onset)

        total_weeks = self.weeks_per_year * n_years
        baseline_preds = np.full(total_weeks, val_no_onset)

        onset_weeks = (
            self.data[self.data["weeks_until_onset"] == 0]
            .groupby("year")["week"]
            .min()
            .dropna()
        )
        total_onsets = len(onset_weeks)
        print(f"Total Onsets: {total_onsets}")

        if total_onsets > n_years:
            raise ValueError("More onsets than years available.")

        np.random.seed(13)
        onset_years = np.random.choice(n_years, size=total_onsets, replace=False)

        for year in onset_years:
            start_idx = year * self.weeks_per_year
            baseline_preds[start_idx : start_idx + len(baseline_range)] = baseline_range
            baseline_preds[start_idx + len(baseline_range) : (year + 1) * self.weeks_per_year] = 0

        print(f"Baseline Predictions Shape: {baseline_preds.shape}")
        return baseline_preds

    def _generate_countdown(self, index, prediction_span, val_no_onset):
        countdown = np.arange(index - 1, -1, -1)
        return np.where(countdown > prediction_span, val_no_onset, countdown)

    def plot_mean_dhw(self):
        plt.figure(figsize=(4, 2))
        plt.plot(
            np.array(self.data["time_variable"]),
            np.array(self.data["meanDhw"]),
            label="meanDHW",
        )
        plt.axhline(y=self.stress_threshold, color="r", linestyle="--", label="Stress Threshold")
        plt.ylim(0, 20)
        plt.yticks(np.arange(0, 21, 2))
        plt.title(f"Mean Degree Heating Week at {self.reef_location.capitalize()}")
        plt.ylabel("deg C weeks")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_onset_histogram(self):
        onset_weeks = (
            self.data[self.data["weeks_until_onset"] == 0]
            .groupby("year")["week"]
            .min()
            .dropna()
        )
        plt.figure(figsize=(4, 2))
        sns.histplot(onset_weeks, bins=20, kde=False, color="skyblue", edgecolor="black")
        plt.xlim(28, 37)
        plt.ylim(0, 4)
        plt.xticks(np.arange(28, 37, 1))
        plt.yticks(np.arange(0, 5, 1))
        plt.title(f"Histogram of Week of Onset at {self.reef_location.capitalize()}")
        plt.xlabel("Week of Onset")
        plt.ylabel("Frequency")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
