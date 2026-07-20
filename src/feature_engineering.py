"""Feature engineering for the cleaned weather dataset.

Adds calendar/seasonality features, per-location lag and rolling-window
statistics, derived meteorological quantities (temperature difference,
humidity ratio, pressure trends) and categorical wind-strength bins.
The enriched dataset is written to ``weather_features.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import load_config, resolve_path, setup_logger, timed_step

logger = setup_logger("features")

#: Core series that receive lag and rolling-window features.
BASE_SERIES: list[str] = ["temperature_celsius", "humidity", "precip_mm"]

#: Meteorological season lookup by month (Northern Hemisphere convention;
#: hemisphere-aware season is derived from latitude at runtime).
_SEASON_BY_MONTH: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

_SEASON_FLIP: dict[str, str] = {
    "Winter": "Summer", "Spring": "Autumn",
    "Summer": "Winter", "Autumn": "Spring",
}


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month, day, week, quarter, year and hemisphere-aware season."""
    df = df.copy()
    ts = df["last_updated"]
    df["year"] = ts.dt.year
    df["quarter"] = ts.dt.quarter
    df["month"] = ts.dt.month
    df["week"] = ts.dt.isocalendar().week.astype(int)
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    north_season = df["month"].map(_SEASON_BY_MONTH)
    df["season"] = np.where(
        df["latitude"] >= 0,
        north_season,
        north_season.map(_SEASON_FLIP),
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-location lagged values of the core weather series.

    Assumes the frame is sorted by (location, timestamp), which the
    preprocessing stage guarantees.
    """
    lags: list[int] = load_config()["features"]["lag_days"]
    df = df.copy()
    grouped = df.groupby("location_name", observed=True)
    for column in BASE_SERIES:
        for lag in lags:
            df[f"{column}_lag{lag}"] = grouped[column].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-location rolling means and standard deviations."""
    windows: list[int] = load_config()["features"]["rolling_windows"]
    df = df.copy()
    grouped = df.groupby("location_name", observed=True)
    for column in BASE_SERIES:
        shifted = grouped[column].shift(1)
        by_location = shifted.groupby(df["location_name"], observed=True)
        for window in windows:
            rolling = by_location.rolling(window, min_periods=max(2, window // 2))
            df[f"{column}_rollmean{window}"] = rolling.mean().reset_index(
                level=0, drop=True
            )
            df[f"{column}_rollstd{window}"] = rolling.std().reset_index(
                level=0, drop=True
            )
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived meteorological quantities.

    * ``temp_feels_diff``   - wind-chill / heat-index effect in degrees C.
    * ``humidity_temp_ratio`` - humidity per Kelvin, a crude moisture index.
    * ``pressure_change_1d`` / ``pressure_trend_7d`` - short- and
      medium-term barometric tendencies, classic storm predictors.
    * ``temp_range_7d``     - rolling weekly temperature amplitude.
    """
    df = df.copy()
    grouped = df.groupby("location_name", observed=True)
    df["temp_feels_diff"] = df["feels_like_celsius"] - df["temperature_celsius"]
    df["humidity_temp_ratio"] = df["humidity"] / (df["temperature_celsius"] + 273.15)
    df["pressure_change_1d"] = grouped["pressure_mb"].diff(1)
    df["pressure_trend_7d"] = (
        grouped["pressure_mb"]
        .diff(7)
        .div(7.0)
    )
    rolling_max = (
        grouped["temperature_celsius"]
        .rolling(7, min_periods=3)
        .max()
        .reset_index(level=0, drop=True)
    )
    rolling_min = (
        grouped["temperature_celsius"]
        .rolling(7, min_periods=3)
        .min()
        .reset_index(level=0, drop=True)
    )
    df["temp_range_7d"] = rolling_max - rolling_min
    return df


def add_wind_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Bin wind speed into human-readable strength categories."""
    features_cfg = load_config()["features"]
    df = df.copy()
    df["wind_category"] = pd.cut(
        df["wind_kph"],
        bins=features_cfg["wind_speed_bins_kph"],
        labels=features_cfg["wind_speed_labels"],
        include_lowest=True,
    ).astype(str)
    return df


def run_feature_engineering() -> pd.DataFrame:
    """Load the clean dataset, add all engineered features and persist it."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["processed_data"]))

    with timed_step(logger, "calendar features"):
        df = add_calendar_features(df)
    with timed_step(logger, "lag features"):
        df = add_lag_features(df)
    with timed_step(logger, "rolling features"):
        df = add_rolling_features(df)
    with timed_step(logger, "derived features"):
        df = add_derived_features(df)
        df = add_wind_categories(df)

    out_path = resolve_path(config["paths"]["features_data"])
    df.to_parquet(out_path, index=False)
    logger.info("Feature dataset saved: %s rows, %s columns", *df.shape)
    return df


if __name__ == "__main__":
    run_feature_engineering()
