"""Data loading, validation and cleaning pipeline.

Transforms the raw Global Weather Repository CSV into two artifacts:

* ``weather_clean.parquet``  - physical units, deduplicated, validated,
  IQR-capped, with an Isolation Forest multivariate outlier flag.
* ``weather_scaled.parquet`` - standard-scaled numeric features plus
  label-encoded categoricals, ready for machine-learning models.

A JSON cleaning report describing every row-level decision is written to
``outputs/reports/cleaning_report.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils import (
    load_config,
    resolve_path,
    save_json,
    save_model,
    setup_logger,
    timed_step,
)

logger = setup_logger("preprocessing")

#: Imperial / duplicate columns dropped because a metric twin exists.
REDUNDANT_COLUMNS: list[str] = [
    "temperature_fahrenheit",
    "feels_like_fahrenheit",
    "wind_mph",
    "gust_mph",
    "pressure_in",
    "precip_in",
    "visibility_miles",
    "last_updated_epoch",
]

#: Categorical columns encoded for the ML-ready dataset.
CATEGORICAL_COLUMNS: list[str] = ["country", "location_name", "condition_text", "wind_direction"]

#: Multivariate feature set used for Isolation Forest outlier flagging.
OUTLIER_FEATURES: list[str] = [
    "temperature_celsius",
    "humidity",
    "pressure_mb",
    "wind_kph",
    "precip_mm",
    "cloud",
    "uv_index",
    "visibility_km",
]


def load_raw_data() -> pd.DataFrame:
    """Read the raw CSV and parse ``last_updated`` as a datetime column."""
    path = resolve_path(load_config()["paths"]["raw_data"])
    df = pd.read_csv(path)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    logger.info("Loaded raw data: %s rows, %s columns", *df.shape)
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove imperial-unit twins and other redundant columns."""
    present = [col for col in REDUNDANT_COLUMNS if col in df.columns]
    return df.drop(columns=present)


def remove_duplicates(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    """Drop exact duplicates and repeated (location, timestamp) observations."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["location_name", "last_updated"], keep="first")
    report["duplicates_removed"] = before - len(df)
    return df


def validate_physical_ranges(
    df: pd.DataFrame, report: dict[str, Any]
) -> pd.DataFrame:
    """Drop rows violating hard physical constraints from the config.

    Covers invalid coordinates, impossible temperatures/humidity and
    negative precipitation, among others.
    """
    ranges: dict[str, list[float]] = load_config()["cleaning"]["valid_ranges"]
    violations: dict[str, int] = {}
    mask = pd.Series(True, index=df.index)
    for column, (lower, upper) in ranges.items():
        if column not in df.columns:
            continue
        valid = df[column].between(lower, upper) | df[column].isna()
        violations[column] = int((~valid).sum())
        mask &= valid
    report["range_violations_by_column"] = violations
    report["rows_dropped_invalid_ranges"] = int((~mask).sum())
    return df.loc[mask]


def handle_missing_values(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    """Impute missing values: per-location median for numerics, mode fallback.

    Rows lacking a timestamp or location identity cannot be used and are
    dropped instead of imputed.
    """
    before = len(df)
    df = df.dropna(subset=["last_updated", "location_name", "latitude", "longitude"])
    report["rows_dropped_missing_keys"] = before - len(df)

    missing_before = int(df.isna().sum().sum())
    numeric_cols = df.select_dtypes(include=np.number).columns
    df = df.copy()
    df[numeric_cols] = df.groupby("location_name", observed=True)[numeric_cols].transform(
        lambda series: series.fillna(series.median())
    )
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    object_cols = df.select_dtypes(include="object").columns
    for column in object_cols:
        if df[column].isna().any():
            mode = df[column].mode()
            fill = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill)
    report["missing_values_imputed"] = missing_before
    return df


def cap_outliers_iqr(df: pd.DataFrame, report: dict[str, Any]) -> pd.DataFrame:
    """Winsorize extreme univariate outliers to ``multiplier`` * IQR fences.

    A conservative multiplier (default 3.0) is used so genuine weather
    extremes survive while data glitches are pulled back to the fence.
    Bounded physical quantities keep their hard limits.
    """
    config = load_config()["cleaning"]
    multiplier = float(config["iqr_multiplier"])
    capped_counts: dict[str, int] = {}
    df = df.copy()
    for column in OUTLIER_FEATURES:
        series = df[column]
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        hard = config["valid_ranges"].get(column)
        if hard is not None:
            lower, upper = max(lower, hard[0]), min(upper, hard[1])
        outside = int(((series < lower) | (series > upper)).sum())
        if outside:
            df[column] = series.clip(lower, upper)
        capped_counts[column] = outside
    report["iqr_values_capped_by_column"] = capped_counts
    return df


def flag_outliers_isolation_forest(
    df: pd.DataFrame, report: dict[str, Any]
) -> pd.DataFrame:
    """Add a boolean ``outlier_iforest`` column flagging multivariate outliers.

    Flagged rows are retained so downstream anomaly analysis can study
    them; model-training stages exclude them explicitly.
    """
    config = load_config()
    params = config["cleaning"]["isolation_forest"]
    seed = config["project"]["random_seed"]
    model = IsolationForest(
        contamination=params["contamination"],
        n_estimators=params["n_estimators"],
        random_state=seed,
        n_jobs=-1,
    )
    features = df[OUTLIER_FEATURES].to_numpy()
    df = df.copy()
    df["outlier_iforest"] = model.fit_predict(features) == -1
    report["isolation_forest_flagged"] = int(df["outlier_iforest"].sum())
    save_model(model, "isolation_forest_cleaning")
    return df


def build_scaled_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Create the ML-ready dataset: scaled numerics + encoded categoricals.

    The fitted scaler and label encoders are persisted so the dashboard
    and inference code can invert the transformations.
    """
    numeric_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in ("latitude", "longitude")
    ]
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df[numeric_cols]),
        columns=numeric_cols,
        index=df.index,
    )
    scaled[["latitude", "longitude"]] = df[["latitude", "longitude"]]
    scaled["last_updated"] = df["last_updated"].to_numpy()
    scaled["outlier_iforest"] = df["outlier_iforest"].to_numpy()

    encoders: dict[str, LabelEncoder] = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = LabelEncoder()
        scaled[f"{column}_encoded"] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder
    save_model(scaler, "standard_scaler")
    save_model(encoders, "label_encoders")
    return scaled


def run_preprocessing() -> pd.DataFrame:
    """Execute the full cleaning pipeline and persist all artifacts."""
    config = load_config()
    report: dict[str, Any] = {}

    with timed_step(logger, "load raw data"):
        df = load_raw_data()
        report["raw_shape"] = list(df.shape)

    with timed_step(logger, "clean and validate"):
        df = drop_redundant_columns(df)
        df = remove_duplicates(df, report)
        df = validate_physical_ranges(df, report)
        df = handle_missing_values(df, report)
        df = cap_outliers_iqr(df, report)

    with timed_step(logger, "isolation forest outlier flagging"):
        df = flag_outliers_isolation_forest(df, report)

    with timed_step(logger, "scale and encode"):
        scaled = build_scaled_dataset(df)

    df = df.sort_values(["location_name", "last_updated"]).reset_index(drop=True)
    clean_path = resolve_path(config["paths"]["processed_data"])
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(clean_path, index=False)
    scaled.to_parquet(resolve_path(config["paths"]["scaled_data"]), index=False)

    report["clean_shape"] = list(df.shape)
    save_json(report, "cleaning_report", directory_key="reports_dir")
    logger.info("Cleaning report: %s", report)
    return df


if __name__ == "__main__":
    run_preprocessing()
