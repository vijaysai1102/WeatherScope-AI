"""Climate analysis: long-term trends in the observation period.

The dataset spans roughly May 2024 - July 2026, so "long-term" here means
multi-season structure rather than multi-decade climate change. The stage
produces:

* Yearly average temperature (computed from monthly means so partially
  observed years are not biased toward their observed season).
* Rainfall changes across months and years.
* Seasonal shift curves (month-by-month temperature per year).
* Regional (continent-level) climate comparison over time.
* Daily temperature anomalies vs a day-of-year climatology baseline.
* STL trend decomposition of the daily global temperature series.

Numeric findings are written to ``outputs/reports/climate_summary.json``.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL

from src.utils import (
    CATEGORICAL_PALETTE,
    add_continent_column,
    load_config,
    resolve_path,
    save_figure,
    save_json,
    setup_logger,
    timed_step,
)

logger = setup_logger("climate")

MAJOR_CONTINENTS: list[str] = ["Asia", "Africa", "Europe", "Americas", "Oceania"]


def _common_months(df: pd.DataFrame) -> list[int]:
    """Months observed in every year of the dataset (for fair comparison)."""
    months_by_year = df.groupby("year")["month"].unique()
    common = set(range(1, 13))
    for months in months_by_year:
        common &= set(months)
    return sorted(common)


def yearly_temperature(df: pd.DataFrame) -> pd.Series:
    """Yearly mean temperature over the months every year has in common.

    The first and last years are only partially observed; restricting to
    common months and averaging monthly means keeps the comparison from
    being biased toward the seasons each year happens to cover.
    """
    months = _common_months(df)
    subset = df[df["month"].isin(months)]
    monthly = subset.groupby(["year", "month"])["temperature_celsius"].mean()
    return monthly.groupby("year").mean()


def plot_yearly_averages(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Bar chart of yearly average temperature and mean precipitation."""
    months = _common_months(df)
    subset = df[df["month"].isin(months)]
    temp = yearly_temperature(df)
    precip = (
        subset.groupby(["year", "month"])["precip_mm"].mean().groupby("year").mean()
    )
    summary["comparison_months"] = months
    summary["yearly_mean_temperature"] = temp.round(2).to_dict()
    summary["yearly_mean_precip_mm"] = precip.round(4).to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].bar(temp.index.astype(str), temp.to_numpy(), color=CATEGORICAL_PALETTE[0])
    axes[0].set_title("Average yearly temperature (month-balanced)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].bar(
        precip.index.astype(str), precip.to_numpy(), color=CATEGORICAL_PALETTE[2]
    )
    axes[1].set_title("Average precipitation by year (month-balanced)")
    axes[1].set_ylabel("Precipitation (mm)")
    for ax in axes:
        ax.set_xlabel("Year")
    save_figure(fig, "climate_yearly_averages")
    plt.close(fig)


def plot_rainfall_changes(df: pd.DataFrame) -> None:
    """Monthly mean precipitation, one line per observation year."""
    monthly = (
        df.groupby(["year", "month"])["precip_mm"].mean().unstack("year")
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for year, color in zip(monthly.columns, CATEGORICAL_PALETTE):
        ax.plot(
            monthly.index, monthly[year], marker="o", color=color,
            label=str(year), lw=2,
        )
    ax.set_title("Rainfall changes: monthly mean precipitation by year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xticks(range(1, 13))
    ax.legend(title="Year")
    save_figure(fig, "climate_rainfall_changes")
    plt.close(fig)


def plot_seasonal_shifts(df: pd.DataFrame) -> None:
    """Month-by-month temperature curves per year to reveal seasonal shifts."""
    monthly = (
        df.groupby(["year", "month"])["temperature_celsius"].mean().unstack("year")
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    for year, color in zip(monthly.columns, CATEGORICAL_PALETTE):
        ax.plot(
            monthly.index, monthly[year], marker="o", color=color,
            label=str(year), lw=2,
        )
    ax.set_title("Seasonal temperature cycle by year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.set_xticks(range(1, 13))
    ax.legend(title="Year")
    save_figure(fig, "climate_seasonal_shifts")
    plt.close(fig)


def plot_regional_comparison(df: pd.DataFrame) -> None:
    """Continent-level monthly temperature series over the whole period."""
    major = df[df["continent"].isin(MAJOR_CONTINENTS)]
    series = (
        major.assign(period=major["last_updated"].dt.to_period("M").dt.to_timestamp())
        .groupby(["period", "continent"])["temperature_celsius"]
        .mean()
        .unstack("continent")[MAJOR_CONTINENTS]
    )
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for continent, color in zip(MAJOR_CONTINENTS, CATEGORICAL_PALETTE):
        ax.plot(series.index, series[continent], label=continent, color=color, lw=2)
    ax.set_title("Regional climate comparison: monthly mean temperature")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(ncol=3)
    save_figure(fig, "climate_regional_comparison")
    plt.close(fig)


def compute_anomalies(daily: pd.Series) -> pd.Series:
    """Daily anomalies vs a smoothed day-of-year climatology baseline."""
    day_of_year = daily.index.dayofyear
    climatology = daily.groupby(day_of_year).mean()
    smooth = (
        pd.concat([climatology.tail(15), climatology, climatology.head(15)])
        .rolling(31, center=True, min_periods=1)
        .mean()
        .iloc[15:-15]
    )
    baseline = pd.Series(smooth.loc[day_of_year].to_numpy(), index=daily.index)
    return daily - baseline


def plot_anomalies(daily: pd.Series, summary: dict[str, Any]) -> None:
    """Diverging bar plot of daily global temperature anomalies."""
    anomalies = compute_anomalies(daily)
    colors = np.where(
        anomalies >= 0, CATEGORICAL_PALETTE[1], CATEGORICAL_PALETTE[0]
    )
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(anomalies.index, anomalies.to_numpy(), color=colors, width=1.0)
    ax.axhline(0.0, color="#6b7075", lw=1)
    ax.set_title("Daily global temperature anomaly vs day-of-year climatology")
    ax.set_ylabel("Anomaly (°C)")
    save_figure(fig, "climate_temperature_anomalies")
    plt.close(fig)
    summary["largest_warm_anomaly"] = {
        "date": str(anomalies.idxmax().date()), "value": round(float(anomalies.max()), 2)
    }
    summary["largest_cold_anomaly"] = {
        "date": str(anomalies.idxmin().date()), "value": round(float(anomalies.min()), 2)
    }


def plot_trend_decomposition(daily: pd.Series, summary: dict[str, Any]) -> None:
    """STL decomposition (yearly period) of daily global mean temperature."""
    result = STL(daily, period=365, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 11), sharex=True)
    components = [
        ("Observed", daily), ("Trend", result.trend),
        ("Seasonal", result.seasonal), ("Residual", result.resid),
    ]
    for ax, (label, series), color in zip(
        axes, components, ["#6b7075", *CATEGORICAL_PALETTE[:3]]
    ):
        ax.plot(series.index, series.to_numpy(), color=color, lw=1.2)
        ax.set_ylabel(label)
    axes[0].set_title("STL decomposition of daily global mean temperature")
    save_figure(fig, "climate_stl_decomposition")
    plt.close(fig)

    days = (daily.index - daily.index[0]).days.to_numpy()
    slope, _, _, p_value, _ = stats.linregress(days, result.trend.to_numpy())
    summary["stl_trend_slope_c_per_year"] = round(float(slope * 365.25), 3)
    summary["stl_trend_p_value"] = float(p_value)
    summary["note"] = (
        "Trend estimated over ~26 months of data; it reflects the observation "
        "window (including possible changes in station composition over time), "
        "not multi-decadal climate change."
    )


def run_climate_analysis() -> None:
    """Execute the climate analysis stage end to end."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["features_data"]))
    df = add_continent_column(df)
    daily = pd.read_parquet(
        resolve_path("data/processed/daily_series.parquet")
    )["temperature_celsius"]

    summary: dict[str, Any] = {}
    steps = [
        lambda: plot_yearly_averages(df, summary),
        lambda: plot_rainfall_changes(df),
        lambda: plot_seasonal_shifts(df),
        lambda: plot_regional_comparison(df),
        lambda: plot_anomalies(daily, summary),
        lambda: plot_trend_decomposition(daily, summary),
    ]
    names = [
        "yearly averages", "rainfall changes", "seasonal shifts",
        "regional comparison", "temperature anomalies", "STL decomposition",
    ]
    for name, step in zip(names, steps):
        with timed_step(logger, name):
            step()
    save_json(summary, "climate_summary", directory_key="reports_dir")
    logger.info("Climate summary: %s", summary)


if __name__ == "__main__":
    run_climate_analysis()
