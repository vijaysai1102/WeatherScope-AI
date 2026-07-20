"""Air quality analysis: pollutant levels and weather interactions.

Analyzes CO, O3, NO2, SO2, PM2.5 and PM10 together with the US EPA AQI
category, producing:

* Pollutant distributions (log-count histograms — concentrations are
  heavy-tailed).
* EPA AQI category breakdown.
* Pollutant vs weather correlation heatmap (static + interactive).
* Wind speed vs PM2.5 dispersion curve — the classic ventilation effect.
* Most and least polluted cities and continent-level comparison.
* Monthly PM2.5 trend.

Key numbers are saved to ``outputs/reports/air_quality_summary.json``.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src.utils import (
    CATEGORICAL_PALETTE,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    add_continent_column,
    load_config,
    resolve_path,
    save_figure,
    save_json,
    save_plotly,
    setup_logger,
    timed_step,
)

logger = setup_logger("air_quality")

#: Human-readable pollutant labels keyed by dataset column.
POLLUTANT_LABELS: dict[str, str] = {
    "air_quality_Carbon_Monoxide": "CO (µg/m³)",
    "air_quality_Ozone": "O₃ (µg/m³)",
    "air_quality_Nitrogen_dioxide": "NO₂ (µg/m³)",
    "air_quality_Sulphur_dioxide": "SO₂ (µg/m³)",
    "air_quality_PM2.5": "PM2.5 (µg/m³)",
    "air_quality_PM10": "PM10 (µg/m³)",
}

#: US EPA AQI category names indexed by the dataset's 1-6 code.
EPA_CATEGORIES: dict[int, str] = {
    1: "Good",
    2: "Moderate",
    3: "Unhealthy (sensitive)",
    4: "Unhealthy",
    5: "Very unhealthy",
    6: "Hazardous",
}

MAJOR_CONTINENTS: list[str] = ["Asia", "Africa", "Europe", "Americas", "Oceania"]


def plot_pollutant_distributions(df: pd.DataFrame) -> None:
    """Log-count histograms of the six pollutant concentrations."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (column, label) in zip(axes.flat, POLLUTANT_LABELS.items()):
        ax.hist(df[column], bins=80, color=CATEGORICAL_PALETTE[0])
        ax.set_yscale("log")
        ax.set_title(label)
        ax.set_ylabel("Count (log)")
    fig.suptitle("Pollutant concentration distributions", fontsize=15, y=1.0)
    fig.tight_layout()
    save_figure(fig, "air_pollutant_distributions")
    plt.close(fig)


def plot_epa_breakdown(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Share of observations per US EPA AQI category."""
    counts = df["air_quality_us-epa-index"].map(EPA_CATEGORIES).value_counts()
    counts = counts.reindex([EPA_CATEGORIES[i] for i in range(1, 7)]).fillna(0)
    share = counts / counts.sum() * 100.0
    summary["epa_category_share_pct"] = share.round(2).to_dict()
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(share.index, share.to_numpy(), color=CATEGORICAL_PALETTE[0])
    ax.set_ylabel("Share of observations (%)")
    ax.set_title("US EPA air quality index breakdown")
    ax.tick_params(axis="x", rotation=20)
    save_figure(fig, "air_epa_breakdown")
    plt.close(fig)


def plot_weather_correlations(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Correlation heatmap between pollutants and weather variables."""
    config = load_config()["air_quality"]
    pollutants, weather_vars = config["pollutants"], config["weather_vars"]
    corr = (
        df[pollutants + weather_vars].corr().loc[pollutants, weather_vars]
    )
    corr.index = [POLLUTANT_LABELS[p] for p in corr.index]
    summary["pollutant_weather_correlations"] = corr.round(3).to_dict()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0.0,
        vmin=-0.5, vmax=0.5, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Pollutant vs weather correlations")
    save_figure(fig, "air_weather_correlations")
    plt.close(fig)

    fig_px = px.imshow(
        corr.round(2), text_auto=True, zmin=-0.5, zmax=0.5,
        color_continuous_scale="RdBu_r",
        title="Pollutant vs weather correlations (interactive)",
    )
    fig_px.update_layout(template="plotly_white")
    save_plotly(fig_px, "air_weather_correlations_interactive")


def plot_wind_dispersion(df: pd.DataFrame) -> None:
    """Mean PM2.5 by wind-speed band: atmospheric ventilation effect."""
    bands = pd.cut(df["wind_kph"], bins=[0, 5, 10, 15, 20, 30, 60])
    curve = df.groupby(bands, observed=True)["air_quality_PM2.5"].mean()
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(
        [str(interval) for interval in curve.index], curve.to_numpy(),
        marker="o", color=CATEGORICAL_PALETTE[0], lw=2,
    )
    ax.set_title("Wind speed disperses particulate matter")
    ax.set_xlabel("Wind speed band (kph)")
    ax.set_ylabel("Mean PM2.5 (µg/m³)")
    save_figure(fig, "air_wind_dispersion")
    plt.close(fig)


def plot_city_rankings(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Most polluted cities by mean PM2.5 (static + interactive)."""
    by_city = df.groupby("location_name").agg(
        pm25=("air_quality_PM2.5", "mean"), n=("location_name", "size")
    )
    ranked = by_city[by_city["n"] >= 100]["pm25"].nlargest(15)
    summary["most_polluted_cities_pm25"] = ranked.round(1).to_dict()
    colors = sns.color_palette(SEQUENTIAL_CMAP, n_colors=len(ranked))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ranked.index[::-1], ranked.to_numpy()[::-1], color=colors[::-1])
    ax.set_title("Most polluted cities (mean PM2.5)")
    ax.set_xlabel("PM2.5 (µg/m³)")
    save_figure(fig, "air_city_rankings")
    plt.close(fig)

    fig_px = px.bar(
        ranked.iloc[::-1], orientation="h",
        labels={"value": "PM2.5 (µg/m³)", "location_name": "City"},
        title="Most polluted cities (mean PM2.5)",
        color=ranked.iloc[::-1].to_numpy(),
        color_continuous_scale=SEQUENTIAL_CMAP,
    )
    fig_px.update_layout(showlegend=False, template="plotly_white")
    save_plotly(fig_px, "air_city_rankings_interactive")


def plot_continent_and_trend(df: pd.DataFrame) -> None:
    """Continent-level PM2.5 comparison and global monthly PM2.5 trend."""
    major = df[df["continent"].isin(MAJOR_CONTINENTS)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    means = (
        major.groupby("continent")["air_quality_PM2.5"]
        .mean()
        .reindex(MAJOR_CONTINENTS)
    )
    axes[0].bar(means.index, means.to_numpy(), color=CATEGORICAL_PALETTE[0])
    axes[0].set_title("Mean PM2.5 by continent")
    axes[0].set_ylabel("PM2.5 (µg/m³)")
    monthly = (
        df.set_index("last_updated")["air_quality_PM2.5"].resample("MS").mean()
    )
    axes[1].plot(
        monthly.index, monthly.to_numpy(), marker="o",
        color=CATEGORICAL_PALETTE[1], lw=2,
    )
    axes[1].set_title("Global monthly mean PM2.5")
    axes[1].set_ylabel("PM2.5 (µg/m³)")
    save_figure(fig, "air_continent_and_trend")
    plt.close(fig)


def run_air_quality() -> None:
    """Execute the air-quality stage end to end."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["processed_data"]))
    df = add_continent_column(df)

    summary: dict[str, Any] = {}
    steps = [
        lambda: plot_pollutant_distributions(df),
        lambda: plot_epa_breakdown(df, summary),
        lambda: plot_weather_correlations(df, summary),
        lambda: plot_wind_dispersion(df),
        lambda: plot_city_rankings(df, summary),
        lambda: plot_continent_and_trend(df),
    ]
    names = [
        "pollutant distributions", "EPA breakdown", "weather correlations",
        "wind dispersion", "city rankings", "continent and trend",
    ]
    for name, step in zip(names, steps):
        with timed_step(logger, name):
            step()
    save_json(summary, "air_quality_summary", directory_key="reports_dir")
    logger.info("Air quality stage complete.")


if __name__ == "__main__":
    run_air_quality()
