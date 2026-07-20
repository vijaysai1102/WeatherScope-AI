"""Exploratory data analysis: static and interactive visualizations.

Generates every required EDA figure — distributions, correlations,
temporal trends, geographic comparisons and city rankings — as PNGs
(matplotlib/seaborn) and interactive HTML (Plotly) under
``outputs/figures``.

Design rules follow the project visualization standard: a fixed,
CVD-validated categorical palette, a perceptually uniform sequential
colormap for magnitudes, a diverging colormap centred at zero for
correlations, and one axis per chart.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from src.utils import (
    CATEGORICAL_PALETTE,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    add_continent_column,
    load_config,
    resolve_path,
    save_figure,
    save_plotly,
    setup_logger,
    timed_step,
)

logger = setup_logger("eda")

#: (column, label) pairs for the univariate distribution grid.
DISTRIBUTION_VARS: list[tuple[str, str]] = [
    ("temperature_celsius", "Temperature (°C)"),
    ("humidity", "Humidity (%)"),
    ("precip_mm", "Precipitation (mm)"),
    ("wind_kph", "Wind speed (kph)"),
    ("pressure_mb", "Pressure (mb)"),
    ("visibility_km", "Visibility (km)"),
    ("uv_index", "UV index"),
    ("cloud", "Cloud cover (%)"),
]

#: Numeric columns used for the correlation heatmap.
CORRELATION_COLS: list[str] = [
    "temperature_celsius",
    "feels_like_celsius",
    "humidity",
    "precip_mm",
    "wind_kph",
    "gust_kph",
    "pressure_mb",
    "cloud",
    "uv_index",
    "visibility_km",
    "air_quality_PM2.5",
    "air_quality_Ozone",
]

SEASON_ORDER: list[str] = ["Winter", "Spring", "Summer", "Autumn"]

#: Major continents kept as distinct series (rest folded into aggregates).
MAJOR_CONTINENTS: list[str] = ["Asia", "Africa", "Europe", "Americas", "Oceania"]

PRIMARY = CATEGORICAL_PALETTE[0]


def _apply_style() -> None:
    """Set the global seaborn/matplotlib style for all static figures."""
    sns.set_theme(style="whitegrid", palette=CATEGORICAL_PALETTE)
    plt.rcParams.update({"figure.autolayout": False, "axes.titleweight": "bold"})


def plot_distributions(df: pd.DataFrame) -> None:
    """Histogram grid of the eight core weather variables."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, (column, label) in zip(axes.flat, DISTRIBUTION_VARS):
        sns.histplot(df[column], bins=60, ax=ax, color=PRIMARY, edgecolor="none")
        ax.set_title(label)
        ax.set_xlabel("")
        if column == "precip_mm":
            ax.set_yscale("log")
            ax.set_title(f"{label} — log count")
    fig.suptitle("Distributions of core weather variables", fontsize=16, y=1.02)
    fig.tight_layout()
    save_figure(fig, "distributions_grid")
    plt.close(fig)


def plot_weather_conditions(df: pd.DataFrame) -> None:
    """Horizontal bar chart of the 15 most frequent weather conditions."""
    counts = df["condition_text"].str.strip().value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(counts.index[::-1], counts.to_numpy()[::-1], color=PRIMARY)
    ax.set_title("Top 15 weather conditions")
    ax.set_xlabel("Observations")
    save_figure(fig, "weather_conditions")
    plt.close(fig)

    fig_px = px.bar(
        counts.iloc[::-1],
        orientation="h",
        labels={"value": "Observations", "condition_text": "Condition"},
        title="Top 15 weather conditions",
        color_discrete_sequence=[PRIMARY],
    )
    fig_px.update_layout(showlegend=False, template="plotly_white")
    save_plotly(fig_px, "weather_conditions_interactive")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap (static + interactive) of key numeric variables."""
    corr = df[CORRELATION_COLS].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0.0,
        vmin=-1.0, vmax=1.0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Correlation heatmap of weather and air-quality variables")
    save_figure(fig, "correlation_heatmap")
    plt.close(fig)

    fig_px = px.imshow(
        corr.round(2), text_auto=True, zmin=-1.0, zmax=1.0,
        color_continuous_scale="RdBu_r",
        title="Correlation heatmap (interactive)",
    )
    fig_px.update_layout(template="plotly_white")
    save_plotly(fig_px, "correlation_heatmap_interactive")


def plot_monthly_trends(df: pd.DataFrame) -> None:
    """Average temperature, humidity and precipitation by calendar month."""
    monthly = df.groupby("month")[
        ["temperature_celsius", "humidity", "precip_mm"]
    ].mean()
    labels = ["Temperature (°C)", "Humidity (%)", "Precipitation (mm)"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, column, label, color in zip(
        axes, monthly.columns, labels, CATEGORICAL_PALETTE[:3]
    ):
        ax.plot(monthly.index, monthly[column], marker="o", color=color, lw=2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title("Global monthly weather trends")
    axes[-1].set_xlabel("Month")
    axes[-1].set_xticks(range(1, 13))
    save_figure(fig, "monthly_trends")
    plt.close(fig)


def plot_seasonal_trends(df: pd.DataFrame) -> None:
    """Temperature spread and mean precipitation by (hemisphere-aware) season."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(
        data=df, x="season", y="temperature_celsius", order=SEASON_ORDER,
        color=PRIMARY, showfliers=False, ax=axes[0],
    )
    axes[0].set_title("Temperature by season")
    axes[0].set_ylabel("Temperature (°C)")
    seasonal_precip = (
        df.groupby("season")["precip_mm"].mean().reindex(SEASON_ORDER)
    )
    axes[1].bar(seasonal_precip.index, seasonal_precip.to_numpy(), color=PRIMARY)
    axes[1].set_title("Mean precipitation by season")
    axes[1].set_ylabel("Precipitation (mm)")
    save_figure(fig, "seasonal_trends")
    plt.close(fig)


def _ranked_barh(
    series: pd.Series, title: str, xlabel: str, name: str
) -> None:
    """Save a magnitude-ranked horizontal bar chart with sequential coloring."""
    fig, ax = plt.subplots(figsize=(10, 7))
    values = series.iloc[::-1]
    colors = sns.color_palette(SEQUENTIAL_CMAP, n_colors=len(values))
    ax.barh(values.index.astype(str), values.to_numpy(), color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    save_figure(fig, name)
    plt.close(fig)


def plot_country_comparisons(df: pd.DataFrame) -> None:
    """Country-level rankings: hottest, coldest and wettest countries."""
    by_country = df.groupby("country").agg(
        temp=("temperature_celsius", "mean"),
        precip=("precip_mm", "mean"),
        n=("country", "size"),
    )
    by_country = by_country[by_country["n"] >= 100]
    _ranked_barh(
        by_country["temp"].nlargest(15),
        "Hottest countries (mean temperature)", "Temperature (°C)",
        "countries_hottest",
    )
    _ranked_barh(
        by_country["temp"].nsmallest(15).iloc[::-1],
        "Coldest countries (mean temperature)", "Temperature (°C)",
        "countries_coldest",
    )
    _ranked_barh(
        by_country["precip"].nlargest(15),
        "Wettest countries (mean precipitation)", "Precipitation (mm)",
        "countries_wettest",
    )


def plot_continent_comparison(df: pd.DataFrame) -> None:
    """Temperature spread and mean precipitation across continents."""
    major = df[df["continent"].isin(MAJOR_CONTINENTS)]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(
        data=major, x="continent", y="temperature_celsius",
        order=MAJOR_CONTINENTS, color=PRIMARY, showfliers=False, ax=axes[0],
    )
    axes[0].set_title("Temperature by continent")
    axes[0].set_ylabel("Temperature (°C)")
    precip = (
        major.groupby("continent")["precip_mm"].mean().reindex(MAJOR_CONTINENTS)
    )
    axes[1].bar(precip.index, precip.to_numpy(), color=PRIMARY)
    axes[1].set_title("Mean precipitation by continent")
    axes[1].set_ylabel("Precipitation (mm)")
    save_figure(fig, "continent_comparison")
    plt.close(fig)


def plot_top_cities(df: pd.DataFrame) -> None:
    """Hottest and coldest cities by mean temperature (static + interactive)."""
    by_city = df.groupby("location_name").agg(
        temp=("temperature_celsius", "mean"), n=("location_name", "size")
    )
    by_city = by_city[by_city["n"] >= 100]["temp"]
    hottest, coldest = by_city.nlargest(10), by_city.nsmallest(10)
    _ranked_barh(
        hottest, "Top 10 hottest cities", "Mean temperature (°C)",
        "cities_hottest",
    )
    _ranked_barh(
        coldest.iloc[::-1], "Top 10 coldest cities",
        "Mean temperature (°C)", "cities_coldest",
    )
    ranked = pd.concat([hottest, coldest.iloc[::-1]])
    fig_px = px.bar(
        ranked, orientation="v", color=ranked.to_numpy(),
        color_continuous_scale=SEQUENTIAL_CMAP,
        labels={"value": "Mean temperature (°C)", "location_name": "City"},
        title="Hottest and coldest cities (mean temperature)",
    )
    fig_px.update_layout(showlegend=False, template="plotly_white")
    save_plotly(fig_px, "cities_temperature_interactive")


def plot_daily_averages(df: pd.DataFrame) -> None:
    """Global daily mean temperature, humidity and precipitation series."""
    daily = (
        df.set_index("last_updated")
        .resample("D")[["temperature_celsius", "humidity", "precip_mm"]]
        .mean()
    )
    labels = ["Temperature (°C)", "Humidity (%)", "Precipitation (mm)"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, column, label, color in zip(
        axes, daily.columns, labels, CATEGORICAL_PALETTE[:3]
    ):
        ax.plot(daily.index, daily[column], color=color, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title("Global daily averages")
    save_figure(fig, "daily_averages")
    plt.close(fig)


def plotly_daily_temperature(df: pd.DataFrame) -> None:
    """Interactive global daily mean temperature with a range slider."""
    daily = (
        df.set_index("last_updated")["temperature_celsius"].resample("D").mean()
    )
    fig = go.Figure(
        go.Scatter(
            x=daily.index, y=daily.to_numpy(), mode="lines",
            line={"color": PRIMARY, "width": 2}, name="Daily mean",
        )
    )
    fig.update_layout(
        title="Global daily mean temperature",
        yaxis_title="Temperature (°C)",
        xaxis={"rangeslider": {"visible": True}},
        template="plotly_white",
    )
    save_plotly(fig, "daily_temperature_interactive")


def plotly_monthly_by_continent(df: pd.DataFrame) -> None:
    """Interactive monthly temperature lines per major continent."""
    major = df[df["continent"].isin(MAJOR_CONTINENTS)]
    monthly = (
        major.groupby(["month", "continent"])["temperature_celsius"]
        .mean()
        .reset_index()
    )
    fig = px.line(
        monthly, x="month", y="temperature_celsius", color="continent",
        category_orders={"continent": MAJOR_CONTINENTS},
        color_discrete_sequence=CATEGORICAL_PALETTE, markers=True,
        labels={"temperature_celsius": "Temperature (°C)", "month": "Month"},
        title="Monthly mean temperature by continent",
    )
    fig.update_layout(template="plotly_white", xaxis={"dtick": 1})
    save_plotly(fig, "monthly_temperature_by_continent")


def run_eda() -> None:
    """Generate and save all EDA figures."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["features_data"]))
    df = add_continent_column(df)
    _apply_style()

    steps = [
        plot_distributions,
        plot_weather_conditions,
        plot_correlation_heatmap,
        plot_monthly_trends,
        plot_seasonal_trends,
        plot_country_comparisons,
        plot_continent_comparison,
        plot_top_cities,
        plot_daily_averages,
        plotly_daily_temperature,
        plotly_monthly_by_continent,
    ]
    for step in steps:
        with timed_step(logger, step.__name__):
            step(df)
    logger.info("EDA complete; figures written to outputs/figures.")


if __name__ == "__main__":
    run_eda()
