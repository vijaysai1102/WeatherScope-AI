"""Spatial analysis: interactive maps and geographical patterns.

Uses per-location aggregates (latitude/longitude plus mean weather and
air-quality values) to produce:

* Folium temperature and rainfall heatmaps.
* A Folium country-level temperature choropleth (world GeoJSON is
  downloaded once and cached under ``data/raw``).
* A Plotly country choropleth (name-based, no GeoJSON required).
* An air-quality (PM2.5) circle map.
* A KMeans weather-cluster map grouping cities into climate clusters.
* Geographical pattern figures: latitude vs temperature and climate-zone
  comparisons (altitude is not available in this dataset).
* Country-level summary table saved to ``outputs/reports``.

All interactive maps are written to ``outputs/figures/maps``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from branca.colormap import LinearColormap
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils import (
    CATEGORICAL_PALETTE,
    SEQUENTIAL_CMAP,
    add_continent_column,
    ensure_dir,
    load_config,
    resolve_path,
    save_figure,
    save_plotly,
    setup_logger,
    timed_step,
)

logger = setup_logger("spatial")

#: Dataset country names that differ from the world GeoJSON naming.
COUNTRY_ALIASES: dict[str, str] = {
    "United States Of America": "United States of America",
    "USA United States of America": "United States of America",
    "United States": "United States of America",
    "Russia": "Russia",
    "Tanzania": "United Republic of Tanzania",
    "Congo": "Republic of the Congo",
    "Democratic Republic of Congo": "Democratic Republic of the Congo",
    "Serbia": "Republic of Serbia",
    "Czech Republic": "Czech Republic",
    "North Macedonia": "Macedonia",
    "South Korea": "South Korea",
    "Guinea Bissau": "Guinea Bissau",
}

#: Climate zone by absolute latitude band.
ZONE_BINS: list[float] = [0.0, 23.5, 35.0, 55.0, 90.0]
ZONE_LABELS: list[str] = ["Tropical", "Subtropical", "Temperate", "Polar"]


def _maps_dir() -> Path:
    """Directory for interactive map HTML files."""
    return ensure_dir(Path(load_config()["paths"]["figures_dir"]) / "maps")


def fetch_world_geojson() -> Path:
    """Download (once) and return the cached world-countries GeoJSON path."""
    config = load_config()["spatial"]
    path = resolve_path(config["world_geojson_path"])
    if not path.exists():
        logger.info("Downloading world GeoJSON to %s", path)
        urllib.request.urlretrieve(config["world_geojson_url"], path)
    return path


def build_location_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-location aggregates used by every map."""
    summary = df.groupby("location_name").agg(
        country=("country", "first"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
        temperature=("temperature_celsius", "mean"),
        humidity=("humidity", "mean"),
        precip=("precip_mm", "mean"),
        wind=("wind_kph", "mean"),
        pm25=("air_quality_PM2.5", "mean"),
        observations=("location_name", "size"),
    )
    return summary[summary["observations"] >= 50].reset_index()


def build_country_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Country-level aggregates; also saved as a report artifact."""
    summary = df.groupby("country").agg(
        temperature=("temperature_celsius", "mean"),
        humidity=("humidity", "mean"),
        precip=("precip_mm", "mean"),
        pm25=("air_quality_PM2.5", "mean"),
        observations=("country", "size"),
    )
    summary = summary[summary["observations"] >= 100].round(2).reset_index()
    reports_dir = resolve_path(load_config()["paths"]["reports_dir"])
    summary.to_csv(reports_dir / "country_summaries.csv", index=False)
    return summary


def _base_map() -> folium.Map:
    """A neutral world basemap."""
    return folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")


def map_heatmaps(locations: pd.DataFrame) -> None:
    """Folium heatmaps: temperature intensity and rainfall intensity."""
    for column, name in [("temperature", "heatmap_temperature"),
                         ("precip", "heatmap_rainfall")]:
        values = locations[column] - locations[column].min()
        peak = values.max() or 1.0
        world = _base_map()
        HeatMap(
            list(
                zip(
                    locations["latitude"], locations["longitude"],
                    (values / peak).tolist(),
                )
            ),
            radius=18, blur=22,
        ).add_to(world)
        world.save(_maps_dir() / f"{name}.html")


def map_temperature_circles(locations: pd.DataFrame) -> None:
    """Circle map of mean temperature per city with a shared colormap."""
    colormap = LinearColormap(
        ["#440154", "#31688e", "#35b779", "#fde725"],
        vmin=float(locations["temperature"].min()),
        vmax=float(locations["temperature"].max()),
        caption="Mean temperature (°C)",
    )
    world = _base_map()
    for _, row in locations.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=None,
            fill=True,
            fill_color=colormap(row["temperature"]),
            fill_opacity=0.85,
            tooltip=(
                f"{row['location_name']} ({row['country']}): "
                f"{row['temperature']:.1f} °C"
            ),
        ).add_to(world)
    colormap.add_to(world)
    world.save(_maps_dir() / "temperature_map.html")


def map_air_quality(locations: pd.DataFrame) -> None:
    """Circle map of mean PM2.5 per city."""
    colormap = LinearColormap(
        ["#029e73", "#ece133", "#d55e00", "#5d1a78"],
        vmin=0.0,
        vmax=float(locations["pm25"].quantile(0.95)),
        caption="Mean PM2.5 (µg/m³)",
    )
    world = _base_map()
    for _, row in locations.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=None,
            fill=True,
            fill_color=colormap(min(row["pm25"], colormap.vmax)),
            fill_opacity=0.85,
            tooltip=(
                f"{row['location_name']} ({row['country']}): "
                f"PM2.5 {row['pm25']:.0f}"
            ),
        ).add_to(world)
    colormap.add_to(world)
    world.save(_maps_dir() / "air_quality_map.html")


def map_choropleth(countries: pd.DataFrame) -> None:
    """Country temperature choropleths: Folium (GeoJSON) and Plotly (names)."""
    geojson_path = fetch_world_geojson()
    data = countries.copy()
    data["geo_name"] = data["country"].replace(COUNTRY_ALIASES)
    world = _base_map()
    folium.Choropleth(
        geo_data=str(geojson_path),
        data=data,
        columns=["geo_name", "temperature"],
        key_on="feature.properties.name",
        fill_color="YlOrRd",
        nan_fill_color="#d5d8dc",
        fill_opacity=0.8,
        line_opacity=0.3,
        legend_name="Mean temperature (°C)",
    ).add_to(world)
    world.save(_maps_dir() / "choropleth_temperature.html")

    fig = px.choropleth(
        countries,
        locations="country",
        locationmode="country names",
        color="temperature",
        color_continuous_scale=SEQUENTIAL_CMAP,
        labels={"temperature": "Mean temp (°C)"},
        title="Mean temperature by country",
    )
    fig.update_layout(template="plotly_white")
    save_plotly(fig, "choropleth_temperature_plotly")

    fig_rain = px.choropleth(
        countries,
        locations="country",
        locationmode="country names",
        color="precip",
        color_continuous_scale="Blues",
        labels={"precip": "Mean precip (mm)"},
        title="Mean precipitation by country",
    )
    fig_rain.update_layout(template="plotly_white")
    save_plotly(fig_rain, "choropleth_rainfall_plotly")


def map_weather_clusters(locations: pd.DataFrame) -> pd.DataFrame:
    """KMeans climate clusters over city-level aggregates, mapped in Folium."""
    config = load_config()
    k = config["spatial"]["cluster_count"]
    features = locations[["temperature", "humidity", "precip", "wind", "pm25"]]
    matrix = StandardScaler().fit_transform(features.fillna(features.mean()))
    model = KMeans(
        n_clusters=k, n_init=10, random_state=config["project"]["random_seed"]
    )
    locations = locations.copy()
    locations["cluster"] = model.fit_predict(matrix)

    # Rank clusters by mean temperature for stable, interpretable naming.
    order = (
        locations.groupby("cluster")["temperature"].mean().sort_values().index
    )
    rank_of = {cluster: rank for rank, cluster in enumerate(order)}
    locations["cluster"] = locations["cluster"].map(rank_of)

    world = _base_map()
    for _, row in locations.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=None,
            fill=True,
            fill_color=CATEGORICAL_PALETTE[int(row["cluster"]) % len(CATEGORICAL_PALETTE)],
            fill_opacity=0.9,
            tooltip=f"{row['location_name']}: cluster {int(row['cluster'])}",
        ).add_to(world)
    world.save(_maps_dir() / "weather_clusters_map.html")

    profile = (
        locations.groupby("cluster")[
            ["temperature", "humidity", "precip", "wind", "pm25"]
        ]
        .mean()
        .round(2)
    )
    reports_dir = resolve_path(config["paths"]["reports_dir"])
    profile.to_csv(reports_dir / "weather_cluster_profiles.csv")
    logger.info("Cluster profiles:\n%s", profile)
    return locations


def plot_latitude_effect(locations: pd.DataFrame) -> None:
    """Mean temperature vs latitude with a quadratic fit."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.scatter(
        locations["latitude"], locations["temperature"],
        s=18, alpha=0.7, color=CATEGORICAL_PALETTE[0], label="City",
    )
    coefficients = np.polyfit(
        locations["latitude"], locations["temperature"], deg=2
    )
    grid = np.linspace(
        locations["latitude"].min(), locations["latitude"].max(), 200
    )
    ax.plot(
        grid, np.polyval(coefficients, grid),
        color=CATEGORICAL_PALETTE[1], lw=2.5, label="Quadratic fit",
    )
    ax.set_title("Latitude effect on mean temperature")
    ax.set_xlabel("Latitude (°)")
    ax.set_ylabel("Mean temperature (°C)")
    ax.legend()
    save_figure(fig, "spatial_latitude_effect")
    plt.close(fig)


def plot_climate_zones(locations: pd.DataFrame) -> None:
    """Temperature and rainfall by absolute-latitude climate zone."""
    zones = pd.cut(
        locations["latitude"].abs(), bins=ZONE_BINS, labels=ZONE_LABELS,
        include_lowest=True,
    )
    frame = locations.assign(zone=zones)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    frame.boxplot(
        column="temperature", by="zone", ax=axes[0], grid=False,
        boxprops={"color": CATEGORICAL_PALETTE[0]},
        medianprops={"color": CATEGORICAL_PALETTE[1]},
    )
    axes[0].set_title("Temperature by climate zone")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Mean temperature (°C)")
    means = frame.groupby("zone", observed=True)["precip"].mean()
    axes[1].bar(
        means.index.astype(str), means.to_numpy(), color=CATEGORICAL_PALETTE[0]
    )
    axes[1].set_title("Mean precipitation by climate zone")
    axes[1].set_ylabel("Precipitation (mm)")
    fig.suptitle("")
    save_figure(fig, "spatial_climate_zones")
    plt.close(fig)


def run_spatial_analysis() -> None:
    """Execute the spatial stage end to end."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["processed_data"]))
    df = add_continent_column(df)
    locations = build_location_summary(df)
    countries = build_country_summary(df)
    logger.info(
        "Locations: %d, countries: %d", len(locations), len(countries)
    )

    steps = [
        lambda: map_heatmaps(locations),
        lambda: map_temperature_circles(locations),
        lambda: map_air_quality(locations),
        lambda: map_choropleth(countries),
        lambda: map_weather_clusters(locations),
        lambda: plot_latitude_effect(locations),
        lambda: plot_climate_zones(locations),
    ]
    names = [
        "heatmaps", "temperature circles", "air quality map", "choropleths",
        "weather clusters", "latitude effect", "climate zones",
    ]
    for name, step in zip(names, steps):
        with timed_step(logger, name):
            step()
    logger.info("Spatial stage complete; maps in outputs/figures/maps.")


if __name__ == "__main__":
    run_spatial_analysis()
