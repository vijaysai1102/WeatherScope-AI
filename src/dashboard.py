"""Streamlit dashboard for the WeatherScope AI project.

Eight pages — Overview, EDA, Forecasting, Maps, Feature Importance,
Climate Analysis, Air Quality and Model Comparison — driven by the
artifacts the pipeline writes to ``data/processed`` and ``outputs``.

Sidebar filters (country, city, date range and forecast horizon) apply
to every page that shows observation-level data. Launch with:

    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


from src.utils import (
    CATEGORICAL_PALETTE,
    add_continent_column,
    load_config,
    resolve_path,
)

PRIMARY = CATEGORICAL_PALETTE[0]

PAGES = [
    "Overview",
    "EDA",
    "Forecasting",
    "Maps",
    "Feature Importance",
    "Climate Analysis",
    "Air Quality",
    "Model Comparison",
]

TARGET_LABELS = {
    "temperature_celsius": "Temperature (°C)",
    "humidity": "Humidity (%)",
    "precip_mm": "Precipitation (mm)",
}


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    """Cleaned observation-level dataset with continent column."""
    df = pd.read_parquet(resolve_path(load_config()["paths"]["processed_data"]))
    return add_continent_column(df)


@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame:
    """Global daily-mean target series."""
    return pd.read_parquet(resolve_path("data/processed/daily_series.parquet"))


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    """Holdout predictions of every forecasting model."""
    models_dir = resolve_path(load_config()["paths"]["models_dir"])
    return pd.read_parquet(models_dir / "forecast_predictions.parquet")


@st.cache_data(show_spinner=False)
def load_future(target: str) -> pd.DataFrame:
    """Prophet future forecast (with CI) for one target."""
    models_dir = resolve_path(load_config()["paths"]["models_dir"])
    return pd.read_parquet(models_dir / f"future_forecast_{target}.parquet")


@st.cache_data(show_spinner=False)
def load_report_json(name: str) -> dict[str, Any]:
    """A JSON artifact from ``outputs/reports``."""
    path = resolve_path(load_config()["paths"]["reports_dir"]) / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def figures_dir() -> Path:
    """Absolute path of the figures directory."""
    return resolve_path(load_config()["paths"]["figures_dir"])


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


def sidebar_filters(df: pd.DataFrame) -> dict[str, Any]:
    """Render sidebar controls and return the chosen filter values."""
    st.sidebar.title("🌍 WeatherScope AI")
    page = st.sidebar.radio("Page", PAGES)
    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    countries = ["All"] + sorted(df["country"].unique())
    country = st.sidebar.selectbox("Country", countries)
    scope = df if country == "All" else df[df["country"] == country]
    cities = ["All"] + sorted(scope["location_name"].unique())
    city = st.sidebar.selectbox("City", cities)

    min_day = df["last_updated"].min().date()
    max_day = df["last_updated"].max().date()
    date_range = st.sidebar.slider(
        "Date range", min_value=min_day, max_value=max_day,
        value=(min_day, max_day),
    )
    horizon = st.sidebar.slider("Forecast horizon (days)", 7, 30, 14)
    return {
        "page": page, "country": country, "city": city,
        "date_range": date_range, "horizon": horizon,
    }


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Subset the observation-level frame according to sidebar filters."""
    result = df
    if filters["country"] != "All":
        result = result[result["country"] == filters["country"]]
    if filters["city"] != "All":
        result = result[result["location_name"] == filters["city"]]
    start, end = filters["date_range"]
    stamp = result["last_updated"].dt.date
    return result[(stamp >= start) & (stamp <= end)]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Headline metrics, station map and dataset description."""
    st.title("Global Weather Trend Forecasting")
    st.caption(
        "End-to-end analysis and forecasting of the Global Weather "
        "Repository — data cleaning, EDA, anomaly detection, six "
        "forecasting models, ensembles, explainability, climate, air "
        "quality and spatial analysis."
    )
    tiles = st.columns(5)
    tiles[0].metric("Observations", f"{len(df):,}")
    tiles[1].metric("Countries", f"{df['country'].nunique()}")
    tiles[2].metric("Cities", f"{df['location_name'].nunique()}")
    tiles[3].metric("Mean temp", f"{df['temperature_celsius'].mean():.1f} °C")
    tiles[4].metric("Mean PM2.5", f"{df['air_quality_PM2.5'].mean():.0f} µg/m³")

    st.subheader("Reporting stations")
    stations = (
        df.groupby("location_name")[["latitude", "longitude"]].median().dropna()
    )
    st.map(stations, size=18000)

    st.subheader("Most common weather conditions")
    counts = df["condition_text"].str.strip().value_counts().head(12).iloc[::-1]
    fig = px.bar(
        counts, orientation="h", color_discrete_sequence=[PRIMARY],
        labels={"value": "Observations", "condition_text": "Condition"},
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=420)
    st.plotly_chart(fig, width="stretch")


def page_eda(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Interactive distributions, trends and correlation views."""
    st.title("Exploratory Data Analysis")
    variable = st.selectbox(
        "Variable", list(TARGET_LABELS) + ["wind_kph", "pressure_mb", "uv_index"],
        format_func=lambda v: TARGET_LABELS.get(v, v),
    )
    left, right = st.columns(2)
    with left:
        fig = px.histogram(
            df, x=variable, nbins=60, color_discrete_sequence=[PRIMARY],
            title="Distribution",
        )
        fig.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig, width="stretch")
    with right:
        daily = (
            df.set_index("last_updated")[variable].resample("D").mean().dropna()
        )
        fig = px.line(
            daily, color_discrete_sequence=[CATEGORICAL_PALETTE[1]],
            title="Daily mean over time",
            labels={"value": variable, "last_updated": "Date"},
        )
        fig.update_layout(template="plotly_white", height=380, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Monthly temperature by continent")
    major = df[df["continent"].isin(["Asia", "Africa", "Europe", "Americas", "Oceania"])]
    monthly = (
        major.groupby([major["last_updated"].dt.month, "continent"])
        ["temperature_celsius"].mean().rename("temp").reset_index()
    )
    fig = px.line(
        monthly, x="last_updated", y="temp", color="continent", markers=True,
        color_discrete_sequence=CATEGORICAL_PALETTE,
        labels={"last_updated": "Month", "temp": "Temperature (°C)"},
    )
    fig.update_layout(template="plotly_white", xaxis={"dtick": 1}, height=420)
    st.plotly_chart(fig, width="stretch")


def _plot_future(future: pd.DataFrame, daily: pd.Series, horizon: int, label: str) -> go.Figure:
    """Recent actuals plus the Prophet outlook clipped to the chosen horizon."""
    cutoff = daily.index.max()
    outlook = future[future["ds"] > cutoff].head(horizon)
    recent = daily.iloc[-120:]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=recent.index, y=recent, mode="lines", name="Actual",
                   line={"color": "#6b7075", "width": 2})
    )
    fig.add_trace(
        go.Scatter(x=outlook["ds"], y=outlook["yhat_upper"], mode="lines",
                   line={"width": 0}, showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(x=outlook["ds"], y=outlook["yhat_lower"], mode="lines",
                   fill="tonexty", fillcolor="rgba(1,115,178,0.18)",
                   line={"width": 0}, name="95% interval")
    )
    fig.add_trace(
        go.Scatter(x=outlook["ds"], y=outlook["yhat"], mode="lines",
                   name="Forecast", line={"color": PRIMARY, "width": 2.5})
    )
    fig.update_layout(
        template="plotly_white", height=430, yaxis_title=label,
        legend={"orientation": "h", "y": -0.2},
    )
    return fig


def page_forecasting(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Holdout comparison and future outlook per target."""
    st.title("Forecasting")
    target = st.selectbox(
        "Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get
    )
    label = TARGET_LABELS[target]
    daily = load_daily()[target]

    st.subheader(f"{filters['horizon']}-day outlook (Prophet, 95% CI)")
    st.plotly_chart(
        _plot_future(load_future(target), daily, filters["horizon"], label),
        width="stretch",
    )

    st.subheader("Holdout window: actual vs one-step-ahead predictions")
    predictions = load_predictions()
    subset = predictions[predictions["target"] == target]
    models = st.multiselect(
        "Models", sorted(subset["model"].unique()),
        default=["ARIMA", "XGBoost", "Prophet"],
    )
    fig = go.Figure()
    actual = subset.drop_duplicates("date")
    fig.add_trace(
        go.Scatter(x=actual["date"], y=actual["y_true"], mode="lines",
                   name="Actual", line={"color": "#6b7075", "width": 2.5})
    )
    for model, color in zip(models, CATEGORICAL_PALETTE):
        rows = subset[subset["model"] == model]
        fig.add_trace(
            go.Scatter(x=rows["date"], y=rows["y_pred"], mode="lines",
                       name=model, line={"color": color, "width": 1.8, "dash": "dot"})
        )
    fig.update_layout(template="plotly_white", height=430, yaxis_title=label)
    st.plotly_chart(fig, width="stretch")

    metrics = load_report_json("forecast_metrics")[target]
    table = pd.DataFrame(metrics).T.sort_values("RMSE").round(3)
    st.subheader("Model metrics (holdout)")
    st.dataframe(table, use_container_width=True)


def page_maps(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Embedded interactive Folium maps — generated on-the-fly from data."""
    import folium
    from branca.colormap import LinearColormap
    from folium.plugins import HeatMap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from streamlit_folium import st_folium

    st.title("Interactive Maps")

    # ------------------------------------------------------------------
    # Build per-location summary from the (filtered) clean data
    # ------------------------------------------------------------------
    subset = apply_filters(df, filters)
    locations = (
        subset.groupby("location_name")
        .agg(
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
        .query("observations >= 10")
        .reset_index()
    )

    MAP_CHOICES = [
        "Temperature map",
        "Temperature heatmap",
        "Rainfall heatmap",
        "Country choropleth (temperature)",
        "Air quality (PM2.5)",
        "Weather clusters",
    ]
    choice = st.selectbox("Map", MAP_CHOICES)

    def _base() -> folium.Map:
        return folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")

    # ------------------------------------------------------------------
    if choice == "Temperature map":
        cmap = LinearColormap(
            ["#440154", "#31688e", "#35b779", "#fde725"],
            vmin=float(locations["temperature"].min()),
            vmax=float(locations["temperature"].max()),
            caption="Mean temperature (°C)",
        )
        m = _base()
        for _, r in locations.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5, color=None, fill=True,
                fill_color=cmap(r["temperature"]), fill_opacity=0.85,
                tooltip=f"{r['location_name']} ({r['country']}): {r['temperature']:.1f} °C",
            ).add_to(m)
        cmap.add_to(m)
        st_folium(m, width=700, height=500, returned_objects=[])

    # ------------------------------------------------------------------
    elif choice == "Temperature heatmap":
        m = _base()
        vals = locations["temperature"] - locations["temperature"].min()
        peak = vals.max() or 1.0
        HeatMap(
            list(zip(locations["latitude"], locations["longitude"],
                     (vals / peak).tolist())),
            radius=18, blur=22,
        ).add_to(m)
        st_folium(m, width=700, height=500, returned_objects=[])

    # ------------------------------------------------------------------
    elif choice == "Rainfall heatmap":
        m = _base()
        vals = locations["precip"] - locations["precip"].min()
        peak = vals.max() or 1.0
        HeatMap(
            list(zip(locations["latitude"], locations["longitude"],
                     (vals / peak).tolist())),
            radius=18, blur=22,
        ).add_to(m)
        st_folium(m, width=700, height=500, returned_objects=[])

    # ------------------------------------------------------------------
    elif choice == "Country choropleth (temperature)":
        countries = (
            subset.groupby("country")
            .agg(temperature=("temperature_celsius", "mean"))
            .reset_index()
        )
        fig = px.choropleth(
            countries, locations="country", locationmode="country names",
            color="temperature", color_continuous_scale="YlOrRd",
            labels={"temperature": "Mean temp (°C)"},
            title="Mean temperature by country",
        )
        fig.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig, width="stretch")

    # ------------------------------------------------------------------
    elif choice == "Air quality (PM2.5)":
        cmap = LinearColormap(
            ["#029e73", "#ece133", "#d55e00", "#5d1a78"],
            vmin=0.0,
            vmax=float(locations["pm25"].quantile(0.95)),
            caption="Mean PM2.5 (µg/m³)",
        )
        m = _base()
        for _, r in locations.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5, color=None, fill=True,
                fill_color=cmap(min(r["pm25"], cmap.vmax)),
                fill_opacity=0.85,
                tooltip=f"{r['location_name']} ({r['country']}): PM2.5 {r['pm25']:.0f}",
            ).add_to(m)
        cmap.add_to(m)
        st_folium(m, width=700, height=500, returned_objects=[])

    # ------------------------------------------------------------------
    elif choice == "Weather clusters":
        k = 5
        features = locations[["temperature", "humidity", "precip", "wind", "pm25"]]
        matrix = StandardScaler().fit_transform(features.fillna(features.mean()))
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        locations = locations.copy()
        locations["cluster"] = km.fit_predict(matrix)
        order = locations.groupby("cluster")["temperature"].mean().sort_values().index
        rank_of = {c: r for r, c in enumerate(order)}
        locations["cluster"] = locations["cluster"].map(rank_of)

        palette = CATEGORICAL_PALETTE
        m = _base()
        for _, r in locations.iterrows():
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=5, color=None, fill=True,
                fill_color=palette[int(r["cluster"]) % len(palette)],
                fill_opacity=0.9,
                tooltip=f"{r['location_name']}: cluster {int(r['cluster'])}",
            ).add_to(m)
        st_folium(m, width=700, height=500, returned_objects=[])

        # Show cluster profiles
        profile = (
            locations.groupby("cluster")[
                ["temperature", "humidity", "precip", "wind", "pm25"]
            ].mean().round(2)
        )
        st.subheader("Cluster profiles (city-level means)")
        st.dataframe(profile, use_container_width=True)


def page_importance(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Feature importance and SHAP explanation figures."""
    st.title("Feature Importance")
    st.caption(
        "Random Forest and XGBoost trained to predict same-day temperature "
        "from engineered features (leakage-prone features excluded)."
    )
    images = {
        "Random Forest importance": "importance_random_forest.png",
        "XGBoost importance": "importance_xgboost.png",
        "Permutation importance": "importance_permutation.png",
        "SHAP summary (beeswarm)": "shap_summary.png",
        "SHAP top-20 bar": "shap_top20_bar.png",
        "SHAP dependence plot": "shap_dependence.png",
    }
    tabs = st.tabs(list(images))
    for tab, filename in zip(tabs, images.values()):
        with tab:
            path = figures_dir() / filename
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                st.warning("Run `python main.py --stage importance` first.")


def page_climate(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Climate trend figures with headline numbers."""
    st.title("Climate Analysis")
    summary = load_report_json("climate_summary")
    tiles = st.columns(3)
    slope = summary.get("stl_trend_slope_c_per_year")
    tiles[0].metric("STL trend (obs. window)", f"{slope:+.2f} °C/yr")
    warm = summary.get("largest_warm_anomaly", {})
    tiles[1].metric(
        "Largest warm anomaly", f"{warm.get('value', 0):+.1f} °C",
        warm.get("date", ""), delta_color="off",
    )
    cold = summary.get("largest_cold_anomaly", {})
    tiles[2].metric(
        "Largest cold anomaly", f"{cold.get('value', 0):+.1f} °C",
        cold.get("date", ""), delta_color="off",
    )
    st.info(summary.get("note", ""))
    for filename in [
        "climate_yearly_averages.png", "climate_seasonal_shifts.png",
        "climate_rainfall_changes.png", "climate_regional_comparison.png",
        "climate_temperature_anomalies.png", "climate_stl_decomposition.png",
    ]:
        path = figures_dir() / filename
        if path.exists():
            st.image(str(path), width="stretch")


def page_air_quality(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """Air-quality insights on the filtered data."""
    st.title("Air Quality")
    tiles = st.columns(4)
    tiles[0].metric("Mean PM2.5", f"{df['air_quality_PM2.5'].mean():.1f}")
    tiles[1].metric("Mean PM10", f"{df['air_quality_PM10'].mean():.1f}")
    tiles[2].metric("Mean O₃", f"{df['air_quality_Ozone'].mean():.1f}")
    tiles[3].metric("Mean NO₂", f"{df['air_quality_Nitrogen_dioxide'].mean():.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("PM2.5 vs wind speed")
        bands = pd.cut(df["wind_kph"], bins=[0, 5, 10, 15, 20, 30, 60])
        curve = df.groupby(bands, observed=True)["air_quality_PM2.5"].mean().dropna()
        if curve.empty:
            st.info("Not enough data for the current filters.")
        else:
            fig = px.line(
                x=[str(i) for i in curve.index], y=curve.to_numpy(),
                markers=True,
                labels={"x": "Wind band (kph)", "y": "Mean PM2.5"},
                color_discrete_sequence=[PRIMARY],
            )
            fig.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Most polluted cities (filtered)")
        ranked = (
            df.groupby("location_name")["air_quality_PM2.5"].mean().nlargest(12)
        )
        fig = px.bar(
            ranked.iloc[::-1], orientation="h",
            color_discrete_sequence=[CATEGORICAL_PALETTE[1]],
            labels={"value": "Mean PM2.5", "location_name": "City"},
        )
        fig.update_layout(showlegend=False, template="plotly_white", height=380)
        st.plotly_chart(fig, width="stretch")

    st.image(
        str(figures_dir() / "air_weather_correlations.png"),
        caption="Pollutant vs weather correlations (full dataset)",
        width="stretch",
    )


def page_model_comparison(df: pd.DataFrame, filters: dict[str, Any]) -> None:
    """All-model and ensemble comparison across targets."""
    st.title("Model Comparison")
    metrics = load_report_json("forecast_metrics")
    ensembles = load_report_json("ensemble_results")

    target = st.selectbox(
        "Target", list(TARGET_LABELS), format_func=TARGET_LABELS.get
    )
    combined = dict(metrics[target])
    combined.update(
        {
            name: values
            for name, values in ensembles[target]["metrics"].items()
            if name in ("Voting", "WeightedAverage", "Stacking")
        }
    )
    table = pd.DataFrame(combined).T.sort_values("RMSE").round(3)
    best = table.index[0]
    st.subheader("Holdout metrics (individual models and ensembles)")
    st.dataframe(
        table.style.highlight_min(
            subset=["MAE", "RMSE", "MAPE"], color="#c8e6c9"
        ).highlight_max(subset=["R2"], color="#c8e6c9"),
        use_container_width=True,
    )
    st.success(
        f"Best on the evaluation window: **{best}** "
        f"(RMSE {table.loc[best, 'RMSE']:.3f}, R² {table.loc[best, 'R2']:.3f})."
    )
    fig = px.bar(
        table["RMSE"], color_discrete_sequence=[PRIMARY],
        labels={"value": "RMSE", "index": "Model"},
    )
    fig.update_layout(showlegend=False, template="plotly_white", height=380)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Statistical models excel because the globally averaged daily series "
        "is dominated by smooth seasonal structure and strong day-to-day "
        "persistence; ensembles further reduce variance by combining "
        "complementary error profiles."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_PAGE_RENDERERS = {
    "Overview": page_overview,
    "EDA": page_eda,
    "Forecasting": page_forecasting,
    "Maps": page_maps,
    "Feature Importance": page_importance,
    "Climate Analysis": page_climate,
    "Air Quality": page_air_quality,
    "Model Comparison": page_model_comparison,
}


def main() -> None:
    """Configure the app, apply filters and dispatch to the chosen page."""
    st.set_page_config(
        page_title="WeatherScope AI", page_icon="🌍", layout="wide"
    )
    df = load_clean()
    filters = sidebar_filters(df)
    filtered = apply_filters(df, filters)
    if filtered.empty:
        st.warning("No data matches the current filters.")
        return
    _PAGE_RENDERERS[filters["page"]](filtered, filters)


if __name__ == "__main__":
    main()
