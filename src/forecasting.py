"""Time series forecasting of global daily weather aggregates.

Builds coverage-corrected daily series for temperature, humidity and
precipitation, then fits six forecasting models per target:

Evaluation protocol: **rolling one-step-ahead forecasts** over the held-out
window, mirroring operational use (forecast tomorrow, observe, repeat):

* ARIMA / SARIMA: parameters are estimated on the train window only, then
  the fitted model filters through the test window (``append`` without
  refitting) producing true one-step-ahead predictions with intervals.
* XGBoost / Random Forest / LightGBM: trained on lag/rolling/calendar
  features from the train window; test features are built from *actual*
  history so each prediction is one-step-ahead.
* Prophet is a curve-fitting model without an autoregressive component,
  so its test-window prediction is the same at any horizon; it is fitted
  on train only and evaluated on the test dates.

Artifacts written:

* ``data/processed/daily_series.parquet``       - the daily target frame.
* ``outputs/models/forecast_predictions.parquet`` - long-format test
  predictions (with confidence intervals where available).
* ``outputs/reports/forecast_metrics.json``     - MAE/RMSE/MAPE/R² tables.
* ``outputs/models/future_forecast_<target>.parquet`` - Prophet forecasts
  ``horizon_days`` beyond the last observation, with uncertainty bands.
* Per-model fitted artifacts for the dashboard.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

from src.utils import (
    CATEGORICAL_PALETTE,
    load_config,
    regression_metrics,
    resolve_path,
    save_figure,
    save_json,
    save_model,
    save_plotly,
    setup_logger,
    timed_step,
)

logger = setup_logger("forecasting")
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

STATISTICAL_MODELS = ("ARIMA", "SARIMA", "Prophet")
ML_MODELS = ("XGBoost", "RandomForest", "LightGBM")
ALL_MODELS = STATISTICAL_MODELS + ML_MODELS


def build_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the panel to a global daily-mean frame per target.

    Days with unusually low station coverage (fewer than half the median
    daily observation count) produce biased means, so they are masked and
    time-interpolated instead.
    """
    targets: list[str] = load_config()["forecasting"]["targets"]
    grouped = df.set_index("last_updated").resample("D")
    daily = grouped[targets].mean()
    coverage = grouped.size()
    threshold = coverage.median() * 0.5
    daily[coverage < threshold] = np.nan
    daily = daily.interpolate(method="time").dropna()
    daily.index.name = "date"
    return daily


def build_city_series(
    df: pd.DataFrame, city: str, target: str
) -> pd.Series:
    """Return an interpolated daily series of ``target`` for one city."""
    city_df = df[df["location_name"] == city]
    series = (
        city_df.set_index("last_updated")[target]
        .resample("D")
        .mean()
        .interpolate(method="time")
        .dropna()
    )
    series.index.name = "date"
    return series


def split_series(series: pd.Series, test_days: int) -> tuple[pd.Series, pd.Series]:
    """Split a daily series into train and test (the last ``test_days``)."""
    return series.iloc[:-test_days], series.iloc[-test_days:]


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------


def _one_step_bands(
    fitted: Any, test: pd.Series
) -> tuple[np.ndarray, pd.DataFrame]:
    """Filter a fitted statsmodels result through the test window.

    ``append(..., refit=False)`` keeps the train-estimated parameters and
    produces genuine one-step-ahead predictions with intervals.
    """
    extended = fitted.append(test, refit=False)
    prediction = extended.get_prediction(
        start=test.index[0], end=test.index[-1], dynamic=False
    )
    interval = prediction.conf_int(alpha=0.05)
    bands = pd.DataFrame(
        {"lo": interval.iloc[:, 0].to_numpy(), "hi": interval.iloc[:, 1].to_numpy()},
        index=test.index,
    )
    return np.asarray(prediction.predicted_mean), bands


def fit_predict_arima(
    train: pd.Series, test: pd.Series
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """One-step-ahead forecasts from a non-seasonal ARIMA model."""
    order = tuple(load_config()["forecasting"]["arima_order"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(train, order=order).fit()
        return _one_step_bands(fitted, test)


def fit_predict_sarima(
    train: pd.Series, test: pd.Series
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """One-step-ahead forecasts from a weekly-seasonal SARIMA model."""
    config = load_config()["forecasting"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            train,
            order=tuple(config["sarima_order"]),
            seasonal_order=tuple(config["sarima_seasonal_order"]),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return _one_step_bands(fitted, test)


def _make_prophet_frame(series: pd.Series) -> pd.DataFrame:
    """Convert a series to the two-column frame Prophet expects."""
    return pd.DataFrame({"ds": series.index, "y": series.to_numpy()})


def fit_predict_prophet(
    train: pd.Series, test: pd.Series
) -> tuple[np.ndarray, pd.DataFrame | None]:
    """Forecast the test window with Prophet (weekly + yearly seasonality)."""
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_make_prophet_frame(train))
    prediction = model.predict(pd.DataFrame({"ds": test.index}))
    bands = pd.DataFrame(
        {
            "lo": prediction["yhat_lower"].to_numpy(),
            "hi": prediction["yhat_upper"].to_numpy(),
        },
        index=test.index,
    )
    return prediction["yhat"].to_numpy(), bands


# ---------------------------------------------------------------------------
# Machine-learning models with recursive multi-step forecasting
# ---------------------------------------------------------------------------


def _feature_row(
    history: np.ndarray, date: pd.Timestamp, lags: list[int], windows: list[int]
) -> dict[str, float]:
    """Build one feature row from a value history and a calendar date."""
    row: dict[str, float] = {}
    for lag in lags:
        row[f"lag{lag}"] = history[-lag]
    for window in windows:
        tail = history[-window:]
        row[f"rollmean{window}"] = float(np.mean(tail))
        row[f"rollstd{window}"] = float(np.std(tail))
    day_of_year = date.dayofyear
    row["month"] = date.month
    row["day_of_week"] = date.dayofweek
    row["doy_sin"] = float(np.sin(2 * np.pi * day_of_year / 365.25))
    row["doy_cos"] = float(np.cos(2 * np.pi * day_of_year / 365.25))
    return row


def make_supervised(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Turn a daily series into a supervised lag/rolling/calendar dataset."""
    config = load_config()["forecasting"]
    lags: list[int] = config["ml_lag_days"]
    windows: list[int] = config["ml_rolling_windows"]
    start = max(lags)
    values = series.to_numpy()
    rows = [
        _feature_row(values[:position], series.index[position], lags, windows)
        for position in range(start, len(series))
    ]
    features = pd.DataFrame(rows, index=series.index[start:])
    return features, series.iloc[start:]


def _build_ml_model(name: str, seed: int) -> Any:
    """Instantiate an ML regressor by name with sensible defaults."""
    if name == "XGBoost":
        return XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=-1, verbosity=0,
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
    if name == "LightGBM":
        return LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31,
            subsample=0.9, colsample_bytree=0.9, random_state=seed,
            n_jobs=-1, verbose=-1,
        )
    raise ValueError(f"Unknown ML model: {name}")


def fit_predict_ml(
    name: str, train: pd.Series, test: pd.Series
) -> tuple[np.ndarray, None]:
    """Train an ML regressor and produce one-step-ahead test forecasts.

    The model is fitted only on train-window rows; test-window features
    are built from *actual* history (never from the model's own output),
    so each prediction is a true one-step-ahead forecast — the same
    information regime as the filtered statistical models.
    """
    config = load_config()
    model = _build_ml_model(name, config["project"]["random_seed"])
    full = pd.concat([train, test])
    features, target = make_supervised(full)
    is_train = features.index < test.index[0]
    model.fit(features[is_train], target[is_train])
    predictions = model.predict(features[~is_train])
    save_model(model, f"{name.lower()}_{test.name}")
    return np.asarray(predictions), None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_FITTERS: dict[str, Callable[[pd.Series, pd.Series], tuple[np.ndarray, Any]]] = {
    "ARIMA": fit_predict_arima,
    "SARIMA": fit_predict_sarima,
    "Prophet": fit_predict_prophet,
    "XGBoost": fit_predict_ml,
    "RandomForest": fit_predict_ml,
    "LightGBM": fit_predict_ml,
}


def forecast_target(
    daily: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Fit every model for one target; return predictions and metrics."""
    test_days = load_config()["forecasting"]["test_days"]
    series = daily[target].rename(target)
    train, test = split_series(series, test_days)
    records: list[pd.DataFrame] = []
    metrics: dict[str, dict[str, float]] = {}
    for name in ALL_MODELS:
        fitter = _FITTERS[name]
        with timed_step(logger, f"{name} on {target}"):
            if name in ML_MODELS:
                preds, bands = fitter(name, train, test)
            else:
                preds, bands = fitter(train, test)
        metrics[name] = regression_metrics(test.to_numpy(), preds)
        frame = pd.DataFrame(
            {
                "date": test.index,
                "target": target,
                "model": name,
                "y_true": test.to_numpy(),
                "y_pred": preds,
                "lo": bands["lo"].to_numpy() if bands is not None else np.nan,
                "hi": bands["hi"].to_numpy() if bands is not None else np.nan,
            }
        )
        records.append(frame)
        logger.info("%s %s: %s", target, name, metrics[name])
    return pd.concat(records, ignore_index=True), metrics


def fit_future_prophet(daily: pd.DataFrame, target: str) -> None:
    """Refit Prophet on the full series and save a future forecast with CI."""
    horizon = load_config()["forecasting"]["horizon_days"]
    series = daily[target]
    model = Prophet(
        weekly_seasonality=True, yearly_seasonality=True,
        daily_seasonality=False, interval_width=0.95,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_make_prophet_frame(series))
    future = model.make_future_dataframe(periods=horizon, freq="D")
    forecast = model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    models_dir = resolve_path(load_config()["paths"]["models_dir"])
    forecast.to_parquet(models_dir / f"future_forecast_{target}.parquet", index=False)
    save_model(model, f"prophet_full_{target}")


def plot_predictions(predictions: pd.DataFrame, target: str) -> None:
    """Small-multiples of actual vs predicted for every model (one target)."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, ALL_MODELS):
        subset = predictions[predictions["model"] == name]
        ax.plot(
            subset["date"], subset["y_true"], color="#6b7075", lw=1.6,
            label="Actual",
        )
        ax.plot(
            subset["date"], subset["y_pred"], color=CATEGORICAL_PALETTE[0],
            lw=1.8, label="Predicted",
        )
        if subset["lo"].notna().any():
            ax.fill_between(
                subset["date"], subset["lo"], subset["hi"],
                color=CATEGORICAL_PALETTE[0], alpha=0.15, label="95% CI",
            )
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"Actual vs predicted — {target} (60-day holdout)", fontsize=15)
    fig.tight_layout()
    save_figure(fig, f"forecast_{target}")
    plt.close(fig)


def plot_predictions_interactive(
    daily: pd.DataFrame, predictions: pd.DataFrame, target: str,
    metrics: dict[str, dict[str, float]],
) -> None:
    """Interactive comparison: recent actuals plus every model's forecast."""
    recent = daily[target].iloc[-180:]
    fig = go.Figure(
        go.Scatter(
            x=recent.index, y=recent.to_numpy(), mode="lines",
            name="Actual", line={"color": "#6b7075", "width": 2},
        )
    )
    ranked = sorted(ALL_MODELS, key=lambda name: metrics[name]["RMSE"])
    palette = CATEGORICAL_PALETTE + ["#555b61"]
    for name, color in zip(ranked, palette):
        subset = predictions[predictions["model"] == name]
        fig.add_trace(
            go.Scatter(
                x=subset["date"], y=subset["y_pred"], mode="lines",
                name=f"{name} (RMSE {metrics[name]['RMSE']:.2f})",
                line={"color": color, "width": 2, "dash": "dot"},
            )
        )
    fig.update_layout(
        title=f"Forecast comparison — {target}",
        yaxis_title=target, template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
    )
    save_plotly(fig, f"forecast_{target}_interactive")


def run_forecasting() -> None:
    """Execute the full forecasting stage for every configured target."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["processed_data"]))
    daily = build_daily_frame(df)
    daily.to_parquet(
        resolve_path("data/processed/daily_series.parquet")
    )
    logger.info(
        "Daily series: %d days (%s to %s)",
        len(daily), daily.index.min().date(), daily.index.max().date(),
    )

    all_predictions: list[pd.DataFrame] = []
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for target in config["forecasting"]["targets"]:
        predictions, metrics = forecast_target(daily, target)
        all_predictions.append(predictions)
        all_metrics[target] = metrics
        plot_predictions(predictions, target)
        plot_predictions_interactive(daily, predictions, target, metrics)
        with timed_step(logger, f"Prophet future forecast for {target}"):
            fit_future_prophet(daily, target)

    combined = pd.concat(all_predictions, ignore_index=True)
    models_dir = resolve_path(config["paths"]["models_dir"])
    combined.to_parquet(models_dir / "forecast_predictions.parquet", index=False)
    save_json(all_metrics, "forecast_metrics", directory_key="reports_dir")
    logger.info("Forecasting stage complete.")


if __name__ == "__main__":
    run_forecasting()
