"""Ensemble learning over the individual forecasting models.

Combines the per-model one-step-ahead test predictions produced by the
forecasting stage using three strategies:

* **Voting**            - simple average of the top-N base models.
* **Weighted average**  - weights proportional to inverse RMSE on a
  blend window.
* **Stacking**          - a Ridge meta-learner trained on base-model
  predictions from the blend window.

To keep the comparison fair the held-out window is split in two: the
first third is the *blend window* (used only to pick models, weights and
the meta-learner), the remaining two thirds is the *evaluation window*
on which ensembles and individual models are compared.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.utils import (
    CATEGORICAL_PALETTE,
    load_config,
    regression_metrics,
    resolve_path,
    save_figure,
    save_json,
    save_model,
    setup_logger,
    timed_step,
)

logger = setup_logger("ensemble")

ENSEMBLE_NAMES = ("Voting", "WeightedAverage", "Stacking")


def load_base_predictions(target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the forecasting stage's predictions for one target.

    Returns:
        A (dates x models) prediction matrix and the aligned actuals.
    """
    models_dir = resolve_path(load_config()["paths"]["models_dir"])
    frame = pd.read_parquet(models_dir / "forecast_predictions.parquet")
    frame = frame[frame["target"] == target]
    matrix = frame.pivot(index="date", columns="model", values="y_pred")
    actual = (
        frame.drop_duplicates("date").set_index("date")["y_true"].loc[matrix.index]
    )
    return matrix, actual


def split_blend_eval(
    matrix: pd.DataFrame, actual: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split predictions into blend (first third) and evaluation windows."""
    cut = len(matrix) // 3
    return matrix.iloc[:cut], actual.iloc[:cut], matrix.iloc[cut:], actual.iloc[cut:]


def select_top_models(
    blend_matrix: pd.DataFrame, blend_actual: pd.Series, top_n: int
) -> list[str]:
    """Rank base models by blend-window RMSE and keep the best ``top_n``."""
    rmse = {
        model: float(np.sqrt(np.mean((blend_actual - blend_matrix[model]) ** 2)))
        for model in blend_matrix.columns
    }
    ranked = sorted(rmse, key=rmse.get)
    return ranked[:top_n]


def voting_forecast(eval_matrix: pd.DataFrame, top_models: list[str]) -> np.ndarray:
    """Simple (unweighted) average of the top base models."""
    return eval_matrix[top_models].mean(axis=1).to_numpy()


def weighted_average_forecast(
    blend_matrix: pd.DataFrame, blend_actual: pd.Series,
    eval_matrix: pd.DataFrame, top_models: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """Inverse-RMSE weighted average of the top base models."""
    inverse_rmse = {
        model: 1.0
        / float(np.sqrt(np.mean((blend_actual - blend_matrix[model]) ** 2)))
        for model in top_models
    }
    total = sum(inverse_rmse.values())
    weights = {model: value / total for model, value in inverse_rmse.items()}
    combined = sum(
        eval_matrix[model].to_numpy() * weight for model, weight in weights.items()
    )
    return combined, weights


def stacking_forecast(
    blend_matrix: pd.DataFrame, blend_actual: pd.Series,
    eval_matrix: pd.DataFrame, target: str,
) -> np.ndarray:
    """Ridge meta-learner stacked on all base-model predictions."""
    meta = Ridge(alpha=1.0)
    meta.fit(blend_matrix.to_numpy(), blend_actual.to_numpy())
    save_model(meta, f"stacking_meta_{target}")
    return meta.predict(eval_matrix.to_numpy())


def plot_comparison(
    metrics: dict[str, dict[str, float]], target: str
) -> None:
    """Bar chart of evaluation-window RMSE, ensembles highlighted."""
    rmse = pd.Series(
        {name: values["RMSE"] for name, values in metrics.items()}
    ).sort_values()
    colors = [
        CATEGORICAL_PALETTE[1] if name in ENSEMBLE_NAMES else CATEGORICAL_PALETTE[0]
        for name in rmse.index
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(rmse.index, rmse.to_numpy(), color=colors)
    ax.set_ylabel("RMSE (evaluation window)")
    ax.set_title(f"Individual models vs ensembles — {target}")
    ax.tick_params(axis="x", rotation=25)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL_PALETTE[0]),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL_PALETTE[1]),
    ]
    ax.legend(handles, ["Individual model", "Ensemble"])
    save_figure(fig, f"ensemble_comparison_{target}")
    plt.close(fig)


def ensemble_target(target: str) -> dict[str, Any]:
    """Build and evaluate all ensembles for a single target."""
    top_n = load_config()["ensemble"]["top_n_models"]
    matrix, actual = load_base_predictions(target)
    blend_m, blend_a, eval_m, eval_a = split_blend_eval(matrix, actual)
    top_models = select_top_models(blend_m, blend_a, top_n)
    logger.info("%s top models: %s", target, top_models)

    predictions: dict[str, np.ndarray] = {
        model: eval_m[model].to_numpy() for model in matrix.columns
    }
    predictions["Voting"] = voting_forecast(eval_m, top_models)
    weighted, weights = weighted_average_forecast(
        blend_m, blend_a, eval_m, top_models
    )
    predictions["WeightedAverage"] = weighted
    predictions["Stacking"] = stacking_forecast(blend_m, blend_a, eval_m, target)

    metrics = {
        name: regression_metrics(eval_a.to_numpy(), preds)
        for name, preds in predictions.items()
    }
    plot_comparison(metrics, target)
    best = min(metrics, key=lambda name: metrics[name]["RMSE"])
    return {
        "top_models": top_models,
        "weights": weights,
        "metrics": metrics,
        "best_model": best,
        "eval_window_days": int(len(eval_a)),
    }


def run_ensemble() -> None:
    """Execute the ensemble stage for every forecasting target."""
    results: dict[str, Any] = {}
    for target in load_config()["forecasting"]["targets"]:
        with timed_step(logger, f"ensembles for {target}"):
            results[target] = ensemble_target(target)
        logger.info(
            "%s best on evaluation window: %s", target, results[target]["best_model"]
        )
    save_json(results, "ensemble_results", directory_key="reports_dir")
    logger.info("Ensemble stage complete.")


if __name__ == "__main__":
    run_ensemble()
