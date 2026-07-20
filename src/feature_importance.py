"""Feature importance analysis for temperature prediction.

Trains Random Forest and XGBoost regressors on the engineered feature
dataset and explains them with four complementary techniques:

* Random Forest impurity importance
* XGBoost gain importance
* Permutation importance (model-agnostic, computed on held-out data)
* SHAP values (TreeExplainer): top-20 bar, beeswarm summary and a
  dependence plot for the strongest feature.

Features derived from the *current-day* target (feels-like temperature
and its offshoots) are excluded to avoid trivially circular importance.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.utils import (
    CATEGORICAL_PALETTE,
    load_config,
    resolve_path,
    save_figure,
    save_json,
    save_model,
    setup_logger,
    timed_step,
)

logger = setup_logger("importance")

#: Columns never used as predictors: identifiers, timestamps, text and
#: any quantity computed from the current-day target itself.
EXCLUDED_COLUMNS: set[str] = {
    "temperature_celsius",
    "feels_like_celsius",
    "temp_feels_diff",
    "humidity_temp_ratio",
    "temp_range_7d",
    "last_updated",
    "outlier_iforest",
    "latitude",
    "longitude",
}


def load_supervised_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the engineered dataset and split into predictors and target."""
    config = load_config()
    target_name = config["feature_importance"]["target"]
    df = pd.read_parquet(resolve_path(config["paths"]["features_data"]))
    df = df[~df["outlier_iforest"]]
    numeric = df.select_dtypes(include=np.number)
    predictors = [col for col in numeric.columns if col not in EXCLUDED_COLUMNS]
    data = numeric[predictors + [target_name]].dropna()
    return data[predictors], data[target_name]


def train_models(
    features: pd.DataFrame, target: pd.Series
) -> tuple[RandomForestRegressor, XGBRegressor, pd.DataFrame, pd.Series]:
    """Fit the two tree ensembles and return them with the test split."""
    seed = load_config()["project"]["random_seed"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=seed
    )
    forest = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed
    )
    booster = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, n_jobs=-1,
        random_state=seed, verbosity=0,
    )
    with timed_step(logger, "fit RandomForest"):
        forest.fit(x_train, y_train)
    with timed_step(logger, "fit XGBoost"):
        booster.fit(x_train, y_train)
    logger.info(
        "Holdout R2 — RandomForest: %.4f, XGBoost: %.4f",
        forest.score(x_test, y_test), booster.score(x_test, y_test),
    )
    save_model(forest, "importance_random_forest")
    save_model(booster, "importance_xgboost")
    return forest, booster, x_test, y_test


def _plot_top_importances(
    values: pd.Series, title: str, filename: str, top_n: int
) -> None:
    """Horizontal bar chart of the ``top_n`` most important features."""
    top = values.nlargest(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top.index, top.to_numpy(), color=CATEGORICAL_PALETTE[0])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    save_figure(fig, filename)
    plt.close(fig)


def compute_importances(
    forest: RandomForestRegressor, booster: XGBRegressor,
    x_test: pd.DataFrame, y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    """Compute and plot RF, XGBoost and permutation importances."""
    config = load_config()["feature_importance"]
    top_n = config["top_n"]
    seed = load_config()["project"]["random_seed"]
    columns = x_test.columns

    rf_importance = pd.Series(forest.feature_importances_, index=columns)
    xgb_importance = pd.Series(booster.feature_importances_, index=columns)
    _plot_top_importances(
        rf_importance, f"Random Forest importance (top {top_n})",
        "importance_random_forest", top_n,
    )
    _plot_top_importances(
        xgb_importance, f"XGBoost importance (top {top_n})",
        "importance_xgboost", top_n,
    )

    with timed_step(logger, "permutation importance"):
        sample = x_test.sample(min(8000, len(x_test)), random_state=seed)
        permutation = permutation_importance(
            booster, sample, y_test.loc[sample.index],
            n_repeats=5, random_state=seed, n_jobs=-1,
        )
    perm_importance = pd.Series(permutation.importances_mean, index=columns)
    _plot_top_importances(
        perm_importance, f"Permutation importance (top {top_n})",
        "importance_permutation", top_n,
    )
    return {
        "random_forest": rf_importance.sort_values(ascending=False).to_dict(),
        "xgboost": xgb_importance.sort_values(ascending=False).to_dict(),
        "permutation": perm_importance.sort_values(ascending=False).to_dict(),
    }


def shap_analysis(booster: XGBRegressor, x_test: pd.DataFrame) -> None:
    """SHAP summary, top-20 bar and dependence plots for the XGBoost model."""
    config = load_config()["feature_importance"]
    seed = load_config()["project"]["random_seed"]
    sample = x_test.sample(
        min(config["shap_sample_size"], len(x_test)), random_state=seed
    )
    with timed_step(logger, "SHAP values"):
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, max_display=20, show=False)
    save_figure(plt.gcf(), "shap_summary")
    plt.close("all")

    shap.summary_plot(
        shap_values, sample, plot_type="bar", max_display=20, show=False
    )
    save_figure(plt.gcf(), "shap_top20_bar")
    plt.close("all")

    strongest = sample.columns[np.abs(shap_values).mean(axis=0).argmax()]
    shap.dependence_plot(strongest, shap_values, sample, show=False)
    save_figure(plt.gcf(), "shap_dependence")
    plt.close("all")
    logger.info("SHAP strongest feature: %s", strongest)


def run_feature_importance() -> None:
    """Execute the feature-importance stage end to end."""
    features, target = load_supervised_data()
    logger.info("Supervised matrix: %s rows, %s features", *features.shape)
    forest, booster, x_test, y_test = train_models(features, target)
    importances = compute_importances(forest, booster, x_test, y_test)
    save_json(importances, "feature_importances", directory_key="reports_dir")
    shap_analysis(booster, x_test)
    logger.info("Feature importance stage complete.")


if __name__ == "__main__":
    run_feature_importance()
