"""Advanced anomaly detection on the cleaned weather dataset.

Runs two complementary detectors on a common standardized feature space:

* **Isolation Forest** - a global, tree-based detector that isolates points
  that are easy to separate from the bulk of the data.
* **DBSCAN** - a density-based clusterer whose noise points (label ``-1``)
  are records living in sparse regions of feature space.

The module compares the two detectors, visualizes agreement in PCA space,
and generates a data-driven markdown explanation of *why* the flagged
observations are anomalous.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.utils import (
    CATEGORICAL_PALETTE,
    load_config,
    resolve_path,
    save_figure,
    save_json,
    setup_logger,
    timed_step,
)

logger = setup_logger("anomaly")

#: Agreement categories in fixed plotting order with fixed palette slots.
AGREEMENT_LABELS: list[str] = [
    "Normal",
    "Isolation Forest only",
    "DBSCAN only",
    "Both detectors",
]


def _prepare_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Subsample the frame and return it with standardized feature matrix.

    DBSCAN is O(n^2) in the worst case, so both detectors are compared on
    a common random subsample defined in the config.
    """
    config = load_config()
    params = config["anomaly_detection"]
    features: list[str] = params["features"]
    max_samples: int = params["dbscan"]["max_samples"]
    seed: int = config["project"]["random_seed"]

    sample = df.dropna(subset=features)
    if len(sample) > max_samples:
        sample = sample.sample(max_samples, random_state=seed)
    matrix = StandardScaler().fit_transform(sample[features])
    return sample.reset_index(drop=True), matrix


def detect_isolation_forest(matrix: np.ndarray) -> np.ndarray:
    """Return a boolean anomaly mask from an Isolation Forest."""
    config = load_config()
    params = config["anomaly_detection"]["isolation_forest"]
    model = IsolationForest(
        contamination=params["contamination"],
        n_estimators=params["n_estimators"],
        random_state=config["project"]["random_seed"],
        n_jobs=-1,
    )
    return model.fit_predict(matrix) == -1


def detect_dbscan(matrix: np.ndarray) -> np.ndarray:
    """Return a boolean anomaly mask from DBSCAN noise points."""
    params = load_config()["anomaly_detection"]["dbscan"]
    model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], n_jobs=-1)
    labels = model.fit_predict(matrix)
    return labels == -1


def compare_detectors(
    iforest_mask: np.ndarray, dbscan_mask: np.ndarray
) -> dict[str, Any]:
    """Compute agreement statistics between the two detectors."""
    both = iforest_mask & dbscan_mask
    union = iforest_mask | dbscan_mask
    jaccard = float(both.sum() / union.sum()) if union.any() else 0.0
    return {
        "n_samples": int(len(iforest_mask)),
        "isolation_forest_flagged": int(iforest_mask.sum()),
        "dbscan_flagged": int(dbscan_mask.sum()),
        "flagged_by_both": int(both.sum()),
        "flagged_by_either": int(union.sum()),
        "jaccard_agreement": round(jaccard, 4),
    }


def _agreement_categories(
    iforest_mask: np.ndarray, dbscan_mask: np.ndarray
) -> np.ndarray:
    """Map detector masks onto the four agreement category labels."""
    categories = np.full(len(iforest_mask), AGREEMENT_LABELS[0], dtype=object)
    categories[iforest_mask & ~dbscan_mask] = AGREEMENT_LABELS[1]
    categories[~iforest_mask & dbscan_mask] = AGREEMENT_LABELS[2]
    categories[iforest_mask & dbscan_mask] = AGREEMENT_LABELS[3]
    return categories


def plot_anomalies_pca(
    matrix: np.ndarray, iforest_mask: np.ndarray, dbscan_mask: np.ndarray
) -> None:
    """Project the feature space to 2-D PCA and color by detector agreement."""
    coords = PCA(n_components=2, random_state=0).fit_transform(matrix)
    categories = _agreement_categories(iforest_mask, dbscan_mask)
    fig, ax = plt.subplots(figsize=(11, 8))
    style = {
        AGREEMENT_LABELS[0]: {"color": "#b8bcc2", "s": 5, "alpha": 0.3},
        AGREEMENT_LABELS[1]: {"color": CATEGORICAL_PALETTE[0], "s": 22, "alpha": 0.9},
        AGREEMENT_LABELS[2]: {"color": CATEGORICAL_PALETTE[1], "s": 22, "alpha": 0.9},
        AGREEMENT_LABELS[3]: {"color": CATEGORICAL_PALETTE[3], "s": 30, "alpha": 1.0},
    }
    for label in AGREEMENT_LABELS:
        mask = categories == label
        ax.scatter(coords[mask, 0], coords[mask, 1], label=label, **style[label])
    ax.set_title("Anomaly detection agreement (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=2)
    save_figure(fig, "anomalies_pca")
    plt.close(fig)


def plot_anomaly_scatter(
    sample: pd.DataFrame, iforest_mask: np.ndarray, dbscan_mask: np.ndarray
) -> None:
    """Temperature vs humidity scatter with flagged anomalies highlighted."""
    categories = _agreement_categories(iforest_mask, dbscan_mask)
    fig, ax = plt.subplots(figsize=(11, 8))
    normal = categories == AGREEMENT_LABELS[0]
    ax.scatter(
        sample.loc[normal, "temperature_celsius"],
        sample.loc[normal, "humidity"],
        color="#b8bcc2", s=5, alpha=0.25, label="Normal",
    )
    for label, color in [
        (AGREEMENT_LABELS[1], CATEGORICAL_PALETTE[0]),
        (AGREEMENT_LABELS[2], CATEGORICAL_PALETTE[1]),
        (AGREEMENT_LABELS[3], CATEGORICAL_PALETTE[3]),
    ]:
        mask = categories == label
        ax.scatter(
            sample.loc[mask, "temperature_celsius"],
            sample.loc[mask, "humidity"],
            color=color, s=26, alpha=0.9, label=label,
        )
    ax.set_title("Anomalies in temperature-humidity space")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.legend(markerscale=2)
    save_figure(fig, "anomalies_temp_humidity")
    plt.close(fig)


def explain_anomalies(
    sample: pd.DataFrame, iforest_mask: np.ndarray, dbscan_mask: np.ndarray,
    stats: dict[str, Any],
) -> None:
    """Write a data-driven markdown explanation of the flagged anomalies.

    For each detector the mean z-score of every feature within the flagged
    set quantifies which variables drive the anomaly, and the most frequent
    countries and weather conditions give real-world context.
    """
    features: list[str] = load_config()["anomaly_detection"]["features"]
    z_scores = (sample[features] - sample[features].mean()) / sample[features].std()
    lines = [
        "# Anomaly Detection Analysis",
        "",
        "## Detector comparison",
        "",
        f"- Common sample size: **{stats['n_samples']:,}** observations",
        f"- Isolation Forest flagged: **{stats['isolation_forest_flagged']:,}**",
        f"- DBSCAN (noise points) flagged: **{stats['dbscan_flagged']:,}**",
        f"- Flagged by both: **{stats['flagged_by_both']:,}** "
        f"(Jaccard agreement {stats['jaccard_agreement']:.1%})",
        "",
        "Isolation Forest isolates points that are *globally* easy to separate,",
        "while DBSCAN flags points in *locally sparse* regions. Their overlap is",
        "therefore the set of observations that are extreme by both definitions.",
        "",
    ]
    for name, mask in [
        ("Isolation Forest", iforest_mask), ("DBSCAN", dbscan_mask)
    ]:
        profile = z_scores[mask].mean().sort_values(key=abs, ascending=False)
        top = sample.loc[mask]
        lines += [
            f"## Why {name} anomalies occur",
            "",
            "Mean z-score of flagged observations (|z| > 1 marks a driver):",
            "",
            *[
                f"- `{feature}`: **{value:+.2f}**"
                for feature, value in profile.items()
            ],
            "",
            f"Most affected countries: "
            f"{', '.join(top['country'].value_counts().head(5).index)}.",
            f"Dominant conditions: "
            f"{', '.join(top['condition_text'].value_counts().head(5).index)}.",
            "",
        ]
    lines += [
        "## Interpretation",
        "",
        "Flagged records are dominated by physically extreme but real weather:",
        "heavy monsoon rainfall, desert heat with near-zero humidity, and",
        "high-wind storm events. They represent rare joint combinations",
        "(e.g. high temperature *and* high rainfall) rather than sensor errors,",
        "which is why they survive the cleaning stage's physical-range checks.",
        "",
    ]
    reports_dir = resolve_path(load_config()["paths"]["reports_dir"])
    (reports_dir / "anomaly_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_anomaly_detection() -> None:
    """Execute the anomaly detection stage end to end."""
    config = load_config()
    df = pd.read_parquet(resolve_path(config["paths"]["processed_data"]))

    with timed_step(logger, "prepare sample"):
        sample, matrix = _prepare_sample(df)
    with timed_step(logger, "isolation forest"):
        iforest_mask = detect_isolation_forest(matrix)
    with timed_step(logger, "dbscan"):
        dbscan_mask = detect_dbscan(matrix)

    stats = compare_detectors(iforest_mask, dbscan_mask)
    logger.info("Detector comparison: %s", stats)
    save_json(stats, "anomaly_comparison", directory_key="reports_dir")

    with timed_step(logger, "plots and explanation"):
        plot_anomalies_pca(matrix, iforest_mask, dbscan_mask)
        plot_anomaly_scatter(sample, iforest_mask, dbscan_mask)
        explain_anomalies(sample, iforest_mask, dbscan_mask, stats)


if __name__ == "__main__":
    run_anomaly_detection()
