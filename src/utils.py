"""Shared utilities for the weather trend forecasting project.

Provides configuration loading, logging, filesystem helpers,
figure/model persistence and regression evaluation metrics used
across every pipeline stage.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import yaml

_CONFIG_CACHE: dict[str, Any] | None = None

#: Fixed-order categorical palette, validated for CVD safety and contrast
#: (dataviz six-checks validator, light surface). Never cycled or re-ordered.
CATEGORICAL_PALETTE: list[str] = [
    "#0173b2",  # blue
    "#b25d00",  # burnt orange
    "#029e73",  # green
    "#a8488f",  # magenta
    "#8f6d13",  # olive
]

#: Sequential colormap for magnitude encodings (perceptually uniform).
SEQUENTIAL_CMAP: str = "viridis"

#: Diverging colormap for signed quantities (correlations, anomalies).
DIVERGING_CMAP: str = "RdBu_r"

#: Continent derived from the IANA timezone prefix of each location.
_CONTINENT_BY_TZ_PREFIX: dict[str, str] = {
    "Africa": "Africa",
    "America": "Americas",
    "Asia": "Asia",
    "Atlantic": "Atlantic",
    "Australia": "Oceania",
    "Europe": "Europe",
    "Indian": "Indian Ocean",
    "Pacific": "Oceania",
    "Arctic": "Europe",
    "Antarctica": "Antarctica",
    "US": "Americas",
}


def add_continent_column(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a ``continent`` column from the IANA timezone identifier.

    The timezone prefix (e.g. ``Asia/Tokyo`` -> ``Asia``) is a reliable,
    dependency-free proxy for the continent of each location.

    Args:
        df: Frame containing a ``timezone`` column.

    Returns:
        A copy of ``df`` with an added ``continent`` column.
    """
    df = df.copy()
    prefix = df["timezone"].astype(str).str.split("/").str[0]
    df["continent"] = prefix.map(_CONTINENT_BY_TZ_PREFIX).fillna("Other")
    return df


def get_project_root() -> Path:
    """Return the absolute path of the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the YAML configuration file, caching it after the first read.

    Args:
        config_path: Optional explicit path to a config file. Defaults to
            ``config.yaml`` in the project root.

    Returns:
        The parsed configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE
    path = Path(config_path) if config_path else get_project_root() / "config.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config_path is None:
        _CONFIG_CACHE = config
    return config


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path from the config (relative to project root) to absolute."""
    path = Path(relative)
    return path if path.is_absolute() else get_project_root() / path


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (and parents) if missing and return it."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def setup_logger(name: str = "weatherscope") -> logging.Logger:
    """Create (or fetch) a logger writing to stdout and the pipeline log file.

    Args:
        name: Logger name; child loggers share the root handlers.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    try:
        log_file = resolve_path(load_config()["paths"]["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:  # pragma: no cover - log file is best-effort
        logger.warning("Could not attach file handler for logging.")
    logger.propagate = False
    return logger


@contextmanager
def timed_step(logger: logging.Logger, step_name: str) -> Iterator[None]:
    """Context manager logging the wall-clock duration of a pipeline step."""
    logger.info("START %s", step_name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("DONE  %s (%.1fs)", step_name, elapsed)


def save_figure(fig: Any, name: str, subdir: str = "") -> Path:
    """Save a matplotlib figure as a PNG under ``outputs/figures``.

    Args:
        fig: A matplotlib ``Figure`` instance.
        name: Base filename without extension.
        subdir: Optional subdirectory inside the figures folder.

    Returns:
        The path of the written file.
    """
    figures_dir = ensure_dir(Path(load_config()["paths"]["figures_dir"]) / subdir)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def save_plotly(fig: Any, name: str, subdir: str = "") -> Path:
    """Save a Plotly figure as a standalone interactive HTML file.

    Args:
        fig: A plotly ``Figure`` instance.
        name: Base filename without extension.
        subdir: Optional subdirectory inside the figures folder.

    Returns:
        The path of the written file.
    """
    figures_dir = ensure_dir(Path(load_config()["paths"]["figures_dir"]) / subdir)
    path = figures_dir / f"{name}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def save_model(model: Any, name: str) -> Path:
    """Persist a fitted model to ``outputs/models`` with joblib."""
    models_dir = ensure_dir(load_config()["paths"]["models_dir"])
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str) -> Any:
    """Load a previously saved model from ``outputs/models``."""
    path = resolve_path(load_config()["paths"]["models_dir"]) / f"{name}.joblib"
    return joblib.load(path)


def save_json(data: Any, name: str, directory_key: str = "models_dir") -> Path:
    """Serialize ``data`` as pretty-printed JSON under a configured directory.

    Args:
        data: JSON-serializable object.
        name: Base filename without extension.
        directory_key: Key inside ``paths`` config pointing at the target dir.

    Returns:
        The path of the written file.
    """
    target_dir = ensure_dir(load_config()["paths"][directory_key])
    path = target_dir / f"{name}.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)
    return path


def load_json(name: str, directory_key: str = "models_dir") -> Any:
    """Load a JSON artifact previously written with :func:`save_json`."""
    path = resolve_path(load_config()["paths"][directory_key]) / f"{name}.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE and R-squared for a forecast.

    MAPE ignores near-zero actuals to avoid division blow-ups (relevant
    for precipitation series that contain many zero values).

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        Mapping of metric name to value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denom_mask = np.abs(y_true) > 1e-3
    if denom_mask.any():
        mape = float(
            np.mean(np.abs(errors[denom_mask] / y_true[denom_mask])) * 100.0
        )
    else:
        mape = float("nan")
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def format_metrics_table(metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert a nested metrics mapping into a tidy comparison DataFrame.

    Args:
        metrics_by_model: Mapping of model name to its metric dictionary.

    Returns:
        DataFrame indexed by model, sorted by RMSE ascending.
    """
    table = pd.DataFrame(metrics_by_model).T
    return table.sort_values("RMSE")
