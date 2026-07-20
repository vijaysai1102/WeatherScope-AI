"""Command-line orchestrator for the weather trend forecasting pipeline.

Runs any subset of pipeline stages in dependency order. Each stage is a
``run_*`` function exposed by a module in :mod:`src`; imports are lazy so
individual stages can be executed without importing heavy optional
dependencies of other stages.

Examples:
    python main.py --stage all
    python main.py --stage preprocess features eda
    python main.py --list
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable

from src.utils import setup_logger, timed_step

# Ordered mapping of stage name -> (module, entry-point function).
STAGES: dict[str, tuple[str, str]] = {
    "preprocess": ("src.preprocessing", "run_preprocessing"),
    "features": ("src.feature_engineering", "run_feature_engineering"),
    "eda": ("src.eda", "run_eda"),
    "anomaly": ("src.anomaly_detection", "run_anomaly_detection"),
    "forecast": ("src.forecasting", "run_forecasting"),
    "ensemble": ("src.ensemble", "run_ensemble"),
    "importance": ("src.feature_importance", "run_feature_importance"),
    "climate": ("src.climate_analysis", "run_climate_analysis"),
    "air_quality": ("src.air_quality", "run_air_quality"),
    "spatial": ("src.spatial_analysis", "run_spatial_analysis"),
    "report": ("src.report", "run_report"),
}


def _resolve_stage(name: str) -> Callable[[], None]:
    """Import and return the entry-point callable for a stage name."""
    module_name, func_name = STAGES[name]
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for stage selection."""
    parser = argparse.ArgumentParser(
        description="Weather trend forecasting pipeline runner."
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        default=["all"],
        choices=["all", *STAGES],
        help="Pipeline stage(s) to run, in order. Default: all.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available stages and exit."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested pipeline stages and return an exit code."""
    args = parse_args(argv)
    if args.list:
        print("Available stages (in execution order):")
        for stage in STAGES:
            print(f"  - {stage}")
        return 0

    selected = list(STAGES) if "all" in args.stage else args.stage
    # Preserve canonical execution order regardless of CLI ordering.
    selected = [stage for stage in STAGES if stage in selected]

    logger = setup_logger("pipeline")
    logger.info("Running stages: %s", ", ".join(selected))
    for stage in selected:
        entry_point = _resolve_stage(stage)
        with timed_step(logger, f"stage '{stage}'"):
            entry_point()
    logger.info("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
