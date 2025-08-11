#!/usr/bin/env python
"""
End-to-end training pipeline for the hrprediction project.

Stages:
  1) Data Ingestion  -> downloads raw CSV from GCS
  2) Data Processing -> clean/encode/skew-handle/SMOTE/train feature selection
  3) Model Training  -> LightGBM + RandomizedSearchCV, metrics + MLflow logging

This script wires together all modules and persists outputs to
paths defined in `config.path_config`.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.utility_functions import read_yaml
from config.path_config import (
    CONFIG_PATH,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    MODEL_OUTPUT_PATH,
)

# Optional: load environment variables from .env if present
try:  # pragma: no cover - optional convenience
    from dotenv import load_dotenv  # type: ignore

    _ROOT = Path(__file__).resolve().parents[1]
    _ENV = _ROOT / ".env"
    if _ENV.exists():
        load_dotenv(dotenv_path=_ENV)
except Exception:
    pass

logger = get_logger(__name__)


def _check_prereqs() -> None:
    """Log helpful hints if common env configuration is missing."""
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.warning(
            "GOOGLE_APPLICATION_CREDENTIALS is not set. GCS access may fail."
        )


def run_pipeline(config_path: Path | str = CONFIG_PATH) -> None:
    """
    Run the full pipeline: ingestion → preprocessing → training.

    Parameters
    ----------
    config_path : Path | str
        Path to the YAML configuration file used by component stages.
    """
    start = time.perf_counter()
    logger.info("Pipeline started")

    try:
        _check_prereqs()

        # Load config once and pass where needed
        config = read_yaml(config_path)

        # 1) Data Ingestion
        logger.info("[1/3] Data Ingestion → start")
        ingestion = DataIngestion(config)
        ingestion.run()
        logger.info("[1/3] Data Ingestion → done (raw: %s)", Path(TRAIN_FILE_PATH).parent)

        # 2) Data Processing
        logger.info("[2/3] Data Processing → start")
        # processed_dir is deprecated in DataProcessor; pass None to avoid warning
        processor = DataProcessor(
            train_path=TRAIN_FILE_PATH,
            test_path=TEST_FILE_PATH,
            processed_dir=None,
            config_path=config_path,
        )
        processor.process()
        logger.info(
            "[2/3] Data Processing → done (train: %s, test: %s)",
            PROCESSED_TRAIN_DATA_PATH,
            PROCESSED_TEST_DATA_PATH,
        )

        # 3) Model Training
        logger.info("[3/3] Model Training → start")
        trainer = ModelTraining(
            train_path=PROCESSED_TRAIN_DATA_PATH,
            test_path=PROCESSED_TEST_DATA_PATH,
            model_output_path=MODEL_OUTPUT_PATH,
        )
        trainer.run()
        logger.info("[3/3] Model Training → done (model: %s)", MODEL_OUTPUT_PATH)

        elapsed = time.perf_counter() - start
        logger.info("Pipeline finished successfully in %.2f s", elapsed)

    except CustomException as ce:
        logger.exception("Pipeline failed with CustomException: %s", ce)
        raise
    except Exception as e:  # catch-all to wrap unexpected errors
        logger.exception("Pipeline failed with unexpected error")
        raise CustomException("Pipeline execution failed", e) from e


if __name__ == "__main__":
    run_pipeline()
