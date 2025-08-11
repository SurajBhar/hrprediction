#!/usr/bin/env python
"""
Model training module for the hrprediction project.

This module trains a LightGBM classifier with hyperparameter search,
logs artifacts/metrics to MLflow, and persists the best model to disk.

Pipeline steps:
- Load processed train/test CSVs
- RandomizedSearchCV over LightGBM hyperparameters
- Evaluate on the held-out test set (accuracy/precision/recall/F1)
- Save model with joblib and log everything to MLflow
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

from src.custom_exception import CustomException
from src.logger import get_logger
from utils.utility_functions import load_data
from config.path_config import (
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
    MODEL_OUTPUT_PATH,
)

# Optional param grids from config; fall back to sane defaults if missing
try:  # noqa: SIM105
    from config.model_params import LIGHTGM_PARAMS as _LIGHTGBM_PARAM_DIST  # type: ignore
except Exception:  # pragma: no cover - fallback if constant not defined
    _LIGHTGBM_PARAM_DIST = {
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 7, 9, 12],
        "learning_rate": np.linspace(0.01, 0.2, 10).tolist(),
        "n_estimators": [100, 200, 400, 800],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [0.0, 0.1, 0.5, 1.0],
    }

try:  # noqa: SIM105
    from config.model_params import RANDOM_SEARCH_PARAMS as _RANDOM_SEARCH_PARAMS  # type: ignore
except Exception:  # pragma: no cover
    _RANDOM_SEARCH_PARAMS = {
        "n_iter": 30,
        "cv": 5,
        "n_jobs": -1,
        "verbose": 1,
        "random_state": 42,
        "scoring": "f1",
    }


logger = get_logger(__name__)


class ModelTraining:
    """
    Train and evaluate a LightGBM model with hyperparameter search.

    Parameters
    ----------
    train_path : str | Path
        Path to processed training CSV.
    test_path : str | Path
        Path to processed test CSV.
    model_output_path : str | Path
        Where to persist the trained model (joblib file).
    """

    def __init__(self, train_path: str | Path, test_path: str | Path, model_output_path: str | Path) -> None:
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.model_output_path = Path(model_output_path)

        self.param_dist: Dict[str, Any] = dict(_LIGHTGBM_PARAM_DIST)
        self.search_cfg: Dict[str, Any] = dict(_RANDOM_SEARCH_PARAMS)

    # ------------------------------
    # Data loading
    # ------------------------------
    def load_and_split_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Load processed train/test and separate features/target."""
        try:
            logger.info("Loading processed data: train=%s test=%s", self.train_path, self.test_path)
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            if "booking_status" not in train_df.columns or "booking_status" not in test_df.columns:
                raise KeyError("Target column 'booking_status' not found in processed datasets")

            X_train = train_df.drop(columns=["booking_status"])  # features
            y_train = train_df["booking_status"].astype(int)
            X_test = test_df.drop(columns=["booking_status"])   # features
            y_test = test_df["booking_status"].astype(int)
            logger.info("Shapes -> X_train=%s, X_test=%s", X_train.shape, X_test.shape)
            return X_train, y_train, X_test, y_test
        except Exception as e:
            logger.error("Error while loading/splitting data: %s", e)
            raise CustomException("Failed to load processed datasets", e)

    # ------------------------------
    # Training
    # ------------------------------
    def train_lgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> lgb.LGBMClassifier:
        """Run RandomizedSearchCV for LightGBM and return the best estimator."""
        try:
            logger.info("Initializing LightGBM model and RandomizedSearchCV")
            base_model = lgb.LGBMClassifier(random_state=self.search_cfg.get("random_state", 42))

            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_dist,
                n_iter=int(self.search_cfg.get("n_iter", 30)),
                cv=int(self.search_cfg.get("cv", 5)),
                n_jobs=int(self.search_cfg.get("n_jobs", -1)),
                verbose=int(self.search_cfg.get("verbose", 1)),
                random_state=int(self.search_cfg.get("random_state", 42)),
                scoring=self.search_cfg.get("scoring", "f1"),
                refit=True,
                return_train_score=False,
            )

            logger.info("Starting hyperparameter search (n_iter=%s, cv=%s)", search.n_iter, search.cv)
            search.fit(X_train, y_train)

            logger.info("Search completed. Best score=%.5f", search.best_score_)
            logger.info("Best params: %s", search.best_params_)
            return search.best_estimator_
        except Exception as e:
            logger.error("Error during model training: %s", e)
            raise CustomException("Failed to train LightGBM model", e)

    # ------------------------------
    # Evaluation
    # ------------------------------
    @staticmethod
    def evaluate_model(model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Compute standard binary classification metrics."""
        try:
            y_pred = model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }
            return metrics
        except Exception as e:
            raise CustomException("Failed to evaluate model", e)

    # ------------------------------
    # Persistence
    # ------------------------------
    def save_model(self, model: lgb.LGBMClassifier) -> Path:
        """Persist the trained model to disk and return the path."""
        try:
            self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, self.model_output_path)
            logger.info("Model saved to %s", self.model_output_path)
            return self.model_output_path
        except Exception as e:
            raise CustomException("Failed to save model", e)

    # ------------------------------
    # Orchestration
    # ------------------------------
    def run(self) -> None:
        """Execute the end-to-end training pipeline with MLflow logging."""
        try:
            with mlflow.start_run():
                logger.info("Model training pipeline started")

                # Log datasets as artifacts
                mlflow.log_artifact(str(self.train_path), artifact_path="datasets")
                mlflow.log_artifact(str(self.test_path), artifact_path="datasets")

                # Load and train
                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_model = self.train_lgbm(X_train, y_train)

                # Evaluate
                metrics = self.evaluate_model(best_model, X_test, y_test)
                logger.info(
                    "Metrics -> accuracy=%.4f, precision=%.4f, recall=%.4f, f1=%.4f",
                    metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
                )

                # Save model
                model_path = self.save_model(best_model)

                # MLflow logging
                mlflow.log_params(best_model.get_params(deep=False))
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(model_path), artifact_path="model")
                # Also log model in MLflow-native format
                mlflow.sklearn.log_model(best_model, artifact_path="sklearn-model")

                logger.info("Model training pipeline completed successfully")
        except Exception as e:
            logger.exception("Training pipeline failed")
            raise CustomException("Failed during model training pipeline", e)


if __name__ == "__main__":
    trainer = ModelTraining(
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
    )
    trainer.run()
