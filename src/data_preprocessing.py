#!/usr/bin/env python
"""
Data preprocessing module for the hrprediction project.

This module performs reproducible tabular preprocessing steps for the
hotel reservation dataset and **writes outputs to configured constants**:
`PROCESSED_TRAIN_DATA_PATH` and `PROCESSED_TEST_DATA_PATH`.

Steps:
- Drop unused columns and duplicates
- Encode categorical features (fit on train, apply to test)
- Handle skewed numerical features via log1p (columns chosen on train)
- Balance the training set with SMOTE (test set is never resampled)
- Select top-N features using RandomForest importance (fit on train)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.utility_functions import read_yaml, load_data
from config.path_config import (
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
    PROCESSED_TRAIN_DATA_PATH,
    PROCESSED_TEST_DATA_PATH,
)

logger = get_logger(__name__)


PathLike = Union[str, Path]


class DataProcessor:
    """
    Orchestrates preprocessing for training and test sets.

    Parameters
    ----------
    train_path : PathLike
        Path to the raw training CSV.
    test_path : PathLike
        Path to the raw test CSV.
    processed_dir : Optional[PathLike]
        **Deprecated/ignored**. Outputs are controlled by
        `PROCESSED_TRAIN_DATA_PATH` and `PROCESSED_TEST_DATA_PATH`.
    config_path : PathLike | Mapping[str, Any]
        Path to YAML config or a pre-loaded config dict.

    Notes
    -----
    Expected config keys under `data_processing`:
      - categorical_columns: List[str]
      - numerical_columns: List[str]
      - drop_columns: List[str] (optional)
      - skewness_threshold: float (e.g., 1.0)
      - no_of_features: int (top-N features to keep)
      - target_column: str (e.g., "booking_status")
      - random_state: int (default 42)
    """

    def __init__(
        self,
        train_path: PathLike,
        test_path: PathLike,
        processed_dir: Optional[PathLike],  # kept for backward compatibility
        config_path: Union[PathLike, Mapping[str, Any]],
    ) -> None:
        try:
            self.train_path = Path(train_path)
            self.test_path = Path(test_path)

            if processed_dir is not None:
                logger.warning(
                    "`processed_dir` argument is deprecated and ignored. "
                    "Outputs will be written to configured constants: %s and %s",
                    PROCESSED_TRAIN_DATA_PATH,
                    PROCESSED_TEST_DATA_PATH,
                )

            self.cfg: Dict[str, Any] = (
                read_yaml(config_path) if isinstance(config_path, (str, Path)) else dict(config_path)
            )
            dp = self.cfg.get("data_processing", {})

            self.cat_cols: List[str] = list(dp.get("categorical_columns", []))
            self.num_cols: List[str] = list(dp.get("numerical_columns", []))
            self.drop_cols: List[str] = list(dp.get("drop_columns", ["Unnamed: 0", "Booking_ID"]))
            self.skew_threshold: float = float(dp.get("skewness_threshold", 1.0))
            self.n_features: int = int(dp.get("no_of_features", 10))
            self.target_col: str = str(dp.get("target_column", "booking_status"))
            self.random_state: int = int(dp.get("random_state", 42))

            # Fitted state
            self._encoders: Dict[str, Dict[Any, int]] = {}
            self._skewed_cols: List[str] = []
            self._selected_features: List[str] = []

            logger.info(
                "Initialized DataProcessor with %s cat cols, %s num cols, target='%s'",
                len(self.cat_cols), len(self.num_cols), self.target_col,
            )
        except Exception as e:
            raise CustomException("Failed to initialize DataProcessor", e)

    # ------------------------------
    # Internal helpers
    # ------------------------------
    @staticmethod
    def _drop_and_dedupe(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(columns=existing)
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if after != before:
            logger.info("Dropped %d duplicate rows", before - after)
        return df

    @staticmethod
    def _build_mapping(values: pd.Series) -> Dict[Any, int]:
        """Create a deterministic value->code mapping (lexicographic order)."""
        unique_vals = pd.Index(sorted(values.dropna().astype(str).unique()))
        return {v: i for i, v in enumerate(unique_vals)}

    def _encode_categoricals_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cat_cols:
            if col not in df.columns:
                logger.warning("Categorical column '%s' not found; skipping.", col)
                continue
            mapping = self._build_mapping(df[col].astype(str))
            self._encoders[col] = mapping
            df[col] = df[col].astype(str).map(mapping).astype("Int64").fillna(-1).astype(int)
        return df

    def _encode_categoricals_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.cat_cols:
            if col not in df.columns:
                logger.warning("Categorical column '%s' not found in transform; skipping.", col)
                continue
            mapping = self._encoders.get(col, {})
            df[col] = df[col].astype(str).map(mapping).astype("Int64").fillna(-1).astype(int)
        return df

    def _detect_skewed_cols(self, df: pd.DataFrame) -> List[str]:
        skewness = df[self.num_cols].apply(pd.Series.skew, axis=0)
        chosen = [c for c, s in skewness.items() if pd.notna(s) and s > self.skew_threshold]
        return chosen

    def _apply_log1p(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        applied = []
        for c in cols:
            if c not in df.columns:
                continue
            min_val = df[c].min(skipna=True)
            if pd.isna(min_val) or min_val <= -1:
                logger.warning("Skipping log1p for '%s' due to min value %s <= -1", c, min_val)
                continue
            df[c] = np.log1p(df[c])
            applied.append(c)
        if applied:
            logger.info("Applied log1p to: %s", applied)
        return df

    # ------------------------------
    # Public API
    # ------------------------------
    def preprocess_fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = self._drop_and_dedupe(df.copy(), self.drop_cols)
            df = self._encode_categoricals_fit(df)
            self._skewed_cols = self._detect_skewed_cols(df)
            df = self._apply_log1p(df, self._skewed_cols)
            return df
        except Exception as e:
            raise CustomException("Error during training preprocessing", e)

    def preprocess_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = self._drop_and_dedupe(df.copy(), self.drop_cols)
            df = self._encode_categoricals_transform(df)
            df = self._apply_log1p(df, self._skewed_cols)
            return df
        except Exception as e:
            raise CustomException("Error during inference preprocessing", e)

    def balance_train(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.target_col not in df.columns:
                raise KeyError(f"Target column '{self.target_col}' not found")
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]

            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X, y)
            out = pd.DataFrame(X_res, columns=X.columns)
            out[self.target_col] = y_res
            logger.info("SMOTE applied: %s -> %s", X.shape, X_res.shape)
            return out
        except Exception as e:
            raise CustomException("Error while balancing training data", e)

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        try:
            if self.target_col not in df.columns:
                raise KeyError(f"Target column '{self.target_col}' not found")
            X = df.drop(columns=[self.target_col])
            y = df[self.target_col]

            model = RandomForestClassifier(random_state=self.random_state)
            model.fit(X, y)
            importances = pd.Series(model.feature_importances_, index=X.columns)
            self._selected_features = (
                importances.sort_values(ascending=False).head(self.n_features).index.tolist()
            )
            logger.info("Selected top-%d features: %s", self.n_features, self._selected_features)
            kept = self._selected_features + [self.target_col]
            return df[kept], self._selected_features
        except Exception as e:
            raise CustomException("Error during feature selection", e)

    def apply_selected_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if not self._selected_features:
                raise RuntimeError("No features selected. Call select_features on train first.")
            kept = self._selected_features + [self.target_col]
            missing = [c for c in kept if c not in df.columns]
            if missing:
                raise KeyError(f"Columns missing for projection: {missing}")
            return df[kept].copy()
        except Exception as e:
            raise CustomException("Error applying selected features", e)

    # ------------------------------
    # Pipeline
    # ------------------------------
    def process(self) -> None:
        """Full preprocessing pipeline. Outputs saved to configured constants."""
        try:
            logger.info("Loading raw datasets: %s | %s", self.train_path, self.test_path)
            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # Fit on train, transform both
            train_p = self.preprocess_fit_transform(train_df)
            test_p = self.preprocess_transform(test_df)

            # Balance training only
            train_bal = self.balance_train(train_p)

            # Feature selection on balanced train; apply to test
            train_sel, _ = self.select_features(train_bal)
            test_sel = self.apply_selected_features(test_p)

            # Ensure output directories exist and save using constants
            Path(PROCESSED_TRAIN_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(PROCESSED_TEST_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)

            train_out = Path(PROCESSED_TRAIN_DATA_PATH)
            test_out = Path(PROCESSED_TEST_DATA_PATH)

            train_sel.to_csv(train_out, index=False)
            test_sel.to_csv(test_out, index=False)
            logger.info("Saved processed train to %s and test to %s", train_out, test_out)
        except Exception as e:
            logger.exception("Data processing pipeline failed")
            raise CustomException("Error in data processing pipeline", e)


if __name__ == "__main__":
    # Use configured constants for inputs and outputs
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    dp = DataProcessor(
        train_path=TRAIN_FILE_PATH,
        test_path=TEST_FILE_PATH,
        processed_dir=None,  # ignored
        config_path=config_path,
    )
    dp.process()
