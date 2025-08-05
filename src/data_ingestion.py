#!/usr/bin/env python
"""
Data ingestion module for the hrprediction project.

This module handles downloading raw CSV data from Google Cloud Storage and
splitting it into training and testing sets according to the project configuration.
"""
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import (
    RAW_DIR,
    RAW_FILE_PATH,
    TRAIN_FILE_PATH,
    TEST_FILE_PATH,
)
from utils.utility_functions import read_yaml

logger = get_logger(__name__)


class DataIngestion:
    """
    Handles downloading raw data from GCS and splitting into train/test sets.

    Attributes:
        config (Dict[str, Any]): Full configuration dictionary.
        bucket_name (str): GCS bucket name.
        source_blob_name (str): Name of the CSV file in the bucket.
        train_ratio (float): Proportion of data used for training.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config.get("data_ingestion", {})
        self.bucket_name = self.config.get("bucket_name", "")
        self.source_blob_name = self.config.get("bucket_file_name", "")
        self.train_ratio = float(self.config.get("train_ratio", 0.8))

        # Ensure raw data directory exists
        Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Initialized DataIngestion(bucket={self.bucket_name}, "
            f"blob={self.source_blob_name}, train_ratio={self.train_ratio})"
        )

    def download_csv_from_gcp(self) -> Path:
        """
        Download CSV file from Google Cloud Storage to the local filesystem.

        Returns:
            Path: Local path to the downloaded file.

        Raises:
            CustomException: On any failure connecting to GCS or writing the file.
        """
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.source_blob_name)
            destination = Path(RAW_FILE_PATH)

            blob.download_to_filename(str(destination))
            logger.info(f"Downloaded {self.source_blob_name} to {destination}")
            return destination

        except Exception as e:
            logger.error(
                f"Failed to download {self.source_blob_name} "
                f"from bucket {self.bucket_name}: {e}"
            )
            raise CustomException("Error downloading data from GCS", e)

    def split_data(self) -> None:
        """
        Load the raw CSV and split into train/test, saving results to disk.

        Raises:
            CustomException: On read, split, or write failures.
        """
        try:
            df = pd.read_csv(RAW_FILE_PATH)
            train_df, test_df = train_test_split(
                df,
                test_size=1 - self.train_ratio,
                random_state=42
            )

            # Ensure output dirs exist
            Path(TRAIN_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)
            Path(TEST_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

            train_df.to_csv(TRAIN_FILE_PATH, index=False)
            test_df.to_csv(TEST_FILE_PATH, index=False)

            logger.info(
                f"Saved train set to {TRAIN_FILE_PATH} ({train_df.shape}), "
                f"test set to {TEST_FILE_PATH} ({test_df.shape})"
            )
        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise CustomException("Error splitting data into train/test", e)

    def run(self) -> None:
        """
        Execute the data ingestion pipeline: download then split.

        Raises:
            CustomException: Propagates any ingestion errors.
        """
        try:
            logger.info("Starting data ingestion process")
            self.download_csv_from_gcp()
            self.split_data()
            logger.info("Data ingestion completed successfully")
        except CustomException:
            logger.exception("Data ingestion process failed")
            raise


if __name__ == "__main__":
    config = read_yaml(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    ingestion = DataIngestion(config)
    ingestion.run()
