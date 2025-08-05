#!/usr/bin/env python
"""
Utility functions for the hotel-reservation-prediction project.

This module provides common helper functions for configuration
and data loading, with consistent logging and error handling.
"""
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
import yaml

from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a YAML configuration file and return its contents as a dictionary.

    Args:
        file_path (Union[str, Path]): Path to the YAML file.

    Returns:
        Dict[str, Any]: Parsed YAML content.

    Raises:
        CustomException: If the file doesn't exist or cannot be parsed.
    """
    path = Path(file_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"YAML file not found at path: {path}")

        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        logger.info(f"Successfully read YAML file: {path}")
        return config

    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.error(f"Error reading YAML file at {path}: {e}")
        raise CustomException(f"Failed to read YAML file at {path}", e)
    except Exception as e:
        logger.error(f"Unexpected error reading YAML file at {path}: {e}")
        raise CustomException("Unexpected error in read_yaml", e)


def load_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load tabular data from a CSV file into a pandas DataFrame.

    Args:
        file_path (Union[str, Path]): Path to the CSV file.
        **kwargs: Optional keyword arguments passed to pandas.read_csv.

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        CustomException: If the file isn't found or cannot be read.
    """
    path = Path(file_path)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found at path: {path}")

        logger.info(f"Loading data from CSV: {path}")
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Data loaded successfully, shape: {df.shape}")
        return df

    except FileNotFoundError as e:
        logger.error(f"File not found: {path}")
        raise CustomException(f"CSV file not found at {path}", e)
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error for CSV file {path}: {e}")
        raise CustomException(f"Failed to parse CSV file at {path}", e)
    except Exception as e:
        logger.error(f"Unexpected error loading data from {path}: {e}")
        raise CustomException("Unexpected error in load_data", e)
