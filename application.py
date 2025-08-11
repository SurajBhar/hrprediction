#!/usr/bin/env python
"""
Flask app for Hotel Reservation Prediction.

- Loads a trained model (joblib) from MODEL_OUTPUT_PATH
- Renders a form for user inputs
- Predicts cancellation likelihood based on submitted features
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from flask import Flask, render_template, request

from src.logger import get_logger
from src.custom_exception import CustomException
from config.path_config import MODEL_OUTPUT_PATH  # ensure this matches your project

logger = get_logger(__name__)
app = Flask(__name__)

# Maintain a strict feature order compatible with your trained model
FEATURE_ORDER: List[str] = [
    "lead_time",
    "no_of_special_request",
    "avg_price_per_room",
    "arrival_month",
    "arrival_date",
    "market_segment_type",
    "no_of_week_nights",
    "no_of_weekend_nights",
    "type_of_meal_plan",
    "room_type_reserved",
]

_model = None  # lazy-loaded model


def _load_model():
    """Load and cache the trained model."""
    global _model
    if _model is not None:
        return _model

    model_path = Path(MODEL_OUTPUT_PATH)
    if not model_path.is_file():
        logger.error("Model file not found at %s", model_path)
        raise CustomException(f"Model file not found at {model_path}")

    try:
        _model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
        return _model
    except Exception as e:
        logger.exception("Failed to load model from %s", model_path)
        raise CustomException("Failed to load model", e)


def _to_int(val: str, field: str, min_val: int | None = None, max_val: int | None = None) -> int:
    try:
        x = int(val)
    except Exception as e:
        raise CustomException(f"Invalid integer for '{field}': {val!r}", e)
    if min_val is not None and x < min_val:
        raise CustomException(f"'{field}' must be >= {min_val}, got {x}")
    if max_val is not None and x > max_val:
        raise CustomException(f"'{field}' must be <= {max_val}, got {x}")
    return x


def _to_float(val: str, field: str, min_val: float | None = None) -> float:
    try:
        x = float(val)
    except Exception as e:
        raise CustomException(f"Invalid float for '{field}': {val!r}", e)
    if min_val is not None and x < min_val:
        raise CustomException(f"'{field}' must be >= {min_val}, got {x}")
    return x


def _build_feature_vector(form) -> np.ndarray:
    """Extract and validate form fields, return a (1, n_features) numpy array in FEATURE_ORDER."""
    try:
        vals = {
            "lead_time": _to_int(form["lead_time"], "lead_time", min_val=0),
            "no_of_special_request": _to_int(form["no_of_special_request"], "no_of_special_request", min_val=0),
            "avg_price_per_room": _to_float(form["avg_price_per_room"], "avg_price_per_room", min_val=0.0),
            "arrival_month": _to_int(form["arrival_month"], "arrival_month", min_val=1, max_val=12),
            "arrival_date": _to_int(form["arrival_date"], "arrival_date", min_val=1, max_val=31),
            "market_segment_type": _to_int(form["market_segment_type"], "market_segment_type", min_val=0),
            "no_of_week_nights": _to_int(form["no_of_week_nights"], "no_of_week_nights", min_val=0),
            "no_of_weekend_nights": _to_int(form["no_of_weekend_nights"], "no_of_weekend_nights", min_val=0),
            "type_of_meal_plan": _to_int(form["type_of_meal_plan"], "type_of_meal_plan", min_val=0),
            "room_type_reserved": _to_int(form["room_type_reserved"], "room_type_reserved", min_val=0),
        }
        features = np.array([[vals[name] for name in FEATURE_ORDER]], dtype=float)
        return features
    except KeyError as e:
        miss = str(e).strip("'")
        raise CustomException(f"Missing required field: {miss}")
    except CustomException:
        raise  # re-raise to be handled by caller
    except Exception as e:
        raise CustomException("Unexpected error while building feature vector", e)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            model = _load_model()
            features = _build_feature_vector(request.form)
            pred = model.predict(features)
            # Ensure scalar int for template logic
            prediction = int(pred[0]) if hasattr(pred, "__iter__") else int(pred)
            logger.info("Prediction computed: %s", prediction)
        except CustomException as ce:
            error = str(ce)
            logger.error("Prediction failed: %s", error)
        except Exception as e:
            error = "An unexpected error occurred during prediction."
            logger.exception(error)

    # index.html currently does not render `error`, but we pass it if we want to display it later.
    return render_template("index.html", prediction=prediction, error=error)


@app.get("/healthz")
def healthz():
    """Basic health check endpoint."""
    try:
        _ = _load_model()
        return {"status": "ok"}, 200
    except Exception:
        return {"status": "model_not_loaded"}, 500


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
