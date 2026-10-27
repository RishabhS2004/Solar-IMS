"""
data_loader.py
==============
Centralised data loading, validation, and caching utilities for Solar IMS.
Used by both app.py (full dashboard) and live_demo.py (standalone simulator).

This module handles:
- Loading processed_output.csv and model_metrics.json
- Validating required columns
- Handling missing or malformed data
- Providing consistent data interfaces to both entry points
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

_BASE_DIR = Path(__file__).parent

# Required columns that must exist in the processed CSV
REQUIRED_COLUMNS = [
    "measured_on", "irradiance", "ambient_temp", "module_temp",
    "total_power", "expected_power", "recommendation", "priority",
]

# Model features used by the XGBoost model
MODEL_FEATURES = ["irradiance", "ambient_temp", "module_temp", "temp_diff"]

# Plant rated capacity
RATED_CAPACITY_KW = 4000


def validate_columns(df: pd.DataFrame, required: list[str] | None = None) -> list[str]:
    """
    Validate that required columns exist in a DataFrame.
    Returns a list of missing columns (empty if all present).
    """
    if required is None:
        required = REQUIRED_COLUMNS
    missing = [c for c in required if c not in df.columns]
    return missing


def load_processed_data(filepath: str | Path | None = None) -> pd.DataFrame:
    """
    Load the processed_output.csv dataset.
    
    Parameters
    ----------
    filepath : str or Path, optional
        Path to the CSV file. Defaults to processed_output.csv in the project dir.
    
    Returns
    -------
    pd.DataFrame
        Validated DataFrame with parsed timestamps.
    
    Raises
    ------
    FileNotFoundError
        If the data file does not exist.
    ValueError
        If required columns are missing.
    """
    if filepath is None:
        filepath = _BASE_DIR / "processed_output.csv"
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}. "
            "Run `python prepare_data.py` first to generate it."
        )

    df = pd.read_csv(filepath, parse_dates=["measured_on"])

    # Validate columns
    missing = validate_columns(df)
    if missing:
        raise ValueError(
            f"Required columns missing from dataset: {missing}. "
            "The dataset may be corrupted or from an incompatible version."
        )

    # Ensure date column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = df["measured_on"].dt.date

    # Handle temp_diff if missing
    if "temp_diff" not in df.columns:
        df["temp_diff"] = df["module_temp"] - df["ambient_temp"]

    return df


def load_model_metrics(filepath: str | Path | None = None) -> dict:
    """
    Load model_metrics.json.
    
    Parameters
    ----------
    filepath : str or Path, optional
        Path to the JSON file. Defaults to model_metrics.json in the project dir.
    
    Returns
    -------
    dict
        Model metrics including MAE, RMSE, R², feature importances.
    
    Raises
    ------
    FileNotFoundError
        If the metrics file does not exist.
    """
    if filepath is None:
        filepath = _BASE_DIR / "model_metrics.json"
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Model metrics file not found: {filepath}. "
            "Run `python prepare_data.py` first to generate it."
        )

    with open(filepath, "r") as f:
        metrics = json.load(f)

    # Validate expected keys
    expected_keys = ["mae", "rmse", "r2"]
    for k in expected_keys:
        if k not in metrics:
            raise ValueError(f"Missing key '{k}' in model metrics file.")

    return metrics


def get_operating_subset(df: pd.DataFrame, irr_min: float = 200,
                         expected_min: float = 100,
                         actual_min: float = 50) -> pd.DataFrame:
    """
    Filter the dataset to operating-hours observations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    irr_min : float
        Minimum irradiance threshold (W/m²).
    expected_min : float
        Minimum expected power threshold (kW).
    actual_min : float
        Minimum actual power threshold (kW).
    
    Returns
    -------
    pd.DataFrame
        Filtered subset containing only valid operating-hours records.
    """
    mask = (
        (df["irradiance"] > irr_min)
        & (df["expected_power"] > expected_min)
        & (df["total_power"] > actual_min)
    )
    return df[mask].copy()


def get_data_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the dataset.
    
    Returns
    -------
    dict
        Dictionary with key statistics: date range, record counts, etc.
    """
    return {
        "total_records": len(df),
        "date_start": df["measured_on"].min(),
        "date_end": df["measured_on"].max(),
        "irradiance_range": (float(df["irradiance"].min()), float(df["irradiance"].max())),
        "ambient_temp_range": (float(df["ambient_temp"].min()), float(df["ambient_temp"].max())),
        "module_temp_range": (float(df["module_temp"].min()), float(df["module_temp"].max())),
        "total_power_range": (float(df["total_power"].min()), float(df["total_power"].max())),
        "expected_power_range": (float(df["expected_power"].min()), float(df["expected_power"].max())),
    }


def get_historical_baselines(df: pd.DataFrame) -> dict:
    """
    Compute baseline statistics from historical data for simulation defaults.
    Uses daytime operating-hours records to establish realistic ranges.
    
    Returns
    -------
    dict
        Dictionary with baseline ranges for simulation parameters.
    """
    op = get_operating_subset(df)
    if len(op) == 0:
        op = df[df["irradiance"] > 50]
    if len(op) == 0:
        # Fallback defaults if no valid data
        return {
            "irradiance_mean": 500.0,
            "irradiance_std": 200.0,
            "ambient_temp_mean": 25.0,
            "ambient_temp_std": 5.0,
            "module_temp_mean": 35.0,
            "module_temp_std": 8.0,
            "total_power_mean": 2000.0,
            "total_power_std": 800.0,
            "expected_power_mean": 2200.0,
            "expected_power_std": 700.0,
        }

    return {
        "irradiance_mean": float(op["irradiance"].mean()),
        "irradiance_std": float(op["irradiance"].std()),
        "ambient_temp_mean": float(op["ambient_temp"].mean()),
        "ambient_temp_std": float(op["ambient_temp"].std()),
        "module_temp_mean": float(op["module_temp"].mean()),
        "module_temp_std": float(op["module_temp"].std()),
        "total_power_mean": float(op["total_power"].mean()),
        "total_power_std": float(op["total_power"].std()),
        "expected_power_mean": float(op["expected_power"].mean()),
        "expected_power_std": float(op["expected_power"].std()),
    }
