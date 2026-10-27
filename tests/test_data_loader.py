"""
tests/test_data_loader.py
=========================
Tests for data_loader.py covering:
- Loading processed_output.csv
- Loading model_metrics.json
- Column validation and missing column detection
- Operating subset filtering
- Historical baselines computation
"""

import pandas as pd
from pathlib import Path
from data_loader import (
    load_processed_data, load_model_metrics,
    validate_columns, get_operating_subset,
    get_data_stats, get_historical_baselines,
    REQUIRED_COLUMNS
)

_BASE_DIR = Path(__file__).parent.parent


def test_validate_columns():
    """Verify missing column detection."""
    df_valid = pd.DataFrame(columns=REQUIRED_COLUMNS)
    assert validate_columns(df_valid) == []

    df_incomplete = pd.DataFrame(columns=["measured_on", "irradiance"])
    missing = validate_columns(df_incomplete)
    assert "total_power" in missing
    assert "expected_power" in missing


def test_load_processed_data_success():
    """Verify loading real processed_output.csv."""
    df = load_processed_data(_BASE_DIR / "processed_output.csv")
    assert len(df) > 100000
    assert "measured_on" in df.columns
    assert "total_power" in df.columns
    assert "expected_power" in df.columns
    assert "recommendation" in df.columns


def test_load_processed_data_missing_file():
    """Verify FileNotFoundError on missing file."""
    raised = False
    try:
        load_processed_data(_BASE_DIR / "non_existent_file.csv")
    except FileNotFoundError:
        raised = True
    assert raised, "Expected FileNotFoundError when loading non-existent file"


def test_load_model_metrics():
    """Verify loading model_metrics.json."""
    metrics = load_model_metrics(_BASE_DIR / "model_metrics.json")
    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert metrics["r2"] > 0.90


def test_get_operating_subset():
    """Verify operational daylight filtering."""
    df = load_processed_data(_BASE_DIR / "processed_output.csv")
    op = get_operating_subset(df, irr_min=200, expected_min=100, actual_min=50)

    assert len(op) < len(df)
    assert (op["irradiance"] > 200).all()
    assert (op["expected_power"] > 100).all()
    assert (op["total_power"] > 50).all()


def test_historical_baselines():
    """Verify calculation of baseline statistics from historical records."""
    df = load_processed_data(_BASE_DIR / "processed_output.csv")
    baselines = get_historical_baselines(df)

    assert "irradiance_mean" in baselines
    assert baselines["irradiance_mean"] > 200
    assert "total_power_mean" in baselines
    assert baselines["total_power_mean"] > 100
