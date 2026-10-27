"""
tests/test_recommendations.py
=============================
Tests for recommendation_engine.py covering:
- Positive economic net benefit triggering cleaning recommendation
- Negative economic benefit preventing cleaning recommendation
- Equipment fault prevention of inappropriate cleaning
- Sensor anomaly / missing data producing Insufficient Data
- Boundary testing on cleaning effectiveness (0.0 and 1.0)
- Financial math consistency (Energy value = kWh * rate, Net = Value - Cost)
- Payback period calculations
"""

import numpy as np
from simulation_engine import SimulationEngine
from diagnostic_engine import DiagnosticEngine
from recommendation_engine import (
    RecommendationEngine, RecommendationConfig,
    CleaningRecommendation, EconomicAnalysis
)


def test_positive_net_benefit_recommends_cleaning():
    """When recoverable energy value exceeds cleaning cost, cleaning is recommended."""
    engine = SimulationEngine()
    engine.state.irradiance = 900.0
    engine.state.weather = "Sunny"
    engine.state.soiling_severity = 0.18  # Substantial soiling

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    diag = diag_engine.diagnose(reading)

    cfg = RecommendationConfig(
        cleaning_cost=10000.0,
        energy_rate=6.0,
        evaluation_horizon_hours=720,  # 30 days
        cleaning_effectiveness=0.85,
        min_benefit_ratio=1.2
    )

    rec_engine = RecommendationEngine(cfg)
    rec = rec_engine.recommend(reading, diag.severity, diag.primary_cause, cfg)

    assert "Cleaning Recommended" in rec.action
    assert rec.economics.net_benefit > 0
    assert rec.economics.benefit_ratio >= 1.2
    assert "cleaning cost" in rec.explanation.lower()


def test_negative_net_benefit_prevents_cleaning():
    """When cleaning cost is very high relative to energy value, cleaning is NOT recommended."""
    engine = SimulationEngine()
    engine.state.irradiance = 800.0
    engine.state.weather = "Sunny"
    engine.state.soiling_severity = 0.02  # Clean / minimal soiling

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    diag = diag_engine.diagnose(reading)

    cfg = RecommendationConfig(
        cleaning_cost=80000.0,  # Prohibitively high cleaning cost
        energy_rate=4.0,
        evaluation_horizon_hours=168,  # 7 days only
        cleaning_effectiveness=0.85
    )

    rec_engine = RecommendationEngine(cfg)
    rec = rec_engine.recommend(reading, diag.severity, diag.primary_cause, cfg)

    assert rec.action == "Continue Monitoring"
    assert rec.economics.net_benefit <= 0
    assert any("exceeds" in f.lower() for f in rec.factors_against)


def test_equipment_fault_prevents_cleaning():
    """When equipment loss is primary diagnosis, recommendation tells operator to investigate equipment."""
    engine = SimulationEngine()
    engine.state.irradiance = 850.0
    engine.state.weather = "Sunny"
    engine.state.soiling_severity = 0.02
    engine.state.active_faults = ["inverter_loss"]

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    diag = diag_engine.diagnose(reading)

    rec_engine = RecommendationEngine()
    rec = rec_engine.recommend(reading, diag.severity, diag.primary_cause)

    assert "Investigate Equipment" in rec.action
    assert "inverter" in rec.explanation.lower() or "equipment" in rec.explanation.lower()


def test_insufficient_data_recommendation():
    """Sensor anomaly produces Insufficient Data recommendation without false alarm."""
    engine = SimulationEngine()
    engine.state.active_faults = ["sensor_anomaly"]

    reading = engine._generate_reading_at_step(72)
    rec_engine = RecommendationEngine()
    rec = rec_engine.recommend(reading, "WARNING", "Sensor Anomaly")

    assert rec.action == "Insufficient Data"
    assert "sensor" in rec.explanation.lower()


def test_cleaning_effectiveness_boundaries():
    """Test boundary conditions: 0% and 100% effectiveness."""
    engine = SimulationEngine()
    engine.state.irradiance = 850.0
    engine.state.soiling_severity = 0.15
    reading = engine._generate_reading_at_step(72)

    # 0% effectiveness -> 0 recoverable energy
    cfg_zero = RecommendationConfig(cleaning_effectiveness=0.0)
    rec_zero = RecommendationEngine(cfg_zero).recommend(reading, "WARNING", "Soiling-Related Degradation", cfg_zero)
    assert rec_zero.economics.recoverable_power_kw == 0.0
    assert rec_zero.economics.recoverable_energy_kwh == 0.0

    # 100% effectiveness -> full recovery of soiling loss
    cfg_full = RecommendationConfig(cleaning_effectiveness=1.0)
    rec_full = RecommendationEngine(cfg_full).recommend(reading, "WARNING", "Soiling-Related Degradation", cfg_full)
    assert rec_full.economics.recoverable_power_kw > 0.0
    assert np.isclose(rec_full.economics.recoverable_power_kw, reading.soiling_loss_kw, atol=1.0)


def test_economic_math_consistency():
    """Verify exact formula relationships in EconomicAnalysis."""
    engine = SimulationEngine()
    engine.state.irradiance = 900.0
    engine.state.soiling_severity = 0.12
    reading = engine._generate_reading_at_step(72)

    cfg = RecommendationConfig(
        cleaning_cost=15000.0,
        energy_rate=5.5,
        evaluation_horizon_hours=720,
        cleaning_effectiveness=0.85
    )

    rec_engine = RecommendationEngine(cfg)
    rec = rec_engine.recommend(reading, "WARNING", "Soiling-Related Degradation", cfg)
    econ = rec.economics

    # Energy value = kWh * rate
    expected_val = econ.recoverable_energy_kwh * cfg.energy_rate
    assert np.isclose(econ.energy_value, expected_val, atol=0.1)

    # Net benefit = Energy value - cleaning cost
    expected_net = econ.energy_value - cfg.cleaning_cost
    assert np.isclose(econ.net_benefit, expected_net, atol=0.1)

    # Benefit ratio = Energy value / cleaning cost
    expected_ratio = econ.energy_value / cfg.cleaning_cost
    assert np.isclose(econ.benefit_ratio, expected_ratio, atol=0.1)
