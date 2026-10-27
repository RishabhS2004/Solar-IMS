"""
tests/test_diagnostics.py
=========================
Tests for diagnostic_engine.py covering:
- Normal operation diagnosis
- Soiling-related deviation
- Cloud-related reduction
- Elevated panel temperature
- Equipment-related power loss
- Sensor anomaly & missing data
- Low irradiance handling
- Rule-based suspicion score properties (bounded 0-100)
- Diagnostic limitations disclosure
"""

import numpy as np
from datetime import datetime
from simulation_engine import SimulationEngine, SimulationReading
from diagnostic_engine import DiagnosticEngine, DiagnosticResult, THRESHOLDS, SEVERITY_LEVELS


def test_diagnostic_normal_operation():
    """Verify clean panels under sunny skies diagnose Normal Operation."""
    engine = SimulationEngine()
    engine.load_scenario("Clean Panels – Sunny")
    for _ in range(72):
        engine.step()

    reading = engine.get_current_reading()
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert result.severity == "NORMAL"
    assert result.data_quality == "Valid"
    assert result.is_sufficient_data is True
    assert result.recommended_action == "Continue Monitoring"


def test_diagnostic_soiling_detection():
    """Verify high soiling triggers Soiling-Related Degradation as primary cause."""
    engine = SimulationEngine()
    engine.state.irradiance = 850.0
    engine.state.ambient_temp = 30.0
    engine.state.weather = "Sunny"
    engine.state.soiling_severity = 0.22  # Heavy soiling
    engine.state.active_faults = ["gradual_soiling"]

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert "Soiling" in result.primary_cause
    assert result.primary_score >= 50
    assert result.severity in ("WARNING", "CRITICAL")
    assert any("Soiling severity" in ev for ev in result.evidence_summary)


def test_diagnostic_cloud_cover_detection():
    """Verify cloud-induced drop is attributed to weather, not soiling."""
    engine = SimulationEngine()
    engine.state.irradiance = 320.0
    engine.state.weather = "Cloudy"
    engine.state.soiling_severity = 0.02  # Panels are actually clean
    engine.state.active_faults = ["cloud_cover"]

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert "Cloud" in result.primary_cause or "Weather" in result.primary_cause
    assert result.primary_score >= 40
    assert "Continue Monitoring" in result.recommended_action


def test_diagnostic_equipment_fault_detection():
    """Verify inverter loss is distinguished from pure soiling."""
    engine = SimulationEngine()
    engine.state.irradiance = 800.0
    engine.state.weather = "Sunny"
    engine.state.soiling_severity = 0.03  # Low soiling
    engine.state.active_faults = ["inverter_loss"]

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert "Equipment" in result.primary_cause
    assert result.recommended_action == "Investigate Equipment"


def test_diagnostic_sensor_anomaly():
    """Verify sensor anomaly flags data as anomalous and halts false alerts."""
    engine = SimulationEngine()
    engine.state.active_faults = ["sensor_anomaly"]

    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert result.primary_cause == "Sensor Anomaly"
    assert result.data_quality == "Anomalous"
    assert result.is_sufficient_data is False
    assert result.recommended_action == "Investigate Sensor"


def test_diagnostic_low_irradiance():
    """Verify night or dawn/dusk conditions (<200 W/m²) are handled gracefully."""
    engine = SimulationEngine()
    engine.state.irradiance = 120.0  # Below 200 threshold

    reading = engine._generate_reading_at_step(10)  # Early dawn
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert result.primary_cause == "Low Irradiance"
    assert result.recommended_action == "Continue Monitoring"


def test_diagnostic_suspicion_score_properties():
    """Verify all suspicion scores stay strictly between 0 and 100."""
    engine = SimulationEngine()
    diag_engine = DiagnosticEngine()

    for sc_name in ["Clean Panels – Sunny", "Heavy Soiling – Hot Day", "Partial Shading Event", "Inverter Issue + Soiling"]:
        engine.load_scenario(sc_name)
        for step in [20, 50, 72]:
            rd = engine._generate_reading_at_step(step)
            res = diag_engine.diagnose(rd)
            assert 0.0 <= res.primary_score <= 100.0
            for h in res.hypotheses:
                assert 0.0 <= h.suspicion_score <= 100.0


def test_diagnostic_limitations_disclosure():
    """Verify every diagnostic result includes clear limitation disclaimers."""
    engine = SimulationEngine()
    reading = engine._generate_reading_at_step(72)
    diag_engine = DiagnosticEngine()
    result = diag_engine.diagnose(reading)

    assert len(result.limitations) >= 3
    assert any("simulation" in l.lower() for l in result.limitations)
    assert any("rule-based" in l.lower() or "suspicion" in l.lower() for l in result.limitations)
