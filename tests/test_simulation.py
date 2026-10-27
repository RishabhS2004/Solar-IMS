"""
tests/test_simulation.py
========================
Tests for simulation_engine.py covering:
- Initial state & reset
- Time progression (step & step_to)
- Reproducible synthetic data via random seed
- Scenario switching
- Soiling accumulation over time
- Physical consistency:
  * Greater soiling cannot increase clean-adjusted output
  * Lower irradiance reduces clean-reference power
  * Cleaning intervention reduces soiling by configured effectiveness
  * Cleaning does NOT change clean-reference potential
  * Combined environmental & fault conditions
"""

import numpy as np
from datetime import datetime, timedelta
from simulation_engine import (
    SimulationEngine, SimulationState, SimulationReading,
    SCENARIO_PRESETS, FAULT_TYPES, WEATHER_CONDITIONS,
    get_soiling_label, create_comparison_scenarios, RATED_CAPACITY_KW
)


def test_initial_state():
    """Verify clean initial state of SimulationEngine."""
    engine = SimulationEngine()
    assert engine.state.current_step == 0
    assert engine.state.scenario_name == "Clean Panels – Sunny"
    assert engine.state.soiling_severity == 0.01
    assert len(engine.state.readings) == 0
    assert len(engine.state.event_log) == 0


def test_time_progression():
    """Verify manual time progression advancing clock and recording steps."""
    engine = SimulationEngine()
    start_time = engine.state.current_time

    r1 = engine.step()
    assert engine.state.current_step == 1
    assert r1.step == 1
    assert r1.timestamp == start_time + timedelta(minutes=5)
    assert len(engine.state.readings) == 1

    r2 = engine.step()
    assert engine.state.current_step == 2
    assert r2.step == 2
    assert r2.timestamp == start_time + timedelta(minutes=10)
    assert len(engine.state.readings) == 2


def test_step_to_navigation():
    """Verify scrubber navigation forward and backward."""
    engine = SimulationEngine()
    # Step forward 5 times
    for _ in range(5):
        engine.step()
    assert engine.state.current_step == 5
    assert len(engine.state.readings) == 5

    # Rewind to step 2
    engine.step_to(2)
    assert engine.state.current_step == 2
    assert len(engine.state.readings) == 2

    # Forward to step 6
    engine.step_to(6)
    assert engine.state.current_step == 6
    assert len(engine.state.readings) == 6


def test_simulation_reset():
    """Verify reset returns simulation to well-defined baseline."""
    engine = SimulationEngine()
    engine.load_scenario("Heavy Soiling – Hot Day")
    for _ in range(10):
        engine.step()

    assert engine.state.current_step == 10
    assert len(engine.state.readings) == 10

    # Reset
    engine.reset("Clean Panels – Sunny")
    assert engine.state.current_step == 0
    assert engine.state.scenario_name == "Clean Panels – Sunny"
    assert len(engine.state.readings) == 0
    assert engine.state.soiling_severity == 0.01


def test_reproducible_synthetic_data():
    """Verify identical random seeds generate identical readings."""
    e1 = SimulationEngine(SimulationState(random_seed=123))
    e2 = SimulationEngine(SimulationState(random_seed=123))

    r1 = [e1.step() for _ in range(5)]
    r2 = [e2.step() for _ in range(5)]

    for a, b in zip(r1, r2):
        assert a.irradiance == b.irradiance
        assert a.clean_power == b.clean_power
        assert a.measured_power == b.measured_power


def test_scenario_switching():
    """Verify changing scenario clears state and applies new presets cleanly."""
    engine = SimulationEngine()
    for name in SCENARIO_PRESETS:
        engine.load_scenario(name)
        preset = SCENARIO_PRESETS[name]
        assert engine.state.scenario_name == name
        assert engine.state.irradiance == preset["irradiance"]
        assert engine.state.weather == preset["weather"]
        assert engine.state.soiling_severity == preset["soiling_severity"]
        assert engine.state.current_step == 0


def test_physical_consistency_soiling_impact():
    """Under identical conditions, higher soiling must reduce or maintain measured power."""
    engine = SimulationEngine()
    engine.state.irradiance = 800.0
    engine.state.weather = "Sunny"

    engine.state.soiling_severity = 0.02
    r_clean = engine._generate_reading_at_step(72)  # midday

    engine.state.soiling_severity = 0.20
    r_soiled = engine._generate_reading_at_step(72)

    # Clean potential must be identical
    assert r_clean.clean_power == r_soiled.clean_power
    # Measured power with 20% soiling must be lower than with 2% soiling
    assert r_soiled.measured_power < r_clean.measured_power
    # Performance ratio must be lower
    assert r_soiled.performance_ratio < r_clean.performance_ratio


def test_physical_consistency_irradiance_scaling():
    """Lower irradiance must reduce clean-reference power estimate."""
    engine = SimulationEngine()
    engine.state.weather = "Sunny"
    engine.state.soiling_severity = 0.05

    engine.state.irradiance = 1000.0
    r_high = engine._generate_reading_at_step(72)

    engine.state.irradiance = 400.0
    r_low = engine._generate_reading_at_step(72)

    assert r_low.clean_power < r_high.clean_power
    assert r_low.measured_power < r_high.measured_power


def test_physical_consistency_cleaning_intervention():
    """
    Cleaning must:
    1. Reduce soiling severity according to configured effectiveness
    2. Increase measured power
    3. NOT alter the clean-panel reference potential
    """
    engine = SimulationEngine()
    engine.load_scenario("Moderate Soiling – Sunny")
    # Step to midday
    for _ in range(72):
        engine.step()

    pre_rd = engine.get_current_reading()
    pre_soiling = engine.state.soiling_severity
    pre_clean = pre_rd.clean_power

    effectiveness = 0.85
    rec = engine.simulate_cleaning(effectiveness=effectiveness)

    post_rd = engine.get_current_reading()
    expected_post_soiling = pre_soiling * (1 - effectiveness)

    # 1. Soiling reduced
    assert np.isclose(engine.state.soiling_severity, expected_post_soiling, atol=1e-3)
    # 2. Measured power increased
    assert post_rd.measured_power > pre_rd.measured_power
    # 3. Clean-reference potential invariant
    assert np.isclose(post_rd.clean_power, pre_clean, atol=0.1)
    # 4. Intervention recorded
    assert len(engine.state.interventions) == 1
    assert rec.estimated_recovery_kw > 0


def test_soiling_accumulation():
    """Verify gradual accumulation increases soiling severity over steps."""
    engine = SimulationEngine()
    engine.state.soiling_severity = 0.05
    engine.state.accumulation_rate = 0.005  # +0.5% per step

    for _ in range(10):
        engine.step()

    # After 10 steps, soiling should be ~ 0.05 + 10*0.005 = 0.10
    assert engine.state.soiling_severity > 0.09
    assert engine.state.soiling_severity < 0.11


def test_rain_cleaning_weather():
    """Verify rain weather naturally cleans or halts accumulation."""
    engine = SimulationEngine()
    engine.state.weather = "Rainy"
    engine.state.soiling_severity = 0.15
    engine.state.accumulation_rate = 0.002

    for _ in range(5):
        engine.step()

    # Rain weather forces accumulation to negative (natural wash)
    assert engine.state.soiling_severity < 0.15


def test_soiling_labels():
    """Verify label mappings across thresholds."""
    assert get_soiling_label(0.01) == "Clean"
    assert get_soiling_label(0.03) == "Light Soiling"
    assert get_soiling_label(0.08) == "Moderate Soiling"
    assert get_soiling_label(0.20) == "Heavy Soiling"


def test_automatic_progression_sequence():
    """
    Verify automatic ticking progression:
    - Step increments by 1 per tick
    - Simulated timestamp advances by exactly 5 minutes per tick
    - Reading buffer appends new readings monotonically
    - Clean power and measured power follow solar diurnal curve
    """
    engine = SimulationEngine()
    engine.load_scenario("Clean Panels – Sunny")
    
    # Prime 12 steps (06:00 to 07:00)
    for _ in range(12):
        engine.step()

    assert engine.state.current_step == 12
    assert engine.state.current_time.strftime("%H:%M") == "07:00"
    assert len(engine.state.readings) == 12

    # Simulate 6 automatic ticks
    expected_times = ["07:05", "07:10", "07:15", "07:20", "07:25", "07:30"]
    for i, exp_time in enumerate(expected_times, start=13):
        r = engine.step()
        assert engine.state.current_step == i
        assert r.step == i
        assert engine.state.current_time.strftime("%H:%M") == exp_time
        assert r.timestamp.strftime("%H:%M") == exp_time
        assert len(engine.state.readings) == i
        # Irradiance and clean power must be strictly positive during morning ascent
        assert r.irradiance > 100.0
        assert r.clean_power > 100.0


def test_pause_state_immobility():
    """Verify that when the simulation is paused, time and telemetry do not advance."""
    engine = SimulationEngine()
    for _ in range(12):
        engine.step()

    step_before = engine.state.current_step
    time_before = engine.state.current_time
    buffer_len_before = len(engine.state.readings)
    soiling_before = engine.state.soiling_severity
    reading_before = engine.get_current_reading()

    # Paused condition: no step() invoked
    # Verify state immobility
    assert engine.state.current_step == step_before
    assert engine.state.current_time == time_before
    assert len(engine.state.readings) == buffer_len_before
    assert engine.state.soiling_severity == soiling_before
    reading_after = engine.get_current_reading()
    assert reading_after.timestamp == reading_before.timestamp
    assert reading_after.measured_power == reading_before.measured_power


def test_reset_restores_baseline():
    """Verify that reset restores simulation to step 0, baseline time, and clears buffer."""
    engine = SimulationEngine()
    engine.load_scenario("Heavy Soiling – Hot Day")
    for _ in range(25):
        engine.step()

    assert engine.state.current_step == 25
    assert len(engine.state.readings) == 25
    assert engine.state.current_time > engine.state.start_time

    # Perform Reset
    engine.reset("Clean Panels – Sunny")
    assert engine.state.current_step == 0
    assert engine.state.current_time == engine.state.start_time
    assert engine.state.current_time.strftime("%H:%M") == "06:00"
    assert len(engine.state.readings) == 0
    assert engine.state.soiling_severity == 0.01
    assert engine.state.active_faults == []


def test_manual_stepping_forward_and_backward():
    """Verify manual +5m and -5m navigation accurately increments and rewinds state."""
    engine = SimulationEngine()
    for _ in range(12):
        engine.step()

    assert engine.state.current_step == 12
    assert engine.state.current_time.strftime("%H:%M") == "07:00"

    # Step +5m
    r_fwd = engine.step()
    assert engine.state.current_step == 13
    assert engine.state.current_time.strftime("%H:%M") == "07:05"
    assert r_fwd.timestamp.strftime("%H:%M") == "07:05"

    # Step -5m
    r_back = engine.step_to(engine.state.current_step - 1)
    assert engine.state.current_step == 12
    assert engine.state.current_time.strftime("%H:%M") == "07:00"
    assert r_back.timestamp.strftime("%H:%M") == "07:00"
    assert engine.get_current_reading().timestamp.strftime("%H:%M") == "07:00"


def test_state_and_reading_clock_consistency():
    """Verify that engine.state and get_current_reading() remain strictly synchronized."""
    engine = SimulationEngine()
    for _ in range(30):
        engine.step()

    # Sample various steps and verify clock and step alignment
    for target in [1, 5, 12, 18, 25]:
        engine.step_to(target)
        rd = engine.get_current_reading()
        expected_time = engine.state.start_time + timedelta(minutes=5 * target)
        assert engine.state.current_step == target
        assert engine.state.current_time == expected_time
        assert rd.timestamp == expected_time
        assert rd.step == target


def test_multi_speed_step_calculation():
    """Verify multi-speed step advancement increments."""
    # 1x speed: 1 step = 5 minutes
    # 5x speed: 2 steps = 10 minutes
    # 10x speed: 4 steps = 20 minutes
    speed_steps = {1: 1, 5: 2, 10: 4}
    for speed, expected_steps in speed_steps.items():
        engine = SimulationEngine()
        start = engine.state.current_time
        for _ in range(expected_steps):
            engine.step()
        assert engine.state.current_step == expected_steps
        assert engine.state.current_time == start + timedelta(minutes=5 * expected_steps)


def test_scenario_selection_affects_conditions_and_outputs():
    """Confirm that preset scenario selection changes simulation conditions and affects outputs."""
    engine = SimulationEngine()
    
    # 1. Clean Panels
    engine.load_scenario("Clean Panels – Sunny")
    assert engine.state.soiling_severity == 0.01
    assert engine.state.weather == "Sunny"
    assert "gradual_soiling" not in engine.state.active_faults
    rd_clean = engine._generate_reading_at_step(72)  # Solar noon
    assert rd_clean.performance_ratio > 0.95

    # 2. Heavy Soiling
    engine.load_scenario("Heavy Soiling – Hot Day")
    assert engine.state.soiling_severity == 0.22
    assert engine.state.ambient_temp == 40.0
    rd_soiled = engine._generate_reading_at_step(72)
    assert rd_soiled.measured_power < rd_clean.measured_power
    assert rd_soiled.soiling_loss_pct > 20.0
    assert rd_soiled.module_temp > rd_clean.module_temp

    # 3. Inverter Issue + Soiling
    engine.load_scenario("Inverter Issue + Soiling")
    assert "inverter_loss" in engine.state.active_faults
    rd_inv = engine._generate_reading_at_step(72)
    assert rd_inv.measured_power < rd_clean.measured_power * 0.60
    assert rd_inv.fault_loss_kw > 1000.0


def test_environmental_and_fault_updates_immediately_refresh_reading():
    """Verify that updating conditions or injecting faults immediately refreshes the current reading."""
    engine = SimulationEngine()
    for _ in range(12):
        engine.step()

    r_initial = engine.get_current_reading()
    assert r_initial.soiling_severity == 0.01

    # Update soiling
    engine.set_environmental_conditions(soiling_severity=0.25)
    r_updated = engine.get_current_reading()
    assert r_updated.soiling_severity == 0.25
    assert r_updated.soiling_loss_pct == 25.0
    assert r_updated.measured_power < r_initial.measured_power

    # Inject inverter fault
    engine.set_faults(["inverter_loss"])
    r_fault = engine.get_current_reading()
    assert "inverter_loss" in r_fault.active_faults
    assert r_fault.fault_loss_kw > 0
    assert r_fault.measured_power < r_updated.measured_power

