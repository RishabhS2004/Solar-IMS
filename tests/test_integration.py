"""
tests/test_integration.py
=========================
Integration and end-to-end lifecycle test:
MONITOR → DETECT → DIAGNOSE → RECOMMEND → INTERVENE → EVALUATE

Also verifies:
- Scenario Comparison Laboratory generation
- Event log recording across full lifecycle
- Entry points live_demo.py and app.py module compatibility
"""

import numpy as np
import pandas as pd
from simulation_engine import SimulationEngine, create_comparison_scenarios
from diagnostic_engine import DiagnosticEngine
from recommendation_engine import RecommendationEngine, RecommendationConfig


def test_end_to_end_operational_lifecycle():
    """
    Test complete operator workflow:
    MONITOR → DETECT → DIAGNOSE → RECOMMEND → INTERVENE → EVALUATE
    """
    # 1. MONITOR: Initialize plant with moderate soiling under sunny conditions
    engine = SimulationEngine()
    engine.load_scenario("Moderate Soiling – Sunny")
    # Advance to midday
    for _ in range(72):
        engine.step()

    reading = engine.get_current_reading()
    assert reading.clean_power > 1500.0  # MW-scale midday solar
    assert reading.soiling_severity >= 0.08

    # 2. DETECT: Performance ratio is degraded due to dirt
    assert reading.performance_ratio < 0.95
    assert reading.soiling_loss_pct > 5.0

    # 3. DIAGNOSE: Rule-based diagnostic engine identifies soiling
    diag_engine = DiagnosticEngine()
    diag = diag_engine.diagnose(reading)

    assert "Soiling" in diag.primary_cause
    assert diag.primary_score >= 40.0
    assert len(diag.evidence_summary) > 0

    # 4. RECOMMEND: Economic engine calculates positive ROI and recommends action
    rec_cfg = RecommendationConfig(
        cleaning_cost=15000.0,
        energy_rate=5.5,
        evaluation_horizon_hours=720,  # 30 days
        cleaning_effectiveness=0.85
    )
    rec_engine = RecommendationEngine(rec_cfg)
    rec = rec_engine.recommend(reading, diag.severity, diag.primary_cause, rec_cfg)

    assert "Cleaning" in rec.action
    assert rec.economics.net_benefit > 0

    # 5. INTERVENE: Operator clicks "Simulate Cleaning"
    pre_soiling = engine.state.soiling_severity
    pre_power = reading.measured_power
    interv = engine.simulate_cleaning(effectiveness=rec_cfg.cleaning_effectiveness)

    assert interv.intervention_type == "cleaning"
    assert interv.post_soiling < pre_soiling

    # 6. EVALUATE: Post-intervention state shows recovered power and healthy PR
    post_reading = engine.get_current_reading()
    assert post_reading.measured_power > pre_power
    assert post_reading.performance_ratio > reading.performance_ratio

    # Diagnostics after cleaning should reflect improvement
    post_diag = diag_engine.diagnose(post_reading)
    assert post_diag.severity in ("NORMAL", "ADVISORY")


def test_scenario_comparison_laboratory_output():
    """Verify create_comparison_scenarios generates consistent side-by-side data."""
    results = create_comparison_scenarios([
        "Clean Panels – Sunny",
        "Moderate Soiling – Sunny",
        "Cloudy Day – Light Soiling"
    ])

    assert len(results) == 3
    for r in results:
        assert "scenario" in r
        assert "clean_power" in r
        assert "measured_power" in r
        assert "soiling_severity" in r
        assert "soiling_label" in r


def test_event_log_persistence_across_actions():
    """Verify event log records all key actions throughout operation."""
    engine = SimulationEngine()
    engine.load_scenario("Moderate Soiling – Sunny")
    engine.step()
    engine.set_faults(["inverter_loss"])
    engine.simulate_cleaning(0.85)
    engine.delay_cleaning()

    df_log = engine.get_event_log_dataframe()
    assert len(df_log) >= 4
    event_types = df_log["event_type"].tolist()
    assert "scenario_changed" in event_types
    assert "fault_injected" in event_types
    assert "cleaning_simulated" in event_types
    assert "cleaning_delayed" in event_types


def test_streamlit_apptest_simulator_lifecycle():
    """
    End-to-end integration test using Streamlit AppTest on live_demo.py:
    - Verifies initial boot at step 12 (07:00).
    - Verifies Start button toggles sim_is_running and advances clock to 07:05 (step 13).
    - Verifies subsequent automatic run advances to 07:10 (step 14).
    - Verifies Pause button stops automatic progression.
    - Verifies +5m advances simulation step by 1.
    - Verifies -5m rewinds simulation step by 1.
    - Verifies Reset button returns simulation to step 12 (07:00).
    """
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("live_demo.py").run()
    assert len(at.exception) == 0

    eng = at.session_state["sim_engine"]
    assert eng.state.current_step == 12
    assert eng.state.current_time.strftime("%H:%M") == "07:00"
    assert at.session_state["sim_is_running"] is False

    # 1. Click Start
    start_btn = at.button(key="btn_sim_start")
    assert start_btn is not None
    start_btn.click().run()
    assert len(at.exception) == 0
    assert at.session_state["sim_is_running"] is True
    assert eng.state.current_step == 13
    assert eng.state.current_time.strftime("%H:%M") == "07:05"

    # 2. Simulate next automatic tick after time interval
    import time
    time.sleep(0.9)
    at.run()
    assert len(at.exception) == 0
    assert eng.state.current_step == 14
    assert eng.state.current_time.strftime("%H:%M") == "07:10"

    # 3. Click Pause
    pause_btn = at.button(key="btn_sim_pause")
    assert pause_btn is not None
    pause_btn.click().run()
    assert len(at.exception) == 0
    assert at.session_state["sim_is_running"] is False
    step_paused = eng.state.current_step

    # Verify no advancement while paused
    at.run()
    assert eng.state.current_step == step_paused

    # 4. Step +5m
    fwd_btn = at.button(key="btn_sim_step_fwd")
    fwd_btn.click().run()
    assert len(at.exception) == 0
    assert eng.state.current_step == step_paused + 1

    # 5. Step -5m
    back_btn = at.button(key="btn_sim_step_back")
    back_btn.click().run()
    assert len(at.exception) == 0
    assert eng.state.current_step == step_paused

    # 6. Reset
    reset_btn = at.button(key="btn_sim_reset")
    reset_btn.click().run()
    assert len(at.exception) == 0
    assert eng.state.current_step == 12
    assert eng.state.current_time.strftime("%H:%M") == "07:00"
    assert at.session_state["sim_is_running"] is False


def test_regression_preset_switching_preserves_1x_progression_rate():
    """
    Regression Test 1, 2, 3:
    1. Baseline 1x progression rate before preset change is 1 step per tick.
    2. Effective rate remains strictly 1 step per tick after switching to Moderate Soiling.
    3. Switching through several presets preserves 1x rate.
    4. Returning to Clean Panels maintains 1x baseline without acceleration.
    """
    from streamlit.testing.v1 import AppTest
    import time
    at = AppTest.from_file("live_demo.py").run()
    assert len(at.exception) == 0

    # Start simulation
    at.button(key="btn_sim_start").click().run()
    eng = at.session_state["sim_engine"]

    # Measure tick rate before preset change
    step_pre = eng.state.current_step
    time.sleep(0.9)
    at.run()
    assert eng.state.current_step - step_pre == 1, "Baseline rate must be 1 step/tick"

    # Switch preset to Moderate Soiling – Sunny
    scen_sel = at.selectbox(key="sim_ui_scen_select")
    scen_sel.select("Moderate Soiling – Sunny").run()
    assert eng.state.scenario_name == "Moderate Soiling – Sunny"

    # Verify rate after preset change is strictly 1 step per tick (not accelerated)
    for _ in range(3):
        step_curr = eng.state.current_step
        time.sleep(0.9)
        at.run()
        diff = eng.state.current_step - step_curr
        assert diff == 1, f"Expected 1 step per tick after preset change, but got {diff}"

    # Cycle through several other presets
    for scen_name in ["Heavy Soiling – Hot Day", "Inverter Issue + Soiling", "Cloudy Day – Light Soiling"]:
        scen_sel.select(scen_name).run()
        assert eng.state.scenario_name == scen_name
        step_curr = eng.state.current_step
        time.sleep(0.9)
        at.run()
        diff = eng.state.current_step - step_curr
        assert diff == 1, f"Expected 1 step/tick in {scen_name}, got {diff}"

    # Return to original preset: Clean Panels – Sunny
    scen_sel.select("Clean Panels – Sunny").run()
    assert eng.state.scenario_name == "Clean Panels – Sunny"
    for _ in range(2):
        step_curr = eng.state.current_step
        time.sleep(0.9)
        at.run()
        diff = eng.state.current_step - step_curr
        assert diff == 1, f"Expected 1 step/tick upon returning to Clean Panels, got {diff}"


def test_regression_clock_speed_multipliers():
    """
    Regression Test 4:
    Verify 1x, 5x, and 10x speeds maintain intended relative progression rates
    and that preset changes do not reset or corrupt the speed multiplier.
    """
    from streamlit.testing.v1 import AppTest
    import time
    at = AppTest.from_file("live_demo.py").run()
    at.button(key="btn_sim_start").click().run()
    eng = at.session_state["sim_engine"]

    # 1. Test 5x speed
    speed_sel = at.selectbox(key="sim_ui_speed_select")
    speed_sel.select(5).run()
    step_before = eng.state.current_step
    time.sleep(0.9)
    at.run()
    diff_5x = eng.state.current_step - step_before
    assert diff_5x == 5, f"5x speed must advance 5 steps/tick, got {diff_5x}"

    # 2. Test 10x speed
    speed_sel.select(10).run()
    step_before = eng.state.current_step
    time.sleep(0.9)
    at.run()
    diff_10x = eng.state.current_step - step_before
    assert diff_10x == 10, f"10x speed must advance 10 steps/tick, got {diff_10x}"

    # 3. Switch preset while at 5x: speed must be preserved
    speed_sel.select(5).run()
    scen_sel = at.selectbox(key="sim_ui_scen_select")
    scen_sel.select("Moderate Soiling – Sunny").run()
    assert at.session_state["sim_speed"] == 5
    step_before = eng.state.current_step
    time.sleep(0.9)
    at.run()
    assert eng.state.current_step - step_before == 5

    # 4. Return to 1x: rate must return to 1 step/tick
    speed_sel.select(1).run()
    step_before = eng.state.current_step
    time.sleep(0.9)
    at.run()
    assert eng.state.current_step - step_before == 1


def test_regression_repeated_preset_changes_no_timer_accumulation():
    """
    Regression Test 5:
    Repeated rapid preset changes must not create duplicate timers or cause runaway progression.
    """
    from streamlit.testing.v1 import AppTest
    import time
    at = AppTest.from_file("live_demo.py").run()
    at.button(key="btn_sim_start").click().run()
    eng = at.session_state["sim_engine"]
    scen_sel = at.selectbox(key="sim_ui_scen_select")

    # Rapid succession of preset changes
    presets = [
        "Moderate Soiling – Sunny",
        "Heavy Soiling – Hot Day",
        "Cloudy Day – Light Soiling",
        "Partial Shading Event",
        "Clean Panels – Sunny"
    ]
    for p in presets:
        scen_sel.select(p).run()
        assert eng.state.scenario_name == p

    # Now verify single tick advances exactly 1 step
    step_before = eng.state.current_step
    time.sleep(0.9)
    at.run()
    diff = eng.state.current_step - step_before
    assert diff == 1, f"Expected 1 step after rapid preset cycling, got {diff}"

