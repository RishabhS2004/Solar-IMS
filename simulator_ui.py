"""
simulator_ui.py
===============
Reusable Streamlit UI component for the Solar IMS Live Simulation & Diagnostics.
Shared by both app.py (Full Dashboard) and live_demo.py (Standalone Simulator).

Implements:
- Module A: Live Simulation Control Centre
- Module B: Synthetic Solar Operating Conditions
- Module C: Soiling and Power-Loss Simulator
- Module D: Fault Injection and Diagnostics
- Module E: Cleaning Recommendation Engine
- Module F: Simulated Interventions (Before/After comparison)
- Module G: Historical Replay vs Synthetic Scenarios
- Module H: Scenario Comparison Laboratory
- Module I: Event Log and CSV Export
- Module J: Alerts and Key Performance Indicators
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io

from simulation_engine import (
    SimulationEngine, SimulationState, SimulationReading,
    SCENARIO_PRESETS, FAULT_TYPES, WEATHER_CONDITIONS,
    get_soiling_label, create_comparison_scenarios, RATED_CAPACITY_KW
)
from diagnostic_engine import DiagnosticEngine, DiagnosticResult, SEVERITY_LEVELS
from recommendation_engine import RecommendationEngine, RecommendationConfig, CleaningRecommendation
from data_loader import load_processed_data, load_model_metrics, get_operating_subset


def sync_session_state_to_scenario(engine: SimulationEngine):
    """Synchronize Streamlit widget session state keys to match the engine's current state."""
    st.session_state.slider_irr = float(round(engine.state.irradiance / 25.0) * 25.0)
    st.session_state.slider_amb = float(round(engine.state.ambient_temp))
    st.session_state.select_weather = engine.state.weather if engine.state.weather in WEATHER_CONDITIONS else "Sunny"
    st.session_state.slider_soiling = float(max(0.0, min(0.40, round(engine.state.soiling_severity, 2))))
    st.session_state.slider_acc = float(max(-0.5, min(1.0, round((engine.state.accumulation_rate * 100) / 0.05) * 0.05)))
    st.session_state.slider_shading = float(max(0.0, min(0.80, round(engine.state.shading_severity / 0.05) * 0.05)))
    
    selectable_faults = [k for k in FAULT_TYPES if k != "none"]
    fault_labels = {k: FAULT_TYPES[k]["label"] for k in selectable_faults}
    st.session_state.fault_injector_multiselect = [
        fault_labels[k] for k in engine.state.active_faults if k in fault_labels
    ]
    total_steps = max(len(engine.state.readings), 1)
    st.session_state.sim_ui_scrubber = min(engine.state.current_step, total_steps)


def init_simulator_state():
    """Initialize or verify all simulation session state variables."""
    if "sim_engine" not in st.session_state:
        st.session_state.sim_engine = SimulationEngine()
        # Pre-generate 12 steps so the charts start with initial history
        for _ in range(12):
            st.session_state.sim_engine.step()

    if "diag_engine" not in st.session_state:
        st.session_state.diag_engine = DiagnosticEngine()

    if "rec_config" not in st.session_state:
        st.session_state.rec_config = RecommendationConfig()

    if "rec_engine" not in st.session_state:
        st.session_state.rec_engine = RecommendationEngine(st.session_state.rec_config)

    if "sim_is_running" not in st.session_state:
        st.session_state.sim_is_running = False

    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = 1

    if "sim_last_tick_time" not in st.session_state:
        st.session_state.sim_last_tick_time = 0.0

    if "sim_intervention_snapshot" not in st.session_state:
        st.session_state.sim_intervention_snapshot = None

    if "sim_data_source_mode" not in st.session_state:
        st.session_state.sim_data_source_mode = "synthetic"  # or "historical"

    if "sim_active_tab" not in st.session_state:
        st.session_state.sim_active_tab = "🕹️ Live Simulator"

    if "slider_irr" not in st.session_state:
        sync_session_state_to_scenario(st.session_state.sim_engine)


def get_ui_styles():
    """Return CSS styling for the industrial monitoring look and feel."""
    return """
    <style>
    /* Dark industrial simulation theme */
    .sim-header-box {
        background: linear-gradient(135deg, #121c27 0%, #1a2736 100%);
        border: 1px solid #283a4c;
        border-radius: 12px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .sim-badge-container {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }
    .sim-badge-synthetic {
        background: rgba(245, 166, 35, 0.15);
        border: 1px solid #F5A623;
        color: #F5A623;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 4px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .sim-badge-historical {
        background: rgba(66, 165, 245, 0.15);
        border: 1px solid #42A5F5;
        color: #42A5F5;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 4px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .sim-badge-mode {
        background: rgba(102, 187, 106, 0.15);
        border: 1px solid #66BB6A;
        color: #66BB6A;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 4px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .sim-kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .sim-kpi-card {
        background: rgba(22, 34, 51, 0.75);
        border: 1px solid #2a3f54;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .sim-kpi-label {
        font-size: 0.74rem;
        color: #90A4AE;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .sim-kpi-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #ECEFF1;
        line-height: 1.3;
    }
    .sim-kpi-sub {
        font-size: 0.75rem;
        color: #78909C;
        margin-top: 0.2rem;
    }
    .sim-panel-box {
        background: rgba(20, 31, 44, 0.7);
        border: 1px solid #26384a;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1rem;
    }
    .sim-panel-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #90CAF9;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sim-alert-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.75rem;
        text-transform: uppercase;
    }
    .sim-diag-evidence-item {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #42A5F5;
        border-radius: 0 4px 4px 0;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.4rem;
        font-size: 0.82rem;
        color: #CFD8DC;
    }
    .sim-intervention-card {
        background: linear-gradient(135deg, #182838 0%, #1f3448 100%);
        border: 1px solid #374f66;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.8rem;
    }
    </style>
    """


def render_simulator_interface(is_standalone: bool = False, raw_df: pd.DataFrame | None = None, metrics: dict | None = None):
    """
    Render the entire interactive simulator experience.
    
    Parameters
    ----------
    is_standalone : bool
        True if running in live_demo.py; False if embedded in app.py.
    raw_df : pd.DataFrame, optional
        Pre-loaded processed historical dataset.
    metrics : dict, optional
        Pre-loaded model metrics.
    """
    init_simulator_state()
    st.markdown(get_ui_styles(), unsafe_allow_html=True)

    engine: SimulationEngine = st.session_state.sim_engine
    diag_engine: DiagnosticEngine = st.session_state.diag_engine
    rec_engine: RecommendationEngine = st.session_state.rec_engine
    rec_cfg: RecommendationConfig = st.session_state.rec_config

    # Top Header & Status Badges
    st.markdown(f"""
    <div class="sim-header-box">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.8rem;">
            <div>
                <div style="display: flex; align-items: center; gap: 0.6rem;">
                    <span style="font-size: 1.6rem;">☀️</span>
                    <span style="font-size: 1.35rem; font-weight: 700; color: #F5A623; letter-spacing: 0.5px;">
                        Solar IMS — Interactive Soiling Intelligence Simulator
                    </span>
                </div>
                <div style="color: #90A4AE; font-size: 0.86rem; margin-top: 0.3rem;">
                    Manipulate operating conditions, observe performance drop, investigate root causes, evaluate economics, and test interventions in real time.
                </div>
            </div>
            <div class="sim-badge-container">
                <span class="sim-badge-synthetic">🔬 INTERACTIVE SIMULATION</span>
                <span class="{'sim-badge-synthetic' if st.session_state.sim_data_source_mode == 'synthetic' else 'sim-badge-historical'}">
                    {'📡 SOURCE: SYNTHETIC SIMULATION MODEL' if st.session_state.sim_data_source_mode == 'synthetic' else '📊 SOURCE: HISTORICAL PVDAQ SENSORS'}
                </span>
                <span class="sim-badge-mode">
                    STATUS: {'RUNNING (ACTIVE TICK)' if st.session_state.sim_is_running else 'PAUSED (MANUAL STEP)'}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Simulator Navigation Tabs
    tab_sim, tab_compare, tab_log, tab_replay = st.tabs([
        "🕹️ Live Simulator & Diagnostics",
        "🧪 Scenario Comparison Lab",
        "📜 Simulation Event Log & Export",
        "📼 Historical Replay vs Synthetic Mode"
    ])

    with tab_sim:
        render_live_simulator_tab(engine, diag_engine, rec_engine, rec_cfg)

    with tab_compare:
        render_comparison_lab_tab(engine, diag_engine, rec_cfg)

    with tab_log:
        render_event_log_tab(engine)

    with tab_replay:
        render_historical_replay_tab(raw_df, metrics)


@st.fragment(run_every=1.5)
def _render_live_simulation_running(engine: SimulationEngine, diag_engine: DiagnosticEngine,
                                    rec_engine: RecommendationEngine, rec_cfg: RecommendationConfig):
    """Fragment that runs when simulation is active, auto-refreshing every 1.5 seconds."""
    _render_live_simulation_content(engine, diag_engine, rec_engine, rec_cfg, is_ticking=True)


@st.fragment(run_every=None)
def _render_live_simulation_paused(engine: SimulationEngine, diag_engine: DiagnosticEngine,
                                   rec_engine: RecommendationEngine, rec_cfg: RecommendationConfig):
    """Fragment that runs when simulation is paused, stationary with no background timer."""
    _render_live_simulation_content(engine, diag_engine, rec_engine, rec_cfg, is_ticking=False)


def render_live_simulator_tab(engine: SimulationEngine, diag_engine: DiagnosticEngine,
                              rec_engine: RecommendationEngine, rec_cfg: RecommendationConfig):
    """Render Tab 1: delegates to running or paused fragment based on active state."""
    if st.session_state.get("sim_is_running", False):
        _render_live_simulation_running(engine, diag_engine, rec_engine, rec_cfg)
    else:
        _render_live_simulation_paused(engine, diag_engine, rec_engine, rec_cfg)


def _render_live_simulation_content(engine: SimulationEngine, diag_engine: DiagnosticEngine,
                                    rec_engine: RecommendationEngine, rec_cfg: RecommendationConfig,
                                    is_ticking: bool = False):
    """Authoritative live simulation content rendered inside the active/paused fragment."""
    import time
    now = time.time()
    last_tick = st.session_state.get("sim_last_tick_time", 0.0)

    # 1. Advance simulation step if actively ticking and at least ~0.8s elapsed since last tick
    if is_ticking and st.session_state.get("sim_is_running", False):
        if (now - last_tick) >= 0.8:
            st.session_state.sim_last_tick_time = now
            speed = st.session_state.get("sim_speed", 1)
            steps_to_advance = int(speed)
            for _ in range(steps_to_advance):
                engine.step()
            total_steps = max(len(engine.state.readings), 1)
            st.session_state.sim_ui_scrubber = min(engine.state.current_step, total_steps)

    # 2. Get authoritative reading at current step
    reading = engine.get_current_reading()
    if reading is None:
        engine.step()
        reading = engine.get_current_reading()

    cur_step = engine.state.current_step
    cur_ts = reading.timestamp
    time_str = cur_ts.strftime("%H:%M")
    date_str = cur_ts.strftime("%Y-%m-%d")
    is_running = st.session_state.get("sim_is_running", False)
    sim_speed = st.session_state.get("sim_speed", 1)

    # 3. Live Ticker Status Bar (Inside fragment so it pulses & updates time on every tick)
    status_color = "#66BB6A" if is_running else "#FFA726"
    status_text = f"🟢 SIMULATION CLOCK TICKING (+5m / tick at {sim_speed}x speed)" if is_running else "⏸️ SIMULATION CLOCK PAUSED (Manual Stepping)"
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; background:rgba(0,0,0,0.28); border:1px solid #2a3f54; border-radius:8px; padding:0.45rem 0.9rem; margin-bottom:0.9rem; flex-wrap:wrap; gap:0.5rem;">
        <span style="font-size:0.82rem; font-weight:700; color:{status_color};">
            {status_text}
        </span>
        <span style="font-size:0.82rem; color:#ECEFF1; font-weight:600;">
            Simulated Time: <b style="color:#F5A623; font-size:0.92rem;">{time_str}</b> ({date_str}) &nbsp;|&nbsp; Interval #{cur_step} &nbsp;|&nbsp; {len(engine.state.readings)} Readings Buffered
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 4. MODULE A: Live Simulation Control Centre
    st.markdown("### 🎛️ Live Simulation Control Centre")
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([2.5, 1, 1, 1, 1, 1.2])

    with ctrl_col1:
        scenario_names = list(SCENARIO_PRESETS.keys())
        current_scen = engine.state.scenario_name
        idx = scenario_names.index(current_scen) if current_scen in scenario_names else 0
        selected_scenario = st.selectbox(
            "Selected Scenario Preset",
            scenario_names,
            index=idx,
            key="sim_ui_scen_select"
        )
        if selected_scenario != current_scen:
            engine.load_scenario(selected_scenario)
            st.session_state.sim_intervention_snapshot = None
            for _ in range(12):
                engine.step()
            sync_session_state_to_scenario(engine)
            st.session_state.sim_last_tick_time = time.time()
            st.rerun()

    with ctrl_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if is_running:
            if st.button("⏸️ Pause", width="stretch", key="btn_sim_pause"):
                st.session_state.sim_is_running = False
                engine._log_event("simulation_paused", "INFO", "Simulation paused by operator")
                st.rerun()
        else:
            if st.button("▶️ Start", width="stretch", key="btn_sim_start"):
                st.session_state.sim_is_running = True
                st.session_state.sim_last_tick_time = 0.0
                engine._log_event("simulation_started", "INFO", f"Simulation started ({sim_speed}x speed)")
                st.rerun()

    with ctrl_col3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⏭️ +5m", width="stretch", key="btn_sim_step_fwd", help="Advance simulation by 1 step (5 minutes)"):
            engine.step()
            total_steps = max(len(engine.state.readings), 1)
            st.session_state.sim_ui_scrubber = min(engine.state.current_step, total_steps)
            st.session_state.sim_last_tick_time = time.time()
            st.rerun()

    with ctrl_col4:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("⏮️ -5m", width="stretch", key="btn_sim_step_back", help="Rewind simulation by 1 step (5 minutes)"):
            if engine.state.current_step > 0:
                engine.step_to(engine.state.current_step - 1)
                total_steps = max(len(engine.state.readings), 1)
                st.session_state.sim_ui_scrubber = min(engine.state.current_step, total_steps)
                st.session_state.sim_last_tick_time = time.time()
                st.rerun()

    with ctrl_col5:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Reset", width="stretch", key="btn_sim_reset", help="Reset simulation to initial baseline"):
            engine.reset()
            for _ in range(12):
                engine.step()
            st.session_state.sim_intervention_snapshot = None
            st.session_state.sim_is_running = False
            sync_session_state_to_scenario(engine)
            st.session_state.sim_last_tick_time = time.time()
            st.rerun()

    with ctrl_col6:
        speed_opts = [1, 5, 10]
        sel_speed = st.selectbox(
            "Clock Speed",
            speed_opts,
            index=speed_opts.index(sim_speed) if sim_speed in speed_opts else 0,
            format_func=lambda s: f"{s}x Speed",
            key="sim_ui_speed_select"
        )
        if sel_speed != sim_speed:
            st.session_state.sim_speed = sel_speed
            st.rerun()

    # Timeline scrubber & Clock display
    total_steps = max(len(engine.state.readings), 1)
    scrub_col1, scrub_col2 = st.columns([4, 1])
    
    with scrub_col1:
        def on_scrub_change():
            val = st.session_state.get("sim_ui_scrubber", 0)
            if val != engine.state.current_step:
                engine.step_to(val)

        st.session_state.sim_ui_scrubber = min(cur_step, total_steps)

        new_step = st.slider(
            "⏱️ Timeline Scrubber (Navigate across simulated 5-minute telemetry intervals)",
            min_value=0,
            max_value=total_steps,
            format="Step #%d",
            key="sim_ui_scrubber",
            on_change=on_scrub_change
        )

    with scrub_col2:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid #2a3f54; border-radius: 8px; padding: 0.5rem; text-align: center; margin-top: 15px;">
            <div style="font-size: 0.72rem; color: #90A4AE; text-transform: uppercase;">Simulated Clock</div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #F5A623;">{time_str}</div>
            <div style="font-size: 0.7rem; color: #78909C;">{date_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # Execute Diagnostic & Recommendation on the authoritative current reading
    diag_result: DiagnosticResult = diag_engine.diagnose(reading)
    rec_result: CleaningRecommendation = rec_engine.recommend(
        reading=reading,
        diagnostic_severity=diag_result.severity,
        primary_cause=diag_result.primary_cause,
        config=rec_cfg
    )

    # 5. MODULE J: Alerts & KPIs
    render_kpi_cards(reading, diag_result, rec_result)

    # 6. MODULE C: Dynamic Charts
    st.markdown("### 📈 Dynamic Soiling & Generation Curves")
    render_generation_charts(engine, reading)

    # 7. Side-by-Side: Environmental Controls & Fault/Diagnostics
    col_env, col_diag = st.columns([1, 1.1])
    with col_env:
        render_environmental_controls(engine)
    with col_diag:
        render_fault_and_diagnostics_panel(engine, reading, diag_result)

    # 8. MODULE E & F: Recommendation & Simulated Interventions
    st.markdown("### 💡 Cleaning Decision Support & Interventions")
    render_recommendation_and_intervention_panel(engine, reading, diag_result, rec_result, rec_cfg)


def render_kpi_cards(reading: SimulationReading, diag: DiagnosticResult, rec: CleaningRecommendation):
    """Render executive KPI cards (Module J)."""
    p_clean = reading.clean_power
    p_meas = reading.measured_power if not np.isnan(reading.measured_power) else 0.0
    p_gap = max(0, p_clean - p_meas)
    pr = reading.performance_ratio if not np.isnan(reading.performance_ratio) else 0.0

    # Severity and colors
    pr_color = "#4CAF50" if pr >= 0.95 else ("#FFC107" if pr >= 0.85 else ("#FF9800" if pr >= 0.70 else "#EF5350"))
    soiling_lbl = get_soiling_label(reading.soiling_severity)

    st.markdown(f"""
    <div class="sim-kpi-grid">
        <div class="sim-kpi-card" style="border-top: 3px solid #66BB6A;">
            <div class="sim-kpi-label">Clean-Panel Reference</div>
            <div class="sim-kpi-value" style="color: #66BB6A;">{p_clean:,.0f} <span style="font-size:0.8rem;">kW</span></div>
            <div class="sim-kpi-sub">Theoretical Potential (STC Model)</div>
        </div>
        <div class="sim-kpi-card" style="border-top: 3px solid #42A5F5;">
            <div class="sim-kpi-label">Simulated Measured Power</div>
            <div class="sim-kpi-value" style="color: #42A5F5;">{p_meas:,.0f} <span style="font-size:0.8rem;">kW</span></div>
            <div class="sim-kpi-sub">Synthetic Sensor Telemetry</div>
        </div>
        <div class="sim-kpi-card" style="border-top: 3px solid #EF5350;">
            <div class="sim-kpi-label">Uncaptured Power Gap</div>
            <div class="sim-kpi-value" style="color: #EF5350;">-{p_gap:,.0f} <span style="font-size:0.8rem;">kW</span></div>
            <div class="sim-kpi-sub">Total Loss: {reading.total_loss_pct:.1f}%</div>
        </div>
        <div class="sim-kpi-card" style="border-top: 3px solid {pr_color};">
            <div class="sim-kpi-label">Simulated Plant PR</div>
            <div class="sim-kpi-value" style="color: {pr_color};">{pr*100:.1f}%</div>
            <div class="sim-kpi-sub">Soiling Loss: {reading.soiling_loss_pct:.1f}%</div>
        </div>
        <div class="sim-kpi-card" style="border-top: 3px solid {diag.severity_color};">
            <div class="sim-kpi-label">Diagnostic Status</div>
            <div class="sim-kpi-value" style="color: {diag.severity_color}; font-size: 1.05rem;">{diag.severity_label}</div>
            <div class="sim-kpi-sub" style="color: {diag.severity_color}; font-weight: 600;">{diag.primary_cause[:18]}...</div>
        </div>
        <div class="sim-kpi-card" style="border-top: 3px solid {rec.action_color};">
            <div class="sim-kpi-label">Recommendation</div>
            <div class="sim-kpi-value" style="color: {rec.action_color}; font-size: 1.0rem;">{rec.action[:16]}</div>
            <div class="sim-kpi-sub" style="color: {rec.action_color}; font-weight: 600;">Priority: {rec.priority}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_generation_charts(engine: SimulationEngine, current_reading: SimulationReading):
    """Render dynamic generation curve and loss breakdown Plotly charts (Module C)."""
    df_readings = engine.get_readings_dataframe()
    if len(df_readings) == 0:
        st.info("No telemetry readings available in simulation buffer.")
        return

    c1, c2 = st.columns([1.6, 1])

    with c1:
        fig_power = go.Figure()

        # Clean reference power line
        fig_power.add_trace(go.Scatter(
            x=df_readings["timestamp"],
            y=df_readings["clean_power"],
            mode="lines",
            name="Clean-Reference Potential (P_clean)",
            line=dict(color="#66BB6A", width=2, dash="dash"),
            hovertemplate="Clean Potential: %{y:.0f} kW<extra></extra>"
        ))

        # Measured simulated power line
        fig_power.add_trace(go.Scatter(
            x=df_readings["timestamp"],
            y=df_readings["measured_power"],
            mode="lines+markers",
            name="Simulated Output (P_soiled)",
            line=dict(color="#42A5F5", width=2.5),
            marker=dict(size=4),
            fill="tonexty",
            fillcolor="rgba(239, 83, 80, 0.18)",
            hovertemplate="Simulated Output: %{y:.0f} kW<extra></extra>"
        ))

        # Current scrubber position marker
        fig_power.add_trace(go.Scatter(
            x=[current_reading.timestamp],
            y=[current_reading.measured_power if not np.isnan(current_reading.measured_power) else 0],
            mode="markers",
            name="Current Step Ingestion",
            marker=dict(color="#F5A623", size=12, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
            hovertemplate="Current Step: %{y:.0f} kW<extra></extra>"
        ))

        fig_power.update_layout(
            title="Telemetry Stream: Clean Baseline vs. Simulated Output with Performance Gap",
            xaxis_title="Simulated Time (HH:MM)",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,28,39,0.7)",
            height=340,
            margin=dict(t=40, b=30, l=45, r=20),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_power, use_container_width=True)

    with c2:
        # Loss decomposition chart
        fig_loss = go.Figure()

        fig_loss.add_trace(go.Scatter(
            x=df_readings["timestamp"],
            y=df_readings["soiling_loss_kw"],
            mode="lines",
            name="Soiling Loss (kW)",
            line=dict(color="#FF9800", width=2),
            stackgroup="losses",
            fillcolor="rgba(255, 152, 0, 0.4)"
        ))

        fig_loss.add_trace(go.Scatter(
            x=df_readings["timestamp"],
            y=df_readings["temperature_loss_kw"],
            mode="lines",
            name="Thermal Derate (kW)",
            line=dict(color="#FF7043", width=2),
            stackgroup="losses",
            fillcolor="rgba(255, 112, 67, 0.3)"
        ))

        fig_loss.add_trace(go.Scatter(
            x=df_readings["timestamp"],
            y=df_readings["fault_loss_kw"],
            mode="lines",
            name="Fault/Shading Loss (kW)",
            line=dict(color="#AB47BC", width=2),
            stackgroup="losses",
            fillcolor="rgba(171, 71, 188, 0.3)"
        ))

        fig_loss.update_layout(
            title="Decomposed Power Loss Contributors",
            xaxis_title="Simulated Time",
            yaxis_title="Loss (kW)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,28,39,0.7)",
            height=340,
            margin=dict(t=40, b=30, l=45, r=20),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig_loss, use_container_width=True)


def render_environmental_controls(engine: SimulationEngine):
    """Render Module B: Interactive Environmental Controls."""
    st.markdown("""
    <div class="sim-panel-box">
        <div class="sim-panel-title">
            <span>🌦️</span> Synthetic Operating Conditions
        </div>
    """, unsafe_allow_html=True)

    def on_env_change():
        engine.set_environmental_conditions(
            irradiance=st.session_state.get("slider_irr"),
            ambient_temp=st.session_state.get("slider_amb"),
            weather=st.session_state.get("select_weather"),
            soiling_severity=st.session_state.get("slider_soiling"),
            accumulation_rate=st.session_state.get("slider_acc", 0.0) / 100.0,
            shading=st.session_state.get("slider_shading", 0.0)
        )

    b1, b2 = st.columns(2)
    with b1:
        st.slider(
            "Irradiance (W/m²)",
            min_value=0.0, max_value=1200.0,
            step=25.0,
            key="slider_irr",
            on_change=on_env_change,
            help="Simulated direct & diffuse solar radiation arriving at the array"
        )
        st.slider(
            "Ambient Temp (°C)",
            min_value=-10.0, max_value=50.0,
            step=1.0,
            key="slider_amb",
            on_change=on_env_change,
            help="Air temperature recorded by the local weather station"
        )
        weathers = list(WEATHER_CONDITIONS.keys())
        st.selectbox(
            "Weather Sky Condition",
            weathers,
            key="select_weather",
            on_change=on_env_change,
            help="Sky condition directly affects available irradiance and cloud variability"
        )

    with b2:
        st.slider(
            "Soiling Severity (L_soiling)",
            min_value=0.0, max_value=0.40,
            step=0.01,
            format="%.0f%%",
            key="slider_soiling",
            on_change=on_env_change,
            help="Fraction of sunlight absorbed or scattered by dirt before reaching silicon cells"
        )
        st.slider(
            "Accumulation Rate (% / step)",
            min_value=-0.5, max_value=1.0,
            step=0.05,
            format="%.2f%%",
            key="slider_acc",
            on_change=on_env_change,
            help="Rate at which dust accumulates or gets cleaned per 5-minute interval"
        )
        st.slider(
            "Partial Shading Severity",
            min_value=0.0, max_value=0.80,
            step=0.05,
            format="%.0f%%",
            key="slider_shading",
            on_change=on_env_change,
            help="Shadow obstruction on strings or modules"
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_fault_and_diagnostics_panel(engine: SimulationEngine, reading: SimulationReading, diag: DiagnosticResult):
    """Render Module D: Fault Injection & Transparent Diagnostic Engine."""
    st.markdown("""
    <div class="sim-panel-box">
        <div class="sim-panel-title">
            <span>🩺</span> Fault Injection & Diagnostics
        </div>
    """, unsafe_allow_html=True)

    # Fault Multiselect
    all_fault_keys = list(FAULT_TYPES.keys())
    # Exclude "none" from choices to make selection cleaner
    selectable_faults = [k for k in all_fault_keys if k != "none"]
    fault_labels = {k: FAULT_TYPES[k]["label"] for k in selectable_faults}

    def on_fault_change():
        selected_labels = st.session_state.get("fault_injector_multiselect", [])
        new_fault_keys = [k for k, v in fault_labels.items() if v in selected_labels]
        engine.set_faults(new_fault_keys)

    st.multiselect(
        "Inject Operational Faults (Combine Multiple)",
        options=list(fault_labels.values()),
        key="fault_injector_multiselect",
        on_change=on_fault_change,
        help="Introduce known faults to observe how the diagnostic engine detects and classifies them"
    )

    # Diagnostic Assessment Box
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.02); border-left: 4px solid {diag.severity_color}; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.6rem;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <b style="color: #ECEFF1; font-size: 0.95rem;">Primary Diagnosis: {diag.primary_cause}</b>
            <span class="sim-alert-pill" style="background:{diag.severity_color}22; color:{diag.severity_color}; border: 1px solid {diag.severity_color};">
                {diag.severity_label}
            </span>
        </div>
        <div style="color: #B0BEC5; font-size: 0.84rem; margin-top: 0.4rem; line-height: 1.5;">
            {diag.primary_description}
        </div>
        <div style="margin-top: 0.6rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#90A4AE; margin-bottom: 2px;">
                <span>Rule-Based Suspicion Score</span>
                <b>{diag.primary_score:.0f} / 100</b>
            </div>
            <div style="background:#1e2d3d; border-radius:4px; height:6px; overflow:hidden;">
                <div style="background:{diag.severity_color}; width:{diag.primary_score}%; height:100%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Diagnostic Evidence List
    with st.expander("🔎 View Supporting Evidence & Hypotheses Ranking", expanded=False):
        st.markdown("<b style='color:#90CAF9; font-size:0.82rem;'>Evidence Supporting Assessment:</b>", unsafe_allow_html=True)
        for ev in diag.evidence_summary:
            st.markdown(f"<div class='sim-diag-evidence-item'>• {ev}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        st.markdown("<b style='color:#90CAF9; font-size:0.82rem;'>Alternative Hypotheses Comparison:</b>", unsafe_allow_html=True)
        
        hyp_rows = []
        for h in diag.hypotheses:
            hyp_rows.append({
                "Hypothesis": h.cause,
                "Suspicion Score": f"{h.suspicion_score:.0f}%",
                "Primary": "✓ Primary" if h.is_primary else "—"
            })
        st.dataframe(pd.DataFrame(hyp_rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        st.caption("⚠️ Limitation Notice: Suspicion scores are rule-based simulation heuristics, not calibrated ML probabilities. Physical inspection required before capital expenditure.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_recommendation_and_intervention_panel(engine: SimulationEngine, reading: SimulationReading,
                                                diag: DiagnosticResult, rec: CleaningRecommendation,
                                                rec_cfg: RecommendationConfig):
    """Render Module E (Recommendation Engine) & Module F (Simulated Interventions)."""
    
    col_rec, col_interv = st.columns([1.1, 1])

    with col_rec:
        st.markdown("""
        <div class="sim-panel-box">
            <div class="sim-panel-title">
                <span>💰</span> Cleaning Recommendation & Economics
            </div>
        """, unsafe_allow_html=True)

        # Recommendation Decision Card
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #152433 0%, #1c3042 100%); border-left: 5px solid {rec.action_color}; border-radius: 8px; padding: 1.1rem; margin-bottom: 0.8rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.75rem; text-transform:uppercase; color:#90A4AE; font-weight:700;">Automated Recommendation</span>
                <span style="background:{rec.action_color}22; color:{rec.action_color}; padding:2px 8px; border-radius:4px; font-weight:700; font-size:0.75rem;">
                    Priority: {rec.priority}
                </span>
            </div>
            <div style="font-size:1.3rem; font-weight:700; color:#ECEFF1; margin: 0.4rem 0;">
                {rec.action}
            </div>
            <div style="font-size:0.86rem; color:#CFD8DC; line-height:1.5;">
                {rec.explanation}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Economic Breakdown Cards
        econ = rec.economics
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Recoverable Energy", f"{econ.recoverable_energy_kwh:,.0f} kWh", help="Projected energy recovery over chosen horizon")
        with e2:
            st.metric("Energy Value", f"{econ.currency_symbol}{econ.energy_value:,.0f}", help=f"Value at {econ.currency_symbol}{rec_cfg.energy_rate}/kWh")
        with e3:
            net_color = "normal" if econ.net_benefit >= 0 else "inverse"
            st.metric("Net Benefit", f"{econ.currency_symbol}{econ.net_benefit:,.0f}", delta=f"{econ.benefit_ratio:.2f}x ROI", delta_color=net_color)

        # Economic Assumptions expander
        with st.expander("⚙️ Adjust Financial & Maintenance Assumptions"):
            c_cost, c_rate = st.columns(2)
            with c_cost:
                new_cost = st.number_input("Cleaning Cost (INR ₹)", min_value=1000.0, max_value=200000.0, value=float(rec_cfg.cleaning_cost), step=2500.0)
                new_eff = st.slider("Cleaning Effectiveness (%)", min_value=0.50, max_value=1.0, value=float(rec_cfg.cleaning_effectiveness), step=0.05, format="%.0f%%")
            with c_rate:
                new_rate = st.number_input("Electricity Tariff (INR ₹ / kWh)", min_value=1.0, max_value=25.0, value=float(rec_cfg.energy_rate), step=0.5)
                new_horizon = st.selectbox("Evaluation Horizon", [168, 360, 720, 1440], index=2, format_func=lambda h: f"{h//24} Days ({h} hrs)")

            if (new_cost != rec_cfg.cleaning_cost or new_rate != rec_cfg.energy_rate or
                new_eff != rec_cfg.cleaning_effectiveness or new_horizon != rec_cfg.evaluation_horizon_hours):
                rec_cfg.cleaning_cost = new_cost
                rec_cfg.energy_rate = new_rate
                rec_cfg.cleaning_effectiveness = new_eff
                rec_cfg.evaluation_horizon_hours = new_horizon
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_interv:
        st.markdown("""
        <div class="sim-panel-box">
            <div class="sim-panel-title">
                <span>🛠️</span> Simulated Interventions
            </div>
        """, unsafe_allow_html=True)

        st.caption("Execute simulated physical operations on the array and evaluate immediate state shifts.")

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            if st.button("🧹 Simulate Cleaning", width="stretch", key="btn_run_cleaning", type="primary"):
                interv = engine.simulate_cleaning(effectiveness=rec_cfg.cleaning_effectiveness)
                st.session_state.sim_intervention_snapshot = interv
                st.rerun()

            if st.button("⏳ Delay Cleaning", width="stretch", key="btn_delay_cleaning"):
                engine.delay_cleaning()
                st.info("Decision recorded: Cleaning delayed. Natural or artificial soiling accumulation will continue.")

        with btn_c2:
            if st.button("🔍 Investigate Fault", width="stretch", key="btn_investigate"):
                engine._log_event("investigation_initiated", "ACTION", "Dispatched maintenance crew to inspect array string combiner and sensors.")
                st.info("Investigation initiated. Sensor and inverter inspection tickets logged.")

            if st.button("🔄 Reset Intervention", width="stretch", key="btn_reset_interv"):
                st.session_state.sim_intervention_snapshot = None
                st.rerun()

        # Before and After Comparison Card
        if st.session_state.sim_intervention_snapshot:
            snap = st.session_state.sim_intervention_snapshot
            st.markdown(f"""
            <div class="sim-intervention-card">
                <div style="color: #66BB6A; font-weight:700; font-size:0.88rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:0.4rem;">
                    <span>✓</span> Cleaning Intervention Completed ({snap.cleaning_effectiveness:.0%} Effectiveness)
                </div>
                <table style="width:100%; font-size:0.83rem; color:#CFD8DC; border-collapse:collapse;">
                    <tr style="border-bottom:1px solid #2a3f54;">
                        <th style="text-align:left; padding:4px 0; color:#90A4AE;">Metric</th>
                        <th style="text-align:right; padding:4px 0; color:#EF5350;">Pre-Clean</th>
                        <th style="text-align:right; padding:4px 0; color:#66BB6A;">Post-Clean</th>
                        <th style="text-align:right; padding:4px 0; color:#42A5F5;">Delta</th>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:4px 0;">Soiling Severity</td>
                        <td style="text-align:right;">{snap.pre_soiling:.1%}</td>
                        <td style="text-align:right; font-weight:700;">{snap.post_soiling:.1%}</td>
                        <td style="text-align:right; color:#66BB6A;">-{(snap.pre_soiling - snap.post_soiling):.1%}</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:4px 0;">Delivered Output</td>
                        <td style="text-align:right;">{snap.pre_power:,.0f} kW</td>
                        <td style="text-align:right; font-weight:700;">{snap.post_power:,.0f} kW</td>
                        <td style="text-align:right; color:#66BB6A;">+{snap.estimated_recovery_kw:,.0f} kW</td>
                    </tr>
                    <tr>
                        <td style="padding:4px 0;">Clean Potential</td>
                        <td style="text-align:right;">{snap.pre_clean_power:,.0f} kW</td>
                        <td style="text-align:right;">{snap.pre_clean_power:,.0f} kW</td>
                        <td style="text-align:right; color:#78909C;">0 kW (Unchanged)</td>
                    </tr>
                </table>
                <div style="margin-top:0.6rem; font-size:0.75rem; color:#90A4AE; line-height:1.4;">
                    Physical Verification: Cleaning restores panel transmittance and improves actual output toward clean baseline without altering clean-panel potential.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_comparison_lab_tab(engine: SimulationEngine, diag_engine: DiagnosticEngine, rec_cfg: RecommendationConfig):
    """Render Tab 2: Scenario Comparison Laboratory (Module H)."""
    st.markdown("### 🧪 Scenario Comparison Laboratory")
    st.caption("Compare up to 3 distinct operational environments side by side to observe differential power degradation, diagnostic classifications, and net economic benefits.")

    # Preset comparison choices
    all_scens = list(SCENARIO_PRESETS.keys())
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        s1 = st.selectbox("Scenario A", all_scens, index=0, key="lab_scen_1")
    with col_s2:
        s2 = st.selectbox("Scenario B", all_scens, index=1 if len(all_scens) > 1 else 0, key="lab_scen_2")
    with col_s3:
        s3 = st.selectbox("Scenario C", all_scens, index=3 if len(all_scens) > 3 else 0, key="lab_scen_3")

    scen_choices = [s1, s2, s3]
    comp_data = []

    # Run deterministic simulation for each scenario at midday (step 72)
    for sc_name in scen_choices:
        temp_engine = SimulationEngine()
        temp_engine.load_scenario(sc_name)
        temp_engine.state.current_step = 72
        temp_engine.state.current_time = datetime(2025, 6, 15, 12, 0)
        rd = temp_engine._generate_reading_at_step(72)
        diag = diag_engine.diagnose(rd)
        
        # Calculate economics
        temp_rec_engine = RecommendationEngine(rec_cfg)
        rec = temp_rec_engine.recommend(rd, diag.severity, diag.primary_cause, rec_cfg)

        comp_data.append({
            "Scenario": sc_name,
            "Weather": rd.weather,
            "Irradiance (W/m²)": rd.irradiance,
            "Ambient Temp (°C)": rd.ambient_temp,
            "Soiling Severity": f"{rd.soiling_severity:.1%}",
            "Clean Potential (kW)": rd.clean_power,
            "Simulated Power (kW)": rd.measured_power if not np.isnan(rd.measured_power) else 0.0,
            "Shortfall (kW)": max(0, rd.clean_power - (rd.measured_power if not np.isnan(rd.measured_power) else 0.0)),
            "Performance Ratio": f"{rd.performance_ratio*100:.1f}%" if not np.isnan(rd.performance_ratio) else "N/A",
            "Primary Diagnosis": diag.primary_cause,
            "Recommendation": rec.action,
            "Net Benefit (INR)": f"{rec_cfg.currency_symbol}{rec.economics.net_benefit:,.0f}",
            "raw_clean": rd.clean_power,
            "raw_meas": rd.measured_power if not np.isnan(rd.measured_power) else 0.0,
            "raw_net": rec.economics.net_benefit
        })

    # Render Side-by-Side Comparison Table
    df_comp = pd.DataFrame(comp_data)
    display_cols = [
        "Scenario", "Weather", "Irradiance (W/m²)", "Soiling Severity",
        "Clean Potential (kW)", "Simulated Power (kW)", "Shortfall (kW)",
        "Performance Ratio", "Primary Diagnosis", "Recommendation", "Net Benefit (INR)"
    ]
    st.dataframe(df_comp[display_cols], use_container_width=True, hide_index=True)

    # Comparison Visualizations
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Clean Potential (kW)",
            x=[d["Scenario"] for d in comp_data],
            y=[d["raw_clean"] for d in comp_data],
            marker_color="#66BB6A"
        ))
        fig_bar.add_trace(go.Bar(
            name="Simulated Power (kW)",
            x=[d["Scenario"] for d in comp_data],
            y=[d["raw_meas"] for d in comp_data],
            marker_color="#42A5F5"
        ))
        fig_bar.update_layout(
            title="Clean vs. Simulated Delivered Output",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,28,39,0.7)",
            height=320,
            margin=dict(t=40, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_c2:
        net_benefits = [d["raw_net"] for d in comp_data]
        bar_colors = ["#4CAF50" if b >= 0 else "#EF5350" for b in net_benefits]
        fig_econ = go.Figure(go.Bar(
            x=[d["Scenario"] for d in comp_data],
            y=net_benefits,
            marker_color=bar_colors,
            text=[f"{rec_cfg.currency_symbol}{b:,.0f}" for b in net_benefits],
            textposition="auto"
        ))
        fig_econ.update_layout(
            title=f"Net Cleaning Benefit ({rec_cfg.currency_symbol}) over Horizon",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,28,39,0.7)",
            height=320,
            margin=dict(t=40, b=30, l=40, r=20)
        )
        st.plotly_chart(fig_econ, use_container_width=True)


def render_event_log_tab(engine: SimulationEngine):
    """Render Tab 3: Simulation Event Log & Export (Module I)."""
    st.markdown("### 📜 Simulation Event Log & Audit Export")
    st.caption("Complete chronological record of all simulation states, threshold crossings, injected faults, and operator interventions.")

    df_log = engine.get_event_log_dataframe()

    if len(df_log) == 0:
        st.info("Event log is currently empty.")
        return

    # Filter controls
    f_c1, f_c2, f_c3 = st.columns([1.5, 1.5, 1])
    with f_c1:
        types = ["All"] + list(df_log["event_type"].unique())
        sel_type = st.selectbox("Filter by Event Type", types, key="log_f_type")
    with f_c2:
        sevs = ["All"] + list(df_log["severity"].unique())
        sel_sev = st.selectbox("Filter by Severity", sevs, key="log_f_sev")
    with f_c3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Log", key="btn_clear_log", width="stretch"):
            engine.state.event_log = []
            st.rerun()

    # Apply filters
    filtered_log = df_log.copy()
    if sel_type != "All":
        filtered_log = filtered_log[filtered_log["event_type"] == sel_type]
    if sel_sev != "All":
        filtered_log = filtered_log[filtered_log["severity"] == sel_sev]

    # Display table
    st.dataframe(filtered_log, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_buffer = io.StringIO()
    filtered_log.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Event Log as CSV",
        data=csv_bytes,
        file_name=f"solar_ims_event_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False,
        key="btn_download_log_csv"
    )


def render_historical_replay_tab(raw_df: pd.DataFrame | None, metrics: dict | None):
    """Render Tab 4: Historical Replay vs Synthetic Mode (Module G)."""
    st.markdown("### 📼 Historical Replay vs. Synthetic Mode")
    st.caption("Toggle between verified historical sensor records (PVDAQ System 9068) and the interactive synthetic simulation engine.")

    mode = st.radio(
        "Select Active Operation Mode:",
        ["🔬 Synthetic Scenario Simulation (Default)", "📊 Historical PVDAQ Sensor Replay"],
        index=0 if st.session_state.sim_data_source_mode == "synthetic" else 1,
        horizontal=True,
        key="radio_source_mode"
    )

    if "Synthetic" in mode:
        st.session_state.sim_data_source_mode = "synthetic"
        st.success("Active Mode: Synthetic Scenario Simulation — Operating with fully interactive sliders and fault injection.")
    else:
        st.session_state.sim_data_source_mode = "historical"
        st.info("Active Mode: Historical Sensor Replay — Navigating authentic 5-minute telemetry from PVDAQ Plant 9068.")

        if raw_df is None:
            try:
                raw_df = load_processed_data()
            except Exception as e:
                st.error(f"Could not load historical dataset: {e}")
                return

        # Day slice selection
        daytime = raw_df[(raw_df["irradiance"] > 250) & (raw_df["total_power"] > 80)]
        severe_dates = daytime[daytime["recommendation"] == "Clean Immediately"]["date"].value_counts().index
        sev_dt = severe_dates[0] if len(severe_dates) > 0 else daytime["date"].iloc[0]

        st.markdown(f"**Historical Day:** {sev_dt} (Identified Severe Soiling Event)")
        day_slice = raw_df[raw_df["date"] == sev_dt].sort_values("measured_on").reset_index(drop=True)

        # Plot historical day
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=day_slice["measured_on"],
            y=day_slice["expected_power"],
            mode="lines",
            name="XGBoost Clean Benchmark (Expected kW)",
            line=dict(color="#66BB6A", width=2, dash="dash")
        ))
        fig_hist.add_trace(go.Scatter(
            x=day_slice["measured_on"],
            y=day_slice["total_power"],
            mode="lines+markers",
            name="Actual Measured Power (kW)",
            line=dict(color="#42A5F5", width=2)
        ))
        fig_hist.update_layout(
            title=f"PVDAQ 9068 Ground Truth Telemetry on {sev_dt}",
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,28,39,0.7)",
            height=340
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption("Notice: Historical sensor readings reflect real operational data and cannot be altered by simulation sliders.")
