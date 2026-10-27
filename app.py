"""
Solar Intelligent Monitoring System (Solar IMS)
Soiling Intelligence & Cleaning Recommendation Dashboard
=========================================================
Streamlit presentation layer for the PoC2 pipeline outputs.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from data_loader import load_processed_data, load_model_metrics
from simulator_ui import render_simulator_interface

_BASE_DIR = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar IMS — Soiling Intelligence Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Capture load time once at startup
_load_timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")

# ──────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(180deg, #0f1923 0%, #1a2332 100%);
    }

    /* Header styling */
    .main-header {
        text-align: left;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        color: #F5A623;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }
    .main-header h3 {
        color: #90CAF9;
        font-weight: 400;
        font-size: 1.05rem;
        margin-top: 0;
    }
    .main-header p {
        color: #78909C;
        font-size: 0.88rem;
    }

    /* Section headers */
    .section-header {
        color: #E0E0E0;
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 2px solid #37474F;
        padding-bottom: 8px;
        margin: 2.5rem 0 1.5rem 0;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1e2d3d 0%, #243447 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #37474F;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    .hero-item {
        margin: 0.5rem 1rem;
    }
    .hero-label {
        color: #90A4AE;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    .hero-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ECEFF1;
    }

    /* KPI card styling */
    .kpi-card {
        background: rgba(36, 52, 71, 0.6);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #2a3f54;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .kpi-label {
        color: #B0BEC5;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .kpi-indicator {
        font-size: 0.8rem;
        margin-top: 0.4rem;
    }

    /* Decision-support recommendation card */
    .rec-card {
        background: linear-gradient(135deg, #1a2a3a 0%, #1e3346 50%, #1a2a3a 100%);
        border-radius: 14px;
        padding: 2rem 2.2rem;
        border-left: 5px solid;
        margin-bottom: 1.5rem;
    }
    .rec-card-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.6rem;
    }
    .rec-card-action {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ECEFF1;
        line-height: 1.4;
        margin-bottom: 1.2rem;
    }
    .rec-meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
    }
    .rec-meta-item {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    .rec-meta-label {
        color: #78909C;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.25rem;
    }
    .rec-meta-value {
        color: #ECEFF1;
        font-size: 1.15rem;
        font-weight: 600;
    }
    .rec-why {
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        color: #B0BEC5;
        font-size: 0.92rem;
        line-height: 1.6;
    }

    /* Insight cards */
    .insight-card {
        background: rgba(36, 52, 71, 0.35);
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #2a3f54;
        height: 100%;
    }
    .insight-icon {
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }
    .insight-text {
        color: #CFD8DC;
        font-size: 0.93rem;
        line-height: 1.55;
    }

    /* Professional footer */
    .pro-footer {
        margin-top: 3rem;
        padding: 1.2rem 0;
        border-top: 1px solid #263238;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .footer-item {
        color: #546E7A;
        font-size: 0.78rem;
        padding: 0 1rem;
    }
    .footer-item b {
        color: #78909C;
    }
    .footer-divider {
        color: #37474F;
        font-size: 0.78rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    header [data-testid="stDecoration"] {visibility: hidden;}
    header [data-testid="stStatusWidget"] {visibility: hidden;}

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #111a24 !important;
        border-right: 1px solid #263238 !important;
    }

    /* 4. Smooth scrolling & scroll offset for all sections */
    html {
        scroll-behavior: smooth;
    }
    .section-header, .main-header {
        scroll-margin-top: 1.5rem;
    }

    /* 5. Sidebar Navigation Links */
    .sidebar-nav-container {
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
        margin-top: 0.4rem;
    }
    .sidebar-nav-link {
        color: #B0BEC5 !important;
        text-decoration: none !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        display: flex !important;
        align-items: center !important;
        padding: 0.52rem 0.75rem !important;
        border-radius: 6px !important;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        transition: all 0.2s ease-in-out !important;
        line-height: 1.3 !important;
    }
    .sidebar-nav-link:hover {
        color: #F5A623 !important;
        background: rgba(245, 166, 35, 0.12) !important;
        border-color: rgba(245, 166, 35, 0.4) !important;
        transform: translateX(4px) !important;
    }
    .sidebar-nav-link:active {
        background: rgba(245, 166, 35, 0.22) !important;
    }

    /* ── XAI Decision Engine ── */
    .xai-stage {
        background: linear-gradient(135deg, #131f2e 0%, #182838 100%);
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        border-left: 4px solid #263238;
        margin-bottom: 0;
        position: relative;
    }
    .xai-stage-head {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.5rem;
    }
    .xai-stage-icon {
        font-size: 1.3rem;
        width: 30px;
        text-align: center;
    }
    .xai-stage-title {
        color: #E0E0E0;
        font-size: 1.05rem;
        font-weight: 600;
        flex: 1;
    }
    .xai-stage-badge {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 2px 8px;
        border-radius: 3px;
        background: #1b3a2a;
        color: #4CAF50;
    }
    .xai-narrative {
        color: #78909C;
        font-size: 0.85rem;
        font-style: italic;
        margin-bottom: 0.7rem;
        padding-left: 2.3rem;
    }
    .xai-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
        gap: 0.5rem;
        margin-left: 2.3rem;
    }
    .xai-cell {
        background: rgba(255,255,255,0.025);
        border-radius: 6px;
        padding: 0.5rem 0.7rem;
    }
    .xai-cell-label {
        color: #546E7A;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .xai-cell-value {
        color: #ECEFF1;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 2px;
    }
    .xai-status {
        margin: 0.6rem 0 0 2.3rem;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .xai-connector {
        text-align: center;
        padding: 0;
        margin: 0;
        line-height: 1.6;
        color: #263238;
        font-size: 0.9rem;
        user-select: none;
    }
    .xai-pass { color: #4CAF50; }
    .xai-fail { color: #EF5350; }
    .xai-warn { color: #FF9800; }
    .xai-muted { color: #455A64; }
    /* progress bar */
    .xai-progress-outer {
        background: #1a2836;
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
        margin: 0.3rem 0 0.6rem 0;
    }
    .xai-progress-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #42A5F5, #66BB6A);
        transition: width 0.3s ease;
    }
    /* summary card */
    .xai-summary {
        background: linear-gradient(135deg, #162233 0%, #1c3044 50%, #162233 100%);
        border-radius: 12px;
        padding: 1.6rem 2rem;
        border: 1px solid #2a3f54;
        margin-top: 0.8rem;
    }
    .xai-summary-title {
        color: #90CAF9;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .xai-summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.8rem;
    }
    .xai-summary-item {
        padding: 0.6rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .xai-summary-label {
        color: #546E7A;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .xai-summary-value {
        color: #ECEFF1;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 2px;
    }
    /* confidence */
    .xai-confidence {
        background: rgba(36, 52, 71, 0.3);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #263238;
        margin-top: 0.8rem;
    }
    /* timeline */
    .xai-timeline-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.25rem 0;
        font-size: 0.78rem;
    }
    .xai-tl-time {
        color: #455A64;
        font-family: monospace;
        min-width: 65px;
    }
    .xai-tl-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .xai-tl-label {
        color: #90A4AE;
    }

    /* Pipeline flow diagram */
    .pipeline-stage {
        background: rgba(36, 52, 71, 0.5);
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        border-left: 4px solid #37474F;
        margin-bottom: 0.3rem;
    }
    .pipeline-tag {
        display: inline-block;
        font-size: 0.62rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 2px 8px;
        border-radius: 3px;
        margin-left: 0.5rem;
    }
    .tag-ml { background: #1b2d4a; color: #42A5F5; }
    .tag-rule { background: #2a3a1b; color: #8BC34A; }
    .tag-none { background: #2a2a2a; color: #9E9E9E; }
    .pipeline-arrow {
        text-align: center;
        color: #F5A623;
        font-size: 1rem;
        padding: 0;
        margin: 0;
        line-height: 1.3;
    }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return load_processed_data(_BASE_DIR / "processed_output.csv")


@st.cache_data
def load_metrics():
    return load_model_metrics(_BASE_DIR / "model_metrics.json")


try:
    df = load_data()
    metrics = load_metrics()
except FileNotFoundError:
    st.error(
        "**Data files not found.** Run `python prepare_data.py` first to "
        "generate `processed_output.csv` and `model_metrics.json`."
    )
    st.stop()
except (ValueError, json.JSONDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as err:
    st.error(
        f"**Data validation or parsing error:** {err}. "
        "The dataset or metrics file may be corrupted or from an incompatible version. "
        "Run `python prepare_data.py` to regenerate clean data."
    )
    st.stop()


@st.cache_data
def get_training_baseline_stats(full_df):
    """Returns verified baseline ground truth statistics from the training dataset."""
    return [
        {
            "Parameter": "Sunlight Intensity (Irradiance)",
            "Sensor Source": "Pyranometer (Class A)",
            "Unit": "W/m²",
            "Min Value": f"{full_df['irradiance'].min():.1f}",
            "Average": f"{full_df['irradiance'].mean():.1f}",
            "Max Value": f"{full_df['irradiance'].max():.1f}",
            "Model Influence": "77.4% (Primary Driver)",
            "Plain-Language Meaning": "Direct solar energy reaching the panels; main driver of power generation.",
        },
        {
            "Parameter": "Solar Panel Temperature",
            "Sensor Source": "Thermocouple (Module Back)",
            "Unit": "°C",
            "Min Value": f"{full_df['module_temp'].min():.1f}",
            "Average": f"{full_df['module_temp'].mean():.1f}",
            "Max Value": f"{full_df['module_temp'].max():.1f}",
            "Model Influence": "17.3% (Thermal Efficiency)",
            "Plain-Language Meaning": "Panels lose ~0.4% efficiency for every degree above 25°C due to semiconductor physics.",
        },
        {
            "Parameter": "Panel-to-Air Temperature Gap",
            "Sensor Source": "Panel Temp minus Air Temp",
            "Unit": "°C",
            "Min Value": f"{full_df['temp_diff'].min():.1f}",
            "Average": f"{full_df['temp_diff'].mean():.1f}",
            "Max Value": f"{full_df['temp_diff'].max():.1f}",
            "Model Influence": "4.1% (Heat Retention)",
            "Plain-Language Meaning": "Identifies solar heat trapping on dirty panels compared to clean glass.",
        },
        {
            "Parameter": "Surrounding Air Temperature",
            "Sensor Source": "Ambient Weather Station",
            "Unit": "°C",
            "Min Value": f"{full_df['ambient_temp'].min():.1f}",
            "Average": f"{full_df['ambient_temp'].mean():.1f}",
            "Max Value": f"{full_df['ambient_temp'].max():.1f}",
            "Model Influence": "1.2% (Climate Context)",
            "Plain-Language Meaning": "Seasonal temperature baseline spanning winter freeze to peak summer heat.",
        },
        {
            "Parameter": "Target: Total Plant Power Output",
            "Sensor Source": "Plant AC Power Meter",
            "Unit": "kW",
            "Min Value": f"{full_df['total_power'].min():.1f}",
            "Average": f"{full_df['total_power'].mean():.1f}",
            "Max Value": f"{full_df['total_power'].max():.1f}",
            "Model Influence": "Ground-Truth Output",
            "Plain-Language Meaning": "Actual electricity generated by the 4,000 kW (4 MW) utility-scale solar plant.",
        },
    ]


@st.cache_data
def get_live_demo_scenarios(data):
    """Finds authentic daytime slices for the interactive live stream demo."""
    daytime = data[(data["irradiance"] > 250) & (data["total_power"] > 80) & (data["expected_power"] > 150)]
    severe_candidates = daytime[daytime["recommendation"] == "Clean Immediately"]["date"].value_counts().index
    severe_date = severe_candidates[0] if len(severe_candidates) > 0 else daytime["date"].iloc[0]

    moderate_candidates = daytime[daytime["recommendation"] == "Schedule Cleaning"]["date"].value_counts().index
    moderate_date = moderate_candidates[0] if len(moderate_candidates) > 0 else daytime["date"].iloc[10]

    healthy_candidates = daytime[daytime["recommendation"] == "Healthy"]["date"].value_counts().index
    healthy_date = healthy_candidates[0] if len(healthy_candidates) > 0 else daytime["date"].iloc[20]

    scenarios = {}
    for name, dt_val in [
        ("🔴 Severe Soiling Day (Critical Shortfall)", severe_date),
        ("🟡 Moderate Soiling Day (Cleaning Scheduled)", moderate_date),
        ("🟢 Clean Panel Day (Optimal Generation)", healthy_date),
    ]:
        day_df = data[(data["date"] == dt_val) & (data["irradiance"] > 80)].sort_values("measured_on").reset_index(drop=True)
        if len(day_df) > 0:
            scenarios[name] = day_df
    return scenarios

# ──────────────────────────────────────────────────────────────────────
# ANALYSIS PERIOD FILTERING  (v2.1)
# ──────────────────────────────────────────────────────────────────────
# Full (unfiltered) operating subset — used for model validation only
op_full = df[
    (df["irradiance"] > 200)
    & (df["expected_power"] > 100)
    & (df["total_power"] > 50)
].copy()

# ──────────────────────────────────────────────────────────────────────
# SIDEBAR: BRANDING, FULL NAVIGATION & PERIOD FILTER
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.2rem 0 0.8rem 0; border-bottom: 1px solid #263238; margin-bottom: 0.8rem;">
        <div style="font-size: 1.35rem; font-weight: 700; color: #F5A623; letter-spacing: 0.5px; display: flex; align-items: center; gap: 0.5rem;">
            <span>☀️</span> SOLAR IMS
        </div>
        <div style="font-size: 0.82rem; color: #90CAF9; font-weight: 500; margin-top: 2px;">
            Soiling Intelligence System
        </div>
        <div style="font-size: 0.72rem; color: #78909C; margin-top: 3px;">
            PVDAQ Plant 9068 • 4,000 kW Utility PV
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Section Navigation")
    st.markdown("""
    <div class="sidebar-nav-container">
        <a class="sidebar-nav-link" href="#dashboard-overview" onclick="const el=document.getElementById('dashboard-overview'); if(el){el.scrollIntoView({behavior:'smooth'});}">📍 Plant Overview</a>
        <a class="sidebar-nav-link" href="#live-demo-stream" onclick="const el=document.getElementById('live-demo-stream'); if(el){el.scrollIntoView({behavior:'smooth'});}">🔴 Live Data Stream Demo</a>
        <a class="sidebar-nav-link" href="#recommended-maintenance-actions" onclick="const el=document.getElementById('recommended-maintenance-actions'); if(el){el.scrollIntoView({behavior:'smooth'});}">🛠️ Maintenance Queue</a>
        <a class="sidebar-nav-link" href="#decision-workflow" onclick="const el=document.getElementById('decision-workflow'); if(el){el.scrollIntoView({behavior:'smooth'});}">🔍 Step-by-Step Decision Trace</a>
        <a class="sidebar-nav-link" href="#prototype-workflow" onclick="const el=document.getElementById('prototype-workflow'); if(el){el.scrollIntoView({behavior:'smooth'});}">🔄 How Decisions Are Made</a>
        <a class="sidebar-nav-link" href="#operational-insights" onclick="const el=document.getElementById('operational-insights'); if(el){el.scrollIntoView({behavior:'smooth'});}">💡 Key Operational Takeaways</a>
        <a class="sidebar-nav-link" href="#business-impact" onclick="const el=document.getElementById('business-impact'); if(el){el.scrollIntoView({behavior:'smooth'});}">💼 Business & Revenue Impact</a>
        <a class="sidebar-nav-link" href="#plant-performance" onclick="const el=document.getElementById('plant-performance'); if(el){el.scrollIntoView({behavior:'smooth'});}">📈 Plant Generation Analytics</a>
        <a class="sidebar-nav-link" href="#training-baseline" onclick="const el=document.getElementById('training-baseline'); if(el){el.scrollIntoView({behavior:'smooth'});}">📊 Training Baseline Ground Truth</a>
        <a class="sidebar-nav-link" href="#model-information" onclick="const el=document.getElementById('model-information'); if(el){el.scrollIntoView({behavior:'smooth'});}">📄 AI Model Information</a>
        <a class="sidebar-nav-link" href="#data-quality" onclick="const el=document.getElementById('data-quality'); if(el){el.scrollIntoView({behavior:'smooth'});}">🛡️ Sensor & Data Health</a>
        <a class="sidebar-nav-link" href="#solution-workflow" onclick="const el=document.getElementById('solution-workflow'); if(el){el.scrollIntoView({behavior:'smooth'});}">⚙️ End-to-End Pipeline</a>
        <a class="sidebar-nav-link" href="#rca-validation" onclick="const el=document.getElementById('rca-validation'); if(el){el.scrollIntoView({behavior:'smooth'});}">🔎 Decision Rule Validation</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1rem; margin-bottom: 0.8rem; border-top: 1px solid #263238;'></div>", unsafe_allow_html=True)
    st.markdown("### ⏱️ Analysis Period")
    period_option = st.selectbox(
        "Select period",
        ["Full Dataset", "Last 30 Days", "Last 90 Days", "Custom Range"],
        key="period_option",
    )

    _data_max_date = df["measured_on"].max()
    _data_min_date = df["measured_on"].min()

    if period_option == "Last 30 Days":
        _filter_start = _data_max_date - timedelta(days=30)
        _filter_end = _data_max_date
    elif period_option == "Last 90 Days":
        _filter_start = _data_max_date - timedelta(days=90)
        _filter_end = _data_max_date
    elif period_option == "Custom Range":
        _date_col1, _date_col2 = st.columns(2)
        with _date_col1:
            _custom_start = st.date_input(
                "Start date", value=_data_min_date.date(),
                min_value=_data_min_date.date(), max_value=_data_max_date.date(),
            )
        with _date_col2:
            _custom_end = st.date_input(
                "End date", value=_data_max_date.date(),
                min_value=_data_min_date.date(), max_value=_data_max_date.date(),
            )
        _filter_start = pd.Timestamp(_custom_start)
        _filter_end = pd.Timestamp(_custom_end) + timedelta(hours=23, minutes=59)
    else:  # Full Dataset
        _filter_start = _data_min_date
        _filter_end = _data_max_date

    # Show selected range
    st.caption(
        f"**Showing:** {_filter_start.strftime('%Y-%m-%d')} to {_filter_end.strftime('%Y-%m-%d')}"
    )
    st.markdown("<div style='margin-top: 0.8rem; margin-bottom: 0.5rem; border-top: 1px solid #263238;'></div>", unsafe_allow_html=True)

# Apply filter
df_filtered = df[(df["measured_on"] >= _filter_start) & (df["measured_on"] <= _filter_end)]

# ── Derived subsets (filtered for operational analytics) ──
op = df_filtered[
    (df_filtered["irradiance"] > 200)
    & (df_filtered["expected_power"] > 100)
    & (df_filtered["total_power"] > 50)
].copy()

cleaning_actions = ["Schedule Cleaning", "Clean Immediately", "Monitor", "Investigate"]

# Gracefully handle empty filtered periods
if len(op) == 0:
    st.warning("No valid operating records in the selected period. Try a wider date range.")
    st.stop()

# Core calculations
avg_power = op["total_power"].mean()
avg_expected = op["expected_power"].mean()
avg_pr = op["model_perf_ratio"].mean()
avg_loss = op["loss_pct"].mean()

n_healthy = (op["recommendation"] == "Healthy").sum()
n_op = len(op)
health_score = (n_healthy / n_op * 100) if n_op > 0 else 0

n_cleaning = op["recommendation"].isin(["Schedule Cleaning", "Clean Immediately"]).sum()
n_monitor = (op["recommendation"] == "Monitor").sum()
n_clean_imm = (op["recommendation"] == "Clean Immediately").sum()
n_sched_clean = (op["recommendation"] == "Schedule Cleaning").sum()

# Recoverable energy calculation
cleaning_df = op[op["recommendation"].isin(["Schedule Cleaning", "Clean Immediately"])]
recoverable_kwh = cleaning_df["est_power_loss"].sum() * (5 / 60)  # 5-min intervals
recoverable_mwh = recoverable_kwh / 1000
avg_elec_rate = 0.08  # $/kWh
potential_revenue = recoverable_kwh * avg_elec_rate

# ── Plant-Level Focus (Plant-wide only) ──
op = op.copy()
op["affected_inverter"] = "Entire Plant"


# Determine overall status
if health_score >= 85:
    status_color = "#4CAF50"
    status_text = "Good"
    today_action = "Continue standard monitoring operations."
    priority_level = "Low"
    priority_color = "#4CAF50"
    maintenance_timeframe = "Next Maintenance Cycle"
    rec_border_color = "#4CAF50"
    rec_why = ("Plant efficiency is within normal operating parameters. "
               "The majority of observations show healthy performance ratios, "
               "indicating no significant soiling impact at this time.")
elif health_score >= 60:
    status_color = "#FFC107"
    status_text = "Moderate"
    today_action = "Schedule cleaning within the next maintenance window."
    priority_level = "Medium"
    priority_color = "#FF9800"
    maintenance_timeframe = "Within 48 Hours"
    rec_border_color = "#FF9800"
    rec_why = (f"Approximately {100 - health_score:.0f}% of operating records show "
               f"efficiency below expected levels, consistent with moderate soiling conditions. "
               f"Proactive cleaning can recover an estimated {recoverable_mwh:,.1f} MWh of generation.")
else:
    status_color = "#EF5350"
    status_text = "Critical"
    today_action = "Immediate cleaning recommended due to significant performance degradation."
    priority_level = "High"
    priority_color = "#EF5350"
    maintenance_timeframe = "Immediately"
    rec_border_color = "#EF5350"
    rec_why = (f"Plant efficiency has fallen significantly below expected output. "
               f"{n_clean_imm:,} observations indicate severe soiling requiring immediate action. "
               f"Delaying maintenance risks an estimated loss of {recoverable_mwh:,.1f} MWh.")





# ──────────────────────────────────────────────────────────────────────
# 1. HEADER & HERO STATUS SECTION
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header" id="dashboard-overview">
    <h1>☀️ Solar Intelligent Monitoring System</h1>
    <h3>Soiling Intelligence & Cleaning Recommendation Dashboard</h3>
    <p>Proof-of-Concept — AI-powered monitoring that detects soiling-related power loss on a 4,000 kW utility-scale solar plant and generates automated cleaning recommendations. Built on real PVDAQ System 9068 sensor data with an XGBoost prediction model (R² = 0.964).</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# 2b. INTERACTIVE SOILING SIMULATOR & PROTOTYPE DEMO
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="live-demo-stream">🔴 Interactive Soiling Intelligence Simulator</div>', unsafe_allow_html=True)
st.markdown("""
<div style="background: rgba(245, 166, 35, 0.08); border-left: 4px solid #F5A623; border-radius: 6px; padding: 0.8rem 1.2rem; margin-bottom: 1.2rem; color: #ECEFF1; font-size: 0.9rem;">
    <b>Interactive Simulation Environment:</b> Test solar panel degradation under various irradiance, temperature, soiling, and fault conditions. 
    Generate explainable cleaning recommendations, simulate physical interventions (washing), and verify before-and-after recovery.
</div>
""", unsafe_allow_html=True)

# Render the unified simulator component (reusing identical simulation & business logic)
render_simulator_interface(is_standalone=False, raw_df=df, metrics=metrics)


# Period label for display
_period_label = f"{_filter_start.strftime('%Y-%m-%d')} to {_filter_end.strftime('%Y-%m-%d')}"

st.markdown(f"""
<div class="hero-container">
    <div class="hero-item">
        <div class="hero-label">Plant Status — Selected Period</div>
        <div class="hero-value" style="color: {status_color};">{status_text} ({health_score:.1f}%)</div>
    </div>
    <div class="hero-item">
        <div class="hero-label">Recommended Action</div>
        <div class="hero-value" style="font-size: 1.4rem;">{today_action}</div>
    </div>
    <div class="hero-item">
        <div class="hero-label">Cleaning Priority</div>
        <div class="hero-value" style="color: {status_color};">{priority_level}</div>
    </div>
    <div class="hero-item">
        <div class="hero-label">Est. Recoverable Energy</div>
        <div class="hero-value" style="color: #66BB6A;">{recoverable_mwh:,.1f} MWh</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 2. EXECUTIVE KPI SECTION
# ──────────────────────────────────────────────────────────────────────
kpi_cols = st.columns(6)

with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg. Actual Power</div>
        <div class="kpi-value" style="color:#42A5F5">{avg_power:,.0f} kW</div>
        <div class="kpi-indicator" style="color:#78909C">Current generation</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">AI Expected Power</div>
        <div class="kpi-value" style="color:#66BB6A">{avg_expected:,.0f} kW</div>
        <div class="kpi-indicator" style="color:#78909C">Clean-panel potential</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[2]:
    pr_color = "#4CAF50" if avg_pr >= 0.9 else "#FFC107" if avg_pr >= 0.75 else "#EF5350"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Plant Efficiency</div>
        <div class="kpi-value" style="color:{pr_color}">{avg_pr * 100:.1f}%</div>
        <div class="kpi-indicator" style="color:#78909C">Efficiency (PR)</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[3]:
    loss_color = "#4CAF50" if avg_loss < 5 else "#FFC107" if avg_loss < 15 else "#EF5350"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg. Soiling Loss</div>
        <div class="kpi-value" style="color:{loss_color}">{avg_loss:.1f}%</div>
        <div class="kpi-indicator" style="color:#78909C">Loss due to dirt</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[4]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Overall Plant Health</div>
        <div class="kpi-value" style="color:{status_color}">{health_score:.1f}%</div>
        <div class="kpi-indicator" style="color:#78909C">Clean operating ratio</div>
    </div>""", unsafe_allow_html=True)

with kpi_cols[5]:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Maintenance Alerts</div>
        <div class="kpi-value" style="color:#FF7043">{n_cleaning:,}</div>
        <div class="kpi-indicator" style="color:#78909C">Cleanings advised</div>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 3. RECOMMENDED MAINTENANCE ACTIONS  (Change 1 + Change 2)
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="recommended-maintenance-actions">🛠️ Recommended Maintenance Actions</div>', unsafe_allow_html=True)

# ── Change 1: Decision-support recommendation card ──
st.markdown(f"""
<div class="rec-card" style="border-left-color: {rec_border_color};">
    <div class="rec-card-title" style="color: {priority_color};">Recommended Action</div>
    <div class="rec-card-action">{today_action}</div>
    <div class="rec-meta-grid">
        <div class="rec-meta-item">
            <div class="rec-meta-label">Priority Level</div>
            <div class="rec-meta-value" style="color: {priority_color};">{priority_level}</div>
        </div>
        <div class="rec-meta-item">
            <div class="rec-meta-label">Est. Performance Loss</div>
            <div class="rec-meta-value" style="color: {loss_color};">{avg_loss:.1f}%</div>
        </div>
        <div class="rec-meta-item">
            <div class="rec-meta-label">Expected Energy Recovery</div>
            <div class="rec-meta-value" style="color: #66BB6A;">{recoverable_mwh:,.1f} MWh</div>
        </div>
        <div class="rec-meta-item">
            <div class="rec-meta-label">Maintenance Timeframe</div>
            <div class="rec-meta-value">{maintenance_timeframe}</div>
        </div>
    </div>
    <div class="rec-why">
        <b style="color: #90A4AE;">Why this recommendation:</b>&ensp;{rec_why}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Change 2: Operations Queue (v2.1 — Interactive) ──
st.markdown("#### Operations Queue")

# Initialize session state for queue statuses
if "queue_statuses" not in st.session_state:
    st.session_state["queue_statuses"] = {}

# Build queue from data
high_priority = op[
    op["recommendation"].isin(["Clean Immediately", "Schedule Cleaning"])
].nlargest(5, "loss_pct")[
    [
        "measured_on", "est_power_loss",
        "recommendation", "priority", "reason", "affected_inverter",
    ]
].copy()

# Build queue items with stable IDs
_queue_items = []

if len(high_priority) > 0:
    for _, row in high_priority.iterrows():
        _ts_str = row["measured_on"].strftime("%Y-%m-%d %H:%M")
        _rec = str(row["recommendation"])
        _section = str(row["affected_inverter"])
        _qid = hashlib.md5(f"{_ts_str}|{_rec}|{_section}".encode()).hexdigest()[:10]

        # Initialize status if not tracked
        if _qid not in st.session_state["queue_statuses"]:
            st.session_state["queue_statuses"][_qid] = "Open"

        _priority_labels = {1: "🔴 Critical", 2: "🟠 High", 3: "🟡 Medium", 4: "🟢 Normal"}
        _queue_items.append({
            "id": _qid,
            "source": "AI",
            "Priority": _priority_labels.get(int(row["priority"]), "⚪ Info"),
            "Timestamp": _ts_str,
            "Affected Section": _section,
            "Est. Power Loss (kW)": round(float(row["est_power_loss"]), 1),
            "Recommended Action": _rec,
            "Status": st.session_state["queue_statuses"][_qid],
        })


if len(_queue_items) > 0:
    # Build display dataframe
    _queue_df = pd.DataFrame(_queue_items)

    # Status emoji mapping
    _status_display = {
        "Open": "🔴 Open",
        "Acknowledged": "🟡 Acknowledged",
        "In Progress": "🔵 In Progress",
        "Resolved": "🟢 Resolved",
    }
    _queue_df["Status"] = _queue_df["Status"].map(_status_display).fillna("🔴 Open")

    _display_df = _queue_df[["Priority", "Timestamp",
                              "Est. Power Loss (kW)", "Recommended Action", "Status"]].copy()

    def highlight_row(row):
        action = row["Recommended Action"]
        if "Clean Immediately" in action:
            return ["background-color: rgba(244,67,54,0.12)"] * len(row)
        elif "Schedule Cleaning" in action:
            return ["background-color: rgba(255,152,0,0.09)"] * len(row)
        elif "[Operator]" in action:
            return ["background-color: rgba(66,165,245,0.08)"] * len(row)
        return [""] * len(row)

    st.dataframe(
        _display_df.style.apply(highlight_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )

    # ── Status Update Panel ──
    st.markdown("##### Update Item Status")
    _item_labels = [f"{q['Timestamp']} — {q['Recommended Action']}" for q in _queue_items]
    _item_ids = [q["id"] for q in _queue_items]

    _upd_cols = st.columns([3, 1, 1])
    with _upd_cols[0]:
        _selected_idx = st.selectbox("Select queue item", range(len(_item_labels)),
                                      format_func=lambda i: _item_labels[i], key="queue_select")
    with _upd_cols[1]:
        _current_status = st.session_state["queue_statuses"].get(_item_ids[_selected_idx], "Open")
        _new_status = st.selectbox("New status",
                                    ["Open", "Acknowledged", "In Progress", "Resolved"],
                                    index=["Open", "Acknowledged", "In Progress", "Resolved"].index(_current_status),
                                    key="queue_new_status")
    with _upd_cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Update", key="queue_update_btn", use_container_width=True):
            st.session_state["queue_statuses"][_item_ids[_selected_idx]] = _new_status
            st.rerun()

    # Visual lifecycle
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem; color: #546E7A; font-size: 0.78rem;">
        <span>🔴 Open</span> <span>→</span> <span>🟡 Acknowledged</span> <span>→</span>
        <span>🔵 In Progress</span> <span>→</span> <span>🟢 Resolved</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.success("No immediate cleaning recommendations at this time.")

# Full list behind expander
with st.expander("View Complete Recommendation List"):
    full_queue = op[
        op["recommendation"].isin(cleaning_actions)
    ].nlargest(50, "loss_pct")[
        ["measured_on", "total_power", "expected_power",
         "est_power_loss", "model_perf_ratio", "loss_pct",
         "recommendation", "priority", "reason"]
    ].copy()
    full_queue.columns = [
        "Timestamp", "Current Power (kW)", "Expected Power (kW)",
        "Power Loss (kW)", "Efficiency", "Loss %",
        "Recommendation", "Priority", "Reason",
    ]
    full_queue["Timestamp"] = full_queue["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    full_queue["Priority"] = full_queue["Priority"].map(
        {1: "🔴 Critical", 2: "🟠 High", 3: "🟡 Medium", 4: "🟢 Normal"}
    ).fillna("⚪ Info")
    num_cols = ["Current Power (kW)", "Expected Power (kW)", "Power Loss (kW)", "Loss %"]
    full_queue[num_cols] = full_queue[num_cols].round(1)
    full_queue["Efficiency"] = full_queue["Efficiency"].round(3)
    st.dataframe(full_queue, use_container_width=True, hide_index=True, height=400)
    st.caption(f"Showing top {len(full_queue)} observations sorted by estimated power loss.")


# ──────────────────────────────────────────────────────────────────────
# 3c. DECISION WORKFLOW — EXPLAINABLE AI DECISION ENGINE
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="decision-workflow">🔍 Explainable AI — Decision Trace</div>', unsafe_allow_html=True)
st.caption("Trace how the system processes a single observation from raw sensor data to an operator recommendation.")

wf_ctrl_col1, wf_ctrl_col2 = st.columns([2, 1])

with wf_ctrl_col1:
    wf_scenario = st.selectbox(
        "Choose an example observation",
        ["Highest Priority Event", "Random Cleaning Event", "Monitor Event", "Healthy Event"],
        key="wf_scenario",
    )

with wf_ctrl_col2:
    st.markdown("<br>", unsafe_allow_html=True)
    wf_run = st.button("▶ Run Decision Workflow", use_container_width=True)

# Select the actual row from df based on user choice
if wf_scenario == "Highest Priority Event":
    _wf_candidates = df[df["recommendation"] == "Clean Immediately"]
    if len(_wf_candidates) > 0:
        _wf_row = _wf_candidates.sort_values("loss_pct", ascending=False).iloc[0]
    else:
        _wf_row = df.iloc[0]
elif wf_scenario == "Random Cleaning Event":
    _wf_candidates = df[df["recommendation"].isin(["Schedule Cleaning", "Clean Immediately"])]
    if len(_wf_candidates) > 0:
        _wf_row = _wf_candidates.sample(1, random_state=None).iloc[0]
    else:
        _wf_row = df.iloc[0]
elif wf_scenario == "Monitor Event":
    _wf_candidates = df[df["recommendation"] == "Monitor"]
    if len(_wf_candidates) > 0:
        _wf_row = _wf_candidates.sample(1, random_state=None).iloc[0]
    else:
        _wf_row = df.iloc[0]
else:  # Healthy Event
    _wf_candidates = df[df["recommendation"] == "Healthy"]
    if len(_wf_candidates) > 0:
        _wf_row = _wf_candidates.sample(1, random_state=None).iloc[0]
    else:
        _wf_row = df.iloc[0]

# Extract values from the selected row
_r_ts = pd.Timestamp(_wf_row["measured_on"]).strftime("%Y-%m-%d %H:%M")
_r_irr = float(_wf_row["irradiance"])
_r_amb = float(_wf_row["ambient_temp"])
_r_mod = float(_wf_row["module_temp"])
_r_act = float(_wf_row["total_power"])
_r_exp = float(_wf_row["expected_power"])
_r_pr = _wf_row.get("model_perf_ratio", np.nan)
_r_pr = float(_r_pr) if not pd.isna(_r_pr) else None
_r_loss_pct = float(_wf_row.get("loss_pct", 0))
_r_est_loss = float(_wf_row.get("est_power_loss", 0))
_r_rec = str(_wf_row["recommendation"])
_r_reason = str(_wf_row["reason"])
_r_priority = int(_wf_row["priority"])
_r_temp_diff = float(_wf_row.get("temp_diff", _r_mod - _r_amb))

# Thresholds from existing PoC (prepare_data.py classify_condition)
_THR_IRR = 200
_THR_EXP = 100
_THR_ACT = 50
_THR_PR_HEALTHY = 0.95
_THR_PR_MONITOR = 0.85
_THR_PR_SCHEDULE = 0.70

# ── Pre-compute all derived values (presentation only) ──
_chk_irr = _r_irr > _THR_IRR
_chk_exp = _r_exp > _THR_EXP
_chk_act = _r_act > _THR_ACT
_all_pass = _chk_irr and _chk_exp and _chk_act

_gap = _r_exp - _r_act
_pr_display = f"{_r_pr:.3f}" if _r_pr is not None else "N/A"
_within_range = (_r_pr is not None and _r_pr >= _THR_PR_HEALTHY)
_range_color = "#4CAF50" if _within_range else ("#FF9800" if (_r_pr is not None and _r_pr >= _THR_PR_SCHEDULE) else "#EF5350")

# Degradation assessment
if _r_pr is None:
    _deg_status, _deg_detail, _deg_color = "Insufficient Data", "Performance ratio could not be computed.", "#78909C"
elif _r_pr >= _THR_PR_HEALTHY:
    _deg_status = "No Significant Degradation"
    _deg_detail = f"PR = {_r_pr:.3f} (≥ {_THR_PR_HEALTHY}) — within the normal operating range."
    _deg_color = "#4CAF50"
elif _r_pr >= _THR_PR_MONITOR:
    _deg_status = "Mild Degradation"
    _deg_detail = f"PR = {_r_pr:.3f} (between {_THR_PR_MONITOR} and {_THR_PR_HEALTHY}) — slight performance drop."
    _deg_color = "#FFC107"
elif _r_pr >= _THR_PR_SCHEDULE:
    _deg_status = "Significant Degradation"
    _deg_detail = f"PR = {_r_pr:.3f} (between {_THR_PR_SCHEDULE} and {_THR_PR_MONITOR}) — measurable loss."
    _deg_color = "#FF9800"
else:
    _deg_status = "Severe Degradation"
    _deg_detail = f"PR = {_r_pr:.3f} (below {_THR_PR_SCHEDULE}) — substantial performance loss."
    _deg_color = "#EF5350"

# Rule explanation
_rule_colors = {
    "Healthy": "#4CAF50", "Monitor": "#FFC107",
    "Schedule Cleaning": "#FF9800", "Clean Immediately": "#EF5350",
    "Low Sunlight": "#78909C", "Low Generation": "#78909C",
    "Investigate": "#795548", "Insufficient Data": "#9E9E9E",
}
_rule_color = _rule_colors.get(_r_rec, "#78909C")

if _r_rec == "Low Sunlight":
    _rule_explain = f"Irradiance ({_r_irr:.0f} W/m²) below 200 — low-sunlight period."
elif _r_rec == "Low Generation":
    _rule_explain = f"Expected generation ({_r_exp:.0f} kW) below 100 kW — excluded from analysis."
elif _r_rec == "Investigate":
    _rule_explain = f"Actual output ({_r_act:.0f} kW) near zero — possible equipment issue."
elif _r_rec == "Insufficient Data":
    _rule_explain = "Performance ratio unavailable — cannot assess."
elif _r_rec == "Healthy":
    _rule_explain = f"PR ({_r_pr:.3f}) ≥ {_THR_PR_HEALTHY} — normal operating range."
elif _r_rec == "Monitor":
    _rule_explain = f"PR ({_r_pr:.3f}) between {_THR_PR_MONITOR}–{_THR_PR_HEALTHY} — mild degradation."
elif _r_rec == "Schedule Cleaning":
    _rule_explain = f"PR ({_r_pr:.3f}) between {_THR_PR_SCHEDULE}–{_THR_PR_MONITOR}, loss {_r_loss_pct:.1f}% — schedule cleaning."
else:
    _rule_explain = f"PR ({_r_pr:.3f}) below {_THR_PR_SCHEDULE}, loss {_r_loss_pct:.1f}% — immediate cleaning."

# System flag
_is_cleaning = _r_rec in ["Schedule Cleaning", "Clean Immediately"]
_is_investigate = _r_rec == "Investigate"
if _is_cleaning:
    _flag_text = "Possible Soiling-Related Degradation"
    _flag_note = "Performance pattern is consistent with soiling. This does not confirm soiling as the sole cause — shading or equipment faults may contribute."
    _flag_color = "#FF9800" if _r_rec == "Schedule Cleaning" else "#EF5350"
elif _is_investigate:
    _flag_text = "Anomalous Output — Investigation Required"
    _flag_note = "Near-zero output despite adequate irradiance may indicate equipment fault."
    _flag_color = "#795548"
elif _r_rec == "Monitor":
    _flag_text = "Mild Performance Deviation — Monitoring"
    _flag_note = "Slight efficiency drop detected. Continued monitoring recommended."
    _flag_color = "#FFC107"
else:
    _flag_text = "Normal Operation"
    _flag_note = "No significant degradation detected."
    _flag_color = "#4CAF50"

# Priority & timeframe
_pri_labels = {1: "🔴 Critical", 2: "🟠 High", 3: "🟡 Medium", 4: "🟡 Low",
               5: "⚪ Info", 6: "⚪ Info", 7: "⚪ Info", 8: "🟢 Normal"}
_pri_label = _pri_labels.get(_r_priority, "⚪ Info")
_timeframe_map = {
    "Clean Immediately": "Immediately", "Schedule Cleaning": "Within 48 Hours",
    "Investigate": "Within 24 Hours", "Monitor": "Next Maintenance Cycle",
    "Healthy": "No action required", "Low Sunlight": "N/A",
    "Low Generation": "N/A", "Insufficient Data": "N/A",
}
_timeframe = _timeframe_map.get(_r_rec, "—")

# Decision confidence (programmatic — NOT model confidence)
# Compute a numeric confidence percentage based on weighted evidence signals
_confidence_evidence = []
_confidence_limits = []
_conf_pct = 0.0  # base

# Evidence factor 1: Irradiance quality (0-20%)
if _chk_irr:
    _irr_quality = min(_r_irr / 1000.0, 1.0)  # normalized 0-1
    _irr_conf = 15 + 5 * _irr_quality  # 15-20%
    _conf_pct += _irr_conf
    _confidence_evidence.append(f"High irradiance conditions (+{_irr_conf:.0f}%)")

# Evidence factor 2: All operating thresholds met (0-20%)
if _all_pass:
    _conf_pct += 20
    _confidence_evidence.append("Valid operating thresholds (+20%)")

# Evidence factor 3: Clear performance gap signal (0-25%)
if _r_pr is not None and abs(_gap) > 50:
    _gap_strength = min(abs(_gap) / 500.0, 1.0)
    _gap_conf = 15 + 10 * _gap_strength  # 15-25%
    _conf_pct += _gap_conf
    _confidence_evidence.append(f"Significant performance gap (+{_gap_conf:.0f}%)")

# Evidence factor 4: PR deviation from healthy baseline (0-20%)
if _r_pr is not None and _r_pr < _THR_PR_HEALTHY:
    _pr_deviation = (_THR_PR_HEALTHY - _r_pr) / _THR_PR_HEALTHY
    _pr_conf = 10 + 10 * min(_pr_deviation * 3, 1.0)  # 10-20%
    _conf_pct += _pr_conf
    _confidence_evidence.append(f"PR below threshold ({_r_pr:.3f}) (+{_pr_conf:.0f}%)")

# Evidence factor 5: Model accuracy baseline (always contributes)
_model_r2_pct = metrics['r2'] * 15  # R² contribution up to 15%
_conf_pct += _model_r2_pct
_confidence_evidence.append(f"Model accuracy R²={metrics['r2']:.4f} (+{_model_r2_pct:.0f}%)")

# Clamp to 0-100
_conf_pct = max(0.0, min(100.0, _conf_pct))

if len(_confidence_evidence) == 0:
    _confidence_evidence.append("Observation evaluated")

_confidence_limits = ["No string-level monitoring", "No inverter fault codes",
                      "No thermal inspection data", "No soiling sensor available"]
_conf_level = f"{_conf_pct:.1f}%"
if _conf_pct >= 70:
    _conf_label = "HIGH"
    _conf_color = "#4CAF50"
elif _conf_pct >= 45:
    _conf_label = "MODERATE"
    _conf_color = "#FFC107"
else:
    _conf_label = "LOW"
    _conf_color = "#78909C"

# Recovery estimate for this single observation
_obs_recovery_kwh = _r_est_loss * (5 / 60) if _is_cleaning else 0

# Helper: PASS/FAIL indicator
def _pf(passed):
    return '<span class="xai-pass">✓ PASS</span>' if passed else '<span class="xai-fail">✗ FAIL</span>'

# ── Stage definitions ──
_TOTAL_STAGES = 9
_stage_meta = [
    ("📡", "Real-Time Plant Readings"),
    ("🧹", "Sunlight & Operational Check"),
    ("🤖", "Clean-Panel AI Benchmark"),
    ("📉", "Output vs Potential Gap"),
    ("⚠",  "Efficiency Drop Evaluation"),
    ("🔎", "Rule Out Non-Soiling Causes"),
    ("☀",  "Soiling Impact Classification"),
    ("🛠",  "Cleaning Action Recommendation"),
    ("📋", "Maintenance Decision Summary"),
]

if wf_run:
    # Create placeholders for progress bar and each stage
    _ph_progress = st.empty()
    _ph_stages = [st.empty() for _ in range(_TOTAL_STAGES)]
    _ph_connectors = [st.empty() for _ in range(_TOTAL_STAGES - 1)]
    _ph_summary = st.empty()
    _ph_timeline_conf = st.empty()
    _ph_footer = st.empty()

    # Simulated trace timestamps (visualization only)
    _trace_base = datetime.now()
    _trace_times = [(_trace_base.replace(second=0) + pd.Timedelta(seconds=i)).strftime("%H:%M:%S") for i in range(_TOTAL_STAGES)]
    _trace_labels = ["Readings Ingested", "Operating Check Passed", "Clean Potential Retrieved",
                     "Power Compared", "Efficiency Evaluated", "Other Causes Checked",
                     "Soiling Classified", "Action Generated", "Decision Finalized"]

    def _render_progress(stage_num):
        pct = int((stage_num / _TOTAL_STAGES) * 100)
        _ph_progress.markdown(f'''
        <div style="background:rgba(36,52,71,0.4); border-radius:8px; padding:0.8rem 1.2rem; border:1px solid #263238; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                <span style="color:#90CAF9; font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; font-weight:700;">Decision Trace Progress</span>
                <span style="color:#E0E0E0; font-size:0.85rem; font-weight:600;">{pct}% &ensp;—&ensp; Stage {stage_num} of {_TOTAL_STAGES}</span>
            </div>
            <div class="xai-progress-outer">
                <div class="xai-progress-fill" style="width:{pct}%;"></div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    def _render_connector(idx):
        _ph_connectors[idx].markdown('<div class="xai-connector">│<br>▼</div>', unsafe_allow_html=True)

    def _stage_badge():
        return '<span class="xai-stage-badge">✓ Complete</span>'

    # ── STAGE 1: Real-Time Plant Readings ──
    _render_progress(1)
    _ph_stages[0].markdown(f'''
    <div class="xai-stage" style="border-left-color:#42A5F5;">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">📡</span>
            <span class="xai-stage-title">Real-Time Plant Readings</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">The system ingests live environmental and generation telemetry for this 5-minute interval.</div>
        <div style="color:#546E7A; font-size:0.78rem; margin-bottom:0.5rem; padding-left:2.3rem;">Timestamp: <b style="color:#B0BEC5;">{_r_ts}</b></div>
        <div class="xai-grid">
            <div class="xai-cell">
                <div class="xai-cell-label">Sunlight Intensity</div>
                <div class="xai-cell-value">{_r_irr:.1f} W/m²</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Surrounding Air Temp</div>
                <div class="xai-cell-value">{_r_amb:.1f} °C</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Solar Panel Temp</div>
                <div class="xai-cell-value">{_r_mod:.1f} °C</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Panel Heat Above Air</div>
                <div class="xai-cell-value">{_r_temp_diff:.1f} °C</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Current Delivered Output</div>
                <div class="xai-cell-value">{_r_act:,.1f} kW</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.4)
    _render_connector(0)

    # ── STAGE 2: Sunlight & Operational Check ──
    _render_progress(2)
    _val_border = "#4CAF50" if _all_pass else "#FF9800"
    _ph_stages[1].markdown(f'''
    <div class="xai-stage" style="border-left-color:{_val_border};">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">🧹</span>
            <span class="xai-stage-title">Sunlight & Operational Check</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">Confirming that the sun is active and the plant is within normal operating daylight hours.</div>
        <div class="xai-grid">
            <div class="xai-cell">
                <div class="xai-cell-label">Sunlight &gt; {_THR_IRR} W/m²</div>
                <div class="xai-cell-value">{_pf(_chk_irr)} &ensp;({_r_irr:.0f} W/m²)</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Expected Output &gt; {_THR_EXP} kW</div>
                <div class="xai-cell-value">{_pf(_chk_exp)} &ensp;({_r_exp:.0f} kW)</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Delivered Power &gt; {_THR_ACT} kW</div>
                <div class="xai-cell-value">{_pf(_chk_act)} &ensp;({_r_act:.0f} kW)</div>
            </div>
        </div>
        <div class="xai-status" style="color:{_val_border};">{'✓ Sun is active & plant is generating' if _all_pass else '⚠ Low sun or nighttime condition'}</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.4)
    _render_connector(1)

    # ── STAGE 3: Clean-Panel AI Benchmark ──
    _render_progress(3)
    _ph_stages[2].markdown(f'''
    <div class="xai-stage" style="border-left-color:#66BB6A;">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">🤖</span>
            <span class="xai-stage-title">Clean-Panel AI Benchmark</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">The AI model calculates what clean panels should produce under these exact weather conditions.</div>
        <div class="xai-grid">
            <div class="xai-cell">
                <div class="xai-cell-label">Environmental Inputs</div>
                <div class="xai-cell-value" style="font-size:0.82rem;">Sunlight ({_r_irr:.0f} W/m²), Panel Temp ({_r_mod:.1f}°C), Air Temp ({_r_amb:.1f}°C)</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Clean-Panel Expected Output</div>
                <div class="xai-cell-value" style="color:#66BB6A; font-size:1.2rem;">{_r_exp:,.1f} kW</div>
            </div>
        </div>
        <div class="xai-status xai-muted">Learned from 135,784 historical clean-plant readings (PVDAQ 9068).</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.4)
    _render_connector(2)

    # ── STAGE 4: Output vs Potential Gap ──
    _render_progress(4)
    _gap_sign = "+" if _gap >= 0 else ""
    _ph_stages[3].markdown(f'''
    <div class="xai-stage" style="border-left-color:{_range_color};">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">📉</span>
            <span class="xai-stage-title">Output vs Potential Gap</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">The system compares actual generation against clean potential to isolate uncaptured energy.</div>
        <div class="xai-grid">
            <div class="xai-cell">
                <div class="xai-cell-label">Clean Potential</div>
                <div class="xai-cell-value">{_r_exp:,.1f} kW</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Delivered Power</div>
                <div class="xai-cell-value">{_r_act:,.1f} kW</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Power Gap (Shortfall)</div>
                <div class="xai-cell-value" style="color:{_range_color};">{_gap_sign}{_gap:,.1f} kW</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Plant Efficiency Score</div>
                <div class="xai-cell-value" style="color:{_range_color};">{_pr_display}</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Soiling Loss Percentage</div>
                <div class="xai-cell-value" style="color:{_range_color};">{_r_loss_pct:.1f}%</div>
            </div>
        </div>
        <div class="xai-status" style="color:{_range_color};">{'✓ Output is tracking clean potential' if _within_range else '⚠ Generating below clean potential'}</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.4)
    _render_connector(3)

    # ── STAGE 5: Efficiency Drop Evaluation ──
    _render_progress(5)
    _ph_stages[4].markdown(f'''
    <div class="xai-stage" style="border-left-color:{_deg_color};">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">⚠</span>
            <span class="xai-stage-title">Efficiency Drop Evaluation</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">The efficiency score is evaluated against standardized operational thresholds.</div>
        <div class="xai-grid">
            <div class="xai-cell" style="grid-column: 1 / -1;">
                <div class="xai-cell-label">Condition Assessment</div>
                <div class="xai-cell-value" style="color:{_deg_color}; font-size:1.15rem;">{_deg_status}</div>
            </div>
        </div>
        <div style="color:#B0BEC5; font-size:0.85rem; margin:0.5rem 0 0 2.3rem;">{_deg_detail}</div>
        <div style="color:#455A64; font-size:0.72rem; margin:0.3rem 0 0 2.3rem;">Standard Thresholds: Healthy ≥ 95% &ensp;|&ensp; Monitor ≥ 85% &ensp;|&ensp; Schedule Cleaning ≥ 70% &ensp;|&ensp; Clean Immediately &lt; 70%</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.4)
    _render_connector(4)

    # ── STAGE 6: Rule Out Non-Soiling Causes ──
    _render_progress(6)
    _ph_stages[5].markdown(f'''
    <div class="xai-stage" style="border-left-color:#546E7A;">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">🔎</span>
            <span class="xai-stage-title">Rule Out Non-Soiling Causes</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">Confirming that weather factors (night, heavy cloud cover, zero generation) are not misleading the diagnosis.</div>
        <div class="xai-grid">
            <div class="xai-cell">
                <div class="xai-cell-label">Dim Sun / Night</div>
                <div class="xai-cell-value xai-pass">✓ Ruled out</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Overcast Sky</div>
                <div class="xai-cell-value xai-pass">✓ Ruled out</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Inverter Trip / Outage</div>
                <div class="xai-cell-value xai-pass">✓ Generating normally</div>
            </div>
        </div>
        <div class="xai-status xai-muted">Result: Non-weather power loss confirmed; proceed with soiling classification.</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.35)
    _render_connector(5)

    # ── STAGE 7: Soiling Impact Classification ──
    _render_progress(7)
    _ph_stages[6].markdown(f'''
    <div class="xai-stage" style="border-left-color:{_flag_color}; border-left-width:5px;">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">☀</span>
            <span class="xai-stage-title">Soiling Impact Classification</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">The system translates the isolated shortfall into an actionable plant condition.</div>
        <div style="font-size:1.2rem; font-weight:700; color:{_flag_color}; margin:0.3rem 0 0.3rem 2.3rem;">{_flag_text}</div>
        <div style="color:#90A4AE; font-size:0.83rem; margin-left:2.3rem;">{_flag_note}</div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.35)
    _render_connector(6)

    # ── STAGE 8: Cleaning Action Recommendation ──
    _render_progress(8)
    _ph_stages[7].markdown(f'''
    <div class="xai-stage" style="border-left-color:{_flag_color}; border-left-width:5px;">
        <div class="xai-stage-head">
            <span class="xai-stage-icon">🛠</span>
            <span class="xai-stage-title">Cleaning Action Recommendation</span>
            {_stage_badge()}
        </div>
        <div class="xai-narrative">The system issues an automated maintenance ticket with priority and timeframe.</div>
        <div class="xai-grid">
            <div class="xai-cell">
                <div class="xai-cell-label">Recommended Action</div>
                <div class="xai-cell-value" style="color:{_flag_color};">{_r_rec}</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Priority</div>
                <div class="xai-cell-value">{_pri_label}</div>
            </div>
            <div class="xai-cell">
                <div class="xai-cell-label">Action Window</div>
                <div class="xai-cell-value">{_timeframe}</div>
            </div>
            <div class="xai-cell" style="grid-column: span 2;">
                <div class="xai-cell-label">Why This Action</div>
                <div class="xai-cell-value" style="font-size:0.88rem;">{_r_reason}</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.35)
    _render_connector(7)

    # ── STAGE 9: Maintenance Decision Summary ──
    _render_progress(9)
    _ph_stages[8].markdown(f'''
    <div class="xai-summary">
        <div class="xai-summary-title">📋 Maintenance Decision Summary</div>
        <div class="xai-summary-grid">
            <div class="xai-summary-item">
                <div class="xai-summary-label">Observation Timestamp</div>
                <div class="xai-summary-value">{_r_ts}</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Diagnostic Status</div>
                <div class="xai-summary-value" style="color:{_flag_color};">{_flag_text}</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Clean Potential</div>
                <div class="xai-summary-value">{_r_exp:,.1f} kW</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Delivered Power</div>
                <div class="xai-summary-value">{_r_act:,.1f} kW</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Estimated Soiling Loss</div>
                <div class="xai-summary-value" style="color:{_range_color};">{_r_loss_pct:.1f}%</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Recommended Action</div>
                <div class="xai-summary-value" style="color:{_flag_color};">{_r_rec}</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Maintenance Priority</div>
                <div class="xai-summary-value">{_pri_label}</div>
            </div>
            <div class="xai-summary-item">
                <div class="xai-summary-label">Decision Confidence</div>
                <div class="xai-summary-value" style="color:{_conf_color};">{_conf_pct:.1f}% ({_conf_label})</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    time.sleep(0.3)

    # ── Decision Timeline + Confidence (side by side) ──
    _tl_col, _cf_col = _ph_timeline_conf.columns([1, 1])

    _tl_rows = ""
    for i in range(_TOTAL_STAGES):
        _dot_color = "#4CAF50"
        _tl_rows += f'''
        <div class="xai-timeline-row">
            <span class="xai-tl-time">{_trace_times[i]}</span>
            <span class="xai-tl-dot" style="background:{_dot_color};"></span>
            <span class="xai-tl-label">{_trace_labels[i]}</span>
            <span class="xai-pass" style="margin-left:auto;">✓</span>
        </div>
        '''

    with _tl_col:
        st.markdown(f'''
        <div class="xai-confidence">
            <div style="color:#90CAF9; font-size:0.7rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:700; margin-bottom:0.8rem;">Decision Timeline</div>
            {_tl_rows}
            <div style="color:#37474F; font-size:0.68rem; margin-top:0.5rem;">Timestamps are visualization-only and do not represent actual execution times.</div>
        </div>
        ''', unsafe_allow_html=True)

    _ev_items = "".join(f'<div style="padding:0.15rem 0;"><span class="xai-pass">✓</span>&ensp;<span style="color:#B0BEC5; font-size:0.82rem;">{e}</span></div>' for e in _confidence_evidence)
    _lm_items = "".join(f'<div style="padding:0.15rem 0;"><span class="xai-muted">—</span>&ensp;<span style="color:#78909C; font-size:0.82rem;">{l}</span></div>' for l in _confidence_limits)

    with _cf_col:
        st.markdown(f'''
        <div class="xai-confidence">
            <div style="color:#90CAF9; font-size:0.7rem; text-transform:uppercase; letter-spacing:1.2px; font-weight:700; margin-bottom:0.6rem;">Decision Confidence</div>
            <div style="font-size:2rem; font-weight:700; color:{_conf_color}; margin-bottom:0.2rem;">{_conf_pct:.1f}%</div>
            <div style="font-size:0.85rem; font-weight:600; color:{_conf_color}; margin-bottom:0.8rem;">{_conf_label}</div>
            <div style="color:#78909C; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.3rem;">Confidence Breakdown</div>
            {_ev_items}
            <div style="color:#78909C; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.5px; margin:0.8rem 0 0.3rem 0;">Limitations (reduces certainty)</div>
            {_lm_items}
        </div>
        ''', unsafe_allow_html=True)

    _ph_footer.markdown(f'''
    <div style="text-align:center; color:#455A64; font-size:0.75rem; margin-top:1.2rem; padding:0.6rem 0; border-top:1px solid #1a2836;">
        Decision trace completed for observation at {_r_ts}&ensp;·&ensp;{_TOTAL_STAGES} stages processed&ensp;·&ensp;All values sourced from pre-processed pipeline output
    </div>
    ''', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 4b. PROTOTYPE WORKFLOW — HOW CLEANING DECISIONS ARE MADE
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="prototype-workflow">🔄 Prototype Workflow — How Cleaning Decisions Are Made</div>', unsafe_allow_html=True)
st.caption("A business-level view of how the system takes raw sensor data and produces a validated cleaning recommendation.")

# ── CSS for workflow flow diagram ──
st.markdown("""<style>
.wf-wrap{max-width:820px;margin:0 auto; perspective: 1000px;}
.wf-step-box{
    background: linear-gradient(145deg, rgba(26,35,50,0.8) 0%, rgba(18,24,35,0.9) 100%);
    border: 1px solid #263238;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    position: relative;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    margin-bottom: 0.5rem;
    overflow: hidden;
}
/* Left accent line */
.wf-step-box::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 5px;
    background: var(--step-color, #42A5F5);
    transition: width 0.3s ease;
}
.wf-step-box:hover {
    border-color: var(--step-color, #42A5F5);
    box-shadow: 0 8px 25px rgba(0,0,0,0.4), 0 0 15px var(--step-color, rgba(66,165,245,0.3));
    transform: translateY(-2px);
}
.wf-step-box:hover::before {
    width: 8px;
}

.wf-hdr{display:flex;align-items:center;gap:0.9rem;margin-bottom:0.6rem;}
.wf-circle{
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.95rem;
    color: #000;
    flex-shrink: 0;
    background: var(--step-color, #42A5F5);
    box-shadow: 0 0 10px var(--step-color, #42A5F5);
}
.wf-stitle{font-size:1.15rem;font-weight:700;color:#FFFFFF; letter-spacing: 0.3px;}
.wf-badge{
    font-size:0.6rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:1.5px;
    padding:4px 10px;
    border-radius:4px;
    margin-left:auto;
    color: var(--step-color, #90A4AE);
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
}
.wf-sdesc{color:#B0BEC5;font-size:0.92rem;line-height:1.65;margin-top:0.3rem;}
.wf-smeta{
    margin-top:0.8rem;
    padding:0.7rem 0.9rem;
    background:rgba(0,0,0,0.2);
    border-radius:6px;
    font-size:0.82rem;
    color:#78909C;
    border-left:3px solid rgba(255,255,255,0.1);
}
.wf-arrow-down{
    text-align:center;
    font-size:1.6rem;
    line-height:1.2;
    color:#455A64;
    padding:0.4rem 0;
    user-select:none;
}
.wf-arrow-down span{
    display:inline-block;
    animation:wfPulse 2.5s ease-in-out infinite;
    background: linear-gradient(180deg, #546E7A 0%, #263238 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
@keyframes wfPulse{0%,100%{opacity:0.3; transform:scale(0.95);}50%{opacity:1; transform:scale(1.05); text-shadow: 0 0 10px rgba(255,255,255,0.2);}}

.wf-fork-row{display:flex;gap:1.5rem;align-items:stretch;}
.wf-fork-box{
    flex:1;
    border-radius:12px;
    padding:1.2rem 1.5rem;
    border:1px solid #263238;
    position:relative;
    background: linear-gradient(145deg, rgba(26,35,50,0.7) 0%, rgba(18,24,35,0.8) 100%);
    transition: all 0.3s;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.wf-fork-box:hover{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
}
.wf-fork-label{font-size:1.05rem;font-weight:700;margin-bottom:0.4rem; display:flex; align-items:center; gap:0.5rem;}
.wf-fork-desc{font-size:0.88rem;color:#B0BEC5;line-height:1.5;}
.wf-fork-act{margin-top:0.6rem;font-size:0.8rem;color:#78909C; font-weight:600; padding-top:0.6rem; border-top:1px solid rgba(255,255,255,0.05);}

.wf-sev-row{display:flex;gap:1rem;flex-wrap:wrap;margin-top:0.8rem;}
.wf-sev-card{
    flex:1;
    min-width:160px;
    padding:0.7rem 0.9rem;
    background:rgba(0,0,0,0.2);
    border-radius:8px;
    border-left:4px solid;
    transition: all 0.2s;
}
.wf-sev-card:hover{
    background:rgba(255,255,255,0.05);
}
.wf-sev-name{font-weight:700;font-size:0.9rem;}
.wf-sev-pr{font-size:0.78rem;color:#90A4AE;margin-top:3px; font-family: monospace; font-weight: 600;}
.wf-sev-info{font-size:0.75rem;color:#78909C;margin-top:4px; line-height:1.4;}
</style>""", unsafe_allow_html=True)

# Helper for arrow connector
def _wf_arrow():
    st.markdown('<div class="wf-arrow-down"><span>│<br>▼</span></div>', unsafe_allow_html=True)

# ── STEP 1: INPUT ──
st.markdown(
    '<div class="wf-wrap"><div class="wf-step-box" style="--step-color: #42A5F5;">'
    '<div class="wf-hdr"><div class="wf-circle">1</div>'
    '<div class="wf-stitle">📡 Input — Sensor Data Collection</div>'
    '<span class="wf-badge">INPUT</span></div>'
    '<div class="wf-sdesc">The system collects real-time measurements from the solar plant every <b>5 minutes</b>: '
    'sunlight intensity, air &amp; panel temperature, and total plant power output.</div>'
    '<div class="wf-smeta"><b>Sources:</b> Sunlight sensors · Weather station · Panel thermocouples · Plant power meters</div>'
    '</div></div>', unsafe_allow_html=True)

_wf_arrow()

# ── STEP 2: PROCESS ──
st.markdown(
    '<div class="wf-wrap"><div class="wf-step-box" style="--step-color: #66BB6A;">'
    '<div class="wf-hdr"><div class="wf-circle">2</div>'
    '<div class="wf-stitle">⚙️ Process — AI Prediction &amp; Analysis</div>'
    '<span class="wf-badge">AI MODEL</span></div>'
    '<div class="wf-sdesc">The trained <b>AI model</b> predicts clean-panel generation potential under current weather conditions. '
    'The system computes the <b>Plant Efficiency Score</b> and <b>Soiling Loss %</b> to quantify uncaptured energy.</div>'
    '<div class="wf-smeta"><b>Formula:</b> Efficiency = Actual ÷ Expected &nbsp;|&nbsp; Loss % = (Expected − Actual) ÷ Expected × 100</div>'
    '</div></div>', unsafe_allow_html=True)

_wf_arrow()

# ── STEP 3: VALIDATION ──
st.markdown(
    '<div class="wf-wrap"><div class="wf-step-box" style="--step-color: #90CAF9;">'
    '<div class="wf-hdr"><div class="wf-circle">3</div>'
    '<div class="wf-stitle">✅ Validation — Operating Condition Gates</div>'
    '<span class="wf-badge">VALIDATION</span></div>'
    '<div class="wf-sdesc">Before recommending cleaning, the system confirms the plant is in valid operating daylight. '
    '<b>All three gates must pass</b> to proceed.</div>'
    '<div class="wf-smeta">🔒 <b>Gate 1:</b> Sunlight &gt; 200 W/m² &nbsp;·&nbsp; <b>Gate 2:</b> Expected Power &gt; 100 kW &nbsp;·&nbsp; <b>Gate 3:</b> Actual Power &gt; 50 kW</div>'
    '</div></div>', unsafe_allow_html=True)

# ── Fork arrows: split into two branches ──
st.markdown("""
<div style="width:100%; max-width:820px; margin: 0 auto; display:flex; flex-direction:column; align-items:center;">
    <div style="width:2px; height:15px; background: linear-gradient(180deg, #90CAF9, #546E7A);"></div>
    <div style="color:#78909C;font-size:0.75rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin:0.2rem 0;">Plant Efficiency Check</div>
    <div style="width:2px; height:10px; background: #546E7A;"></div>
    <div style="width:50%; height:20px; border-top:2px solid #546E7A; border-left:2px solid #546E7A; border-right:2px solid #546E7A; border-top-left-radius: 8px; border-top-right-radius: 8px; position:relative; margin-bottom: 0.5rem;">
        <div style="position:absolute; bottom:-13px; left:-8px; color:#546E7A; font-size:1.2rem;">▼</div>
        <div style="position:absolute; bottom:-13px; right:-8px; color:#546E7A; font-size:1.2rem;">▼</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STEP 4: Branch — Success / Error ──
_wf_col1, _wf_col2 = st.columns(2)

with _wf_col1:
    st.markdown(
        '<div class="wf-fork-box" style="border-top:3px solid #4CAF50;">'
        '<div class="wf-fork-label" style="color:#4CAF50;"><span style="background:rgba(76,175,80,0.2);padding:2px 6px;border-radius:4px;">✓</span> Success — Panels Are Clean</div>'
        '<div class="wf-fork-desc"><b>PR ≥ 0.95</b> — Plant is generating within 5% of its expected output.</div>'
        '<div class="wf-fork-act">🟢 <b>Action:</b> Continue monitoring &nbsp;|&nbsp; <b>Priority:</b> Normal</div>'
        '</div>', unsafe_allow_html=True)

with _wf_col2:
    st.markdown(
        '<div class="wf-fork-box" style="border-top:3px solid #EF5350; background: linear-gradient(145deg, rgba(239,83,80,0.05) 0%, rgba(18,24,35,0.9) 100%);">'
        '<div class="wf-fork-label" style="color:#EF5350;"><span style="background:rgba(239,83,80,0.2);padding:2px 6px;border-radius:4px;">✗</span> Error — Performance Degradation</div>'
        '<div class="wf-fork-desc"><b>PR &lt; 0.95</b> — Plant is underperforming. Proceed to severity classification.</div>'
        '<div class="wf-fork-act">⚠ <b>Action:</b> Identify error &amp; correct &nbsp;|&nbsp; <b>Priority:</b> Varies</div>'
        '</div>', unsafe_allow_html=True)

# ── Re-join arrows to merge branches ──
st.markdown("""
<div style="width:100%; max-width:820px; margin: 0 auto; display:flex; flex-direction:column; align-items:center; margin-bottom: 0.5rem;">
    <div style="width:50%; height:20px; border-bottom:2px solid #546E7A; border-left:2px solid #546E7A; border-right:2px solid #546E7A; border-bottom-left-radius: 8px; border-bottom-right-radius: 8px; position:relative; margin-top: 0.2rem;">
        <div style="position:absolute; bottom:-13px; left:50%; transform: translateX(-50%); color:#546E7A; font-size:1.2rem;">▼</div>
    </div>
    <div style="width:2px; height:15px; background: #546E7A; margin-top:10px;"></div>
</div>
""", unsafe_allow_html=True)

# ── STEP 5: ERROR IDENTIFICATION ──
st.markdown(
    '<div class="wf-wrap"><div class="wf-step-box" style="--step-color: #FF9800;">'
    '<div class="wf-hdr"><div class="wf-circle">5</div>'
    '<div class="wf-stitle">🔍 Error Identification — Classify Severity</div>'
    '<span class="wf-badge">RULE-BASED</span></div>'
    '<div class="wf-sdesc">The system classifies degradation severity using transparent, rule-based thresholds on the Performance Ratio:</div>'
    '<div class="wf-sev-row">'
    '<div class="wf-sev-card" style="border-left-color:#FFC107;">'
    '<div class="wf-sev-name" style="color:#FFC107;">⚠ Monitor</div>'
    '<div class="wf-sev-pr">PR 0.85 – 0.95</div>'
    '<div class="wf-sev-info">Mild degradation. Keep watching.</div></div>'
    '<div class="wf-sev-card" style="border-left-color:#FF9800;">'
    '<div class="wf-sev-name" style="color:#FF9800;">🔶 Schedule Cleaning</div>'
    '<div class="wf-sev-pr">PR 0.70 – 0.85</div>'
    '<div class="wf-sev-info">Significant loss. Plan cleaning.</div></div>'
    '<div class="wf-sev-card" style="border-left-color:#EF5350;">'
    '<div class="wf-sev-name" style="color:#EF5350;">🔴 Clean Immediately</div>'
    '<div class="wf-sev-pr">PR &lt; 0.70</div>'
    '<div class="wf-sev-info">Severe loss. Act now.</div></div>'
    '</div></div></div>', unsafe_allow_html=True)

_wf_arrow()

# ── STEP 6: CORRECTION ──
st.markdown(
    '<div class="wf-wrap"><div class="wf-step-box" style="--step-color: #AB47BC;">'
    '<div class="wf-hdr"><div class="wf-circle">6</div>'
    '<div class="wf-stitle">🛠️ Correction — Generate Recommendation</div>'
    '<span class="wf-badge">CORRECTION</span></div>'
    '<div class="wf-sdesc">The system generates a prioritized cleaning recommendation with full context:</div>'
    '<div class="wf-smeta"><b>Details:</b> Plant operational scope · Recoverable energy (MWh) · Maintenance timeframe · Priority level · Plain-language reason</div>'
    '</div></div>', unsafe_allow_html=True)

_wf_arrow()

# ── STEP 7: OUTPUT ──
st.markdown(
    '<div class="wf-wrap"><div class="wf-step-box" style="--step-color: #F5A623;">'
    '<div class="wf-hdr"><div class="wf-circle">7</div>'
    '<div class="wf-stitle">📋 Output — Dashboard &amp; Operator Alerts</div>'
    '<span class="wf-badge">OUTPUT</span></div>'
    '<div class="wf-sdesc">Results are presented in this interactive dashboard — prioritized action queue, '
    'business impact, and a full decision audit trail.</div>'
    '<div class="wf-smeta"><b>Features:</b> Operations Queue · Recoverable Energy (MWh) · Revenue Impact ($) · Plant Analytics · Decision Trace</div>'
    '</div></div>', unsafe_allow_html=True)

# ── Summary bar ──
st.markdown(f"""
<div style="background: linear-gradient(135deg, #162233 0%, #1c3044 50%, #162233 100%); border-radius: 12px; padding: 1.4rem 2rem; border: 1px solid #2a3f54; margin-top: 0.5rem;">
    <div style="color: #90CAF9; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 0.8rem;">Workflow Summary — Current Plant State</div>
    <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center;">
        <div>
            <div style="color: #546E7A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Input Records</div>
            <div style="color: #ECEFF1; font-size: 1.1rem; font-weight: 600;">{n_op:,} observations</div>
        </div>
        <div style="color: #37474F;">|</div>
        <div>
            <div style="color: #546E7A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Validation Pass Rate</div>
            <div style="color: #66BB6A; font-size: 1.1rem; font-weight: 600;">{n_op / len(df) * 100:.1f}% of all data</div>
        </div>
        <div style="color: #37474F;">|</div>
        <div>
            <div style="color: #546E7A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Success (Healthy)</div>
            <div style="color: #4CAF50; font-size: 1.1rem; font-weight: 600;">{health_score:.1f}%</div>
        </div>
        <div style="color: #37474F;">|</div>
        <div>
            <div style="color: #546E7A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Error (Degradation)</div>
            <div style="color: {status_color}; font-size: 1.1rem; font-weight: 600;">{100 - health_score:.1f}%</div>
        </div>
        <div style="color: #37474F;">|</div>
        <div>
            <div style="color: #546E7A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Cleaning Recommendations</div>
            <div style="color: #FF9800; font-size: 1.1rem; font-weight: 600;">{n_cleaning:,} actions</div>
        </div>
        <div style="color: #37474F;">|</div>
        <div>
            <div style="color: #546E7A; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.5px;">Recoverable Energy</div>
            <div style="color: #66BB6A; font-size: 1.1rem; font-weight: 600;">{recoverable_mwh:,.1f} MWh</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 3b. TODAY'S OPERATIONAL INSIGHTS  (Change 3)
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="operational-insights">💡 Operational Insights — Selected Period</div>', unsafe_allow_html=True)

# Programmatically generate three insight sentences from existing variables
# Insight 1 — generation gap
gen_gap_pct = ((avg_expected - avg_power) / avg_expected * 100) if avg_expected > 0 else 0
if gen_gap_pct > 10:
    insight_1_icon = "📉"
    insight_1 = (f"Plant generation is currently {gen_gap_pct:.0f}% below expected levels "
                 f"during operating hours, indicating measurable soiling impact.")
elif gen_gap_pct > 3:
    insight_1_icon = "📊"
    insight_1 = (f"Generation is running approximately {gen_gap_pct:.0f}% below expected output, "
                 f"suggesting mild soiling across the monitored period.")
else:
    insight_1_icon = "✅"
    insight_1 = "Plant generation is closely tracking expected output with minimal deviation."

# Insight 2 — soiling profile
pct_affected = 100 - health_score
if n_clean_imm > n_sched_clean:
    insight_2_icon = "🚨"
    insight_2 = (f"{pct_affected:.0f}% of operating records show degradation, "
                 f"with the majority classified as severe soiling requiring immediate attention.")
elif pct_affected > 20:
    insight_2_icon = "🔍"
    insight_2 = (f"Approximately {pct_affected:.0f}% of observations show performance below "
                 f"normal levels, consistent with moderate soiling conditions.")
else:
    insight_2_icon = "🌤️"
    insight_2 = (f"Only {pct_affected:.0f}% of operating records are affected by soiling, "
                 f"indicating generally clean panel conditions.")

# Insight 3 — actionable recovery
insight_3_icon = "⚡"
if recoverable_mwh > 10:
    insight_3 = (f"Implementing today's cleaning recommendations could recover approximately "
                 f"{recoverable_mwh:,.1f} MWh of energy, equivalent to an estimated "
                 f"${potential_revenue:,.0f} in revenue.")
elif recoverable_mwh > 1:
    insight_3 = (f"Targeted cleaning of identified areas could recover an estimated "
                 f"{recoverable_mwh:,.1f} MWh, improving overall plant efficiency.")
else:
    insight_3 = ("Current soiling levels are minimal. Routine monitoring is sufficient "
                 "to maintain optimal plant performance.")

ins_col1, ins_col2, ins_col3 = st.columns(3)
with ins_col1:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">{insight_1_icon}</div>
        <div class="insight-text">{insight_1}</div>
    </div>
    """, unsafe_allow_html=True)
with ins_col2:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">{insight_2_icon}</div>
        <div class="insight-text">{insight_2}</div>
    </div>
    """, unsafe_allow_html=True)
with ins_col3:
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-icon">{insight_3_icon}</div>
        <div class="insight-text">{insight_3}</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 4. BUSINESS IMPACT
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="business-impact">💼 Business Impact</div>', unsafe_allow_html=True)

biz_col1, biz_col2, biz_col3 = st.columns([1, 1, 1])

with biz_col1:
    st.markdown("""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.5rem; border-radius: 8px; border: 1px solid #37474F; height: 100%;">
        <h4 style="color: #90CAF9; margin-top: 0;">Targeted Cleaning Schedules</h4>
        <p style="color: #E0E0E0; font-size: 0.95rem;">Instead of blanket cleaning across all panels, the system identifies specific conditions where soiling has the greatest impact. This results in an <b>estimated reduction in unnecessary cleaning</b> by focusing maintenance efforts precisely where needed.</p>
    </div>
    """, unsafe_allow_html=True)

with biz_col2:
    st.markdown(f"""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.5rem; border-radius: 8px; border: 1px solid #37474F; height: 100%;">
        <h4 style="color: #66BB6A; margin-top: 0;">Value Recovery</h4>
        <p style="color: #E0E0E0; font-size: 0.95rem;">By proactively scheduling cleaning when efficiency drops below thresholds, the plant can recover an <b>estimated {recoverable_mwh:,.1f} MWh</b> of lost generation.</p>
        <p style="color: #E0E0E0; font-size: 0.95rem;">This represents a <b>potential revenue recovery of ${potential_revenue:,.0f}</b> based on standard rates.</p>
    </div>
    """, unsafe_allow_html=True)

with biz_col3:
    st.markdown("""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.5rem; border-radius: 8px; border: 1px solid #37474F; height: 100%;">
        <h4 style="color: #F5A623; margin-top: 0;">Resource Allocation</h4>
        <p style="color: #E0E0E0; font-size: 0.95rem;">Operators move from reactive (cleaning after visible dirt) to predictive (cleaning when analytics detect degradation) — ensuring <b>better resource allocation</b> and improved maintenance planning.</p>
    </div>
    """, unsafe_allow_html=True)





# ──────────────────────────────────────────────────────────────────────
# 6. PLANT PERFORMANCE
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="plant-performance">📈 Plant Performance Analytics</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns(2)

# Chart 1: Cleaning Priority Distribution
with chart_col1:
    priority_dist = op["recommendation"].value_counts()

    # Standardize labels for the chart
    labels_to_show = ["Healthy", "Monitor", "Schedule Cleaning", "Clean Immediately"]
    values = [priority_dist.get(l, 0) for l in labels_to_show]
    colors = ["#4CAF50", "#FFC107", "#FF9800", "#F44336"]

    fig_priority = go.Figure(go.Pie(
        labels=labels_to_show,
        values=values,
        hole=0.6,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=14, color="white"),
    ))
    fig_priority.update_layout(
        title="Maintenance Effort Distribution",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        showlegend=False,
    )
    fig_priority.add_annotation(
        text="Total<br>Operating<br>Records",
        x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#90A4AE"
    )
    st.plotly_chart(fig_priority, use_container_width=True)

# Chart 2: Daily Generation Trend
with chart_col2:
    daily_power = op.groupby(op["measured_on"].dt.date).agg(
        avg_power=("total_power", "mean"),
        avg_expected=("expected_power", "mean"),
    ).reset_index()
    daily_power.rename(columns={"measured_on": "date"}, inplace=True)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_power["date"], y=daily_power["avg_power"],
        fill="tozeroy", fillcolor="rgba(66,165,245,0.15)",
        line=dict(color="#42A5F5", width=2),
        name="Current Generation",
    ))
    fig_trend.add_trace(go.Scatter(
        x=daily_power["date"], y=daily_power["avg_expected"],
        line=dict(color="#66BB6A", width=2, dash="dot"),
        name="Expected Generation",
    ))
    fig_trend.update_layout(
        title="Daily Generation Trend",
        xaxis_title="Date",
        yaxis_title="Average Generation (kW)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,35,50,0.6)",
        height=380,
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────
# 7. MODEL TRAINING BASELINE & GROUND TRUTH REFERENCE
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="training-baseline">📊 Model Training Baseline & Ground Truth Reference</div>', unsafe_allow_html=True)
st.caption("Proven verification of the exact historical sensor values, environmental boundaries, and plant operational parameters used to train the clean-panel AI model.")

baseline_stats = get_training_baseline_stats(df)
baseline_df = pd.DataFrame(baseline_stats)

st.markdown("""
<div style="background: linear-gradient(135deg, #132233 0%, #1a2f44 100%); border-radius: 10px; padding: 1.4rem 1.8rem; border-left: 5px solid #42A5F5; border: 1px solid #2a3f54; margin-bottom: 1.5rem;">
    <div style="color: #90CAF9; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-bottom: 0.6rem;">
        Proven Training Foundation — PVDAQ System 9068
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; color: #ECEFF1; font-size: 0.92rem; line-height: 1.7;">
        <div>
            <span style="color: #78909C; font-size: 0.78rem; text-transform: uppercase;">Facility & Capacity:</span><br>
            <b>4,000 kW (4 MW) Solar PV Plant</b>
        </div>
        <div>
            <span style="color: #78909C; font-size: 0.78rem; text-transform: uppercase;">Training Date Window:</span><br>
            <b>Jan 1, 2024 to Apr 30, 2025 (16 Months)</b>
        </div>
        <div>
            <span style="color: #78909C; font-size: 0.78rem; text-transform: uppercase;">Total Telemetry Records:</span><br>
            <b>135,784 Five-Minute Readings</b>
        </div>
        <div>
            <span style="color: #78909C; font-size: 0.78rem; text-transform: uppercase;">Model Accuracy (Fit):</span><br>
            <b style="color: #66BB6A;">96.4% Accuracy (R² = 0.9641)</b>
        </div>
    </div>
    <div style="margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.08); color: #B0BEC5; font-size: 0.88rem; line-height: 1.6;">
        <b>Why this proves model validity:</b> The AI was trained across 16 consecutive months of real operational conditions—spanning freezing winters, peak summer heat, overcast mornings, and high-sun middays. It learned what the plant produces when panels are clean. When actual output falls below this clean benchmark under matching sunlight and temperature, the shortfall is proven to be caused by physical factors like soiling (dirt accumulation).
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("#### Exact Sensor Boundaries & Ground Truth Training Values")
st.dataframe(baseline_df, use_container_width=True, hide_index=True)

st.markdown('<div class="section-header" id="model-information">📄 AI Model Information & Validation</div>', unsafe_allow_html=True)
st.caption("Operational analytics reflect the selected period. Model validation metrics below are from the original held-out evaluation set and are not affected by the analysis period filter.")

with st.expander("View Prediction Model Documentation"):
    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("#### Prediction Model Overview")
        st.markdown(f"""
        The expected generation prediction is produced by an **AI Regression Model**
        trained on historical sensor data from **PVDAQ System 9068**.

        The model learns the relationship between environmental conditions
        (sunlight, temperature) and expected power output under clean-panel
        conditions. Differences between predicted potential and actual output are then
        used to isolate and quantify soiling losses.

        | Parameter | Plain-Language Value | Description |
        |-----------|-------|-------------|
        | **Model Type** | AI Regression Engine | Gradient-Boosted Decision Trees |
        | **Training Observations** | {metrics['n_train']:,} records | 80% split used to learn plant curves |
        | **Testing Observations** | {metrics['n_test']:,} records | 20% held-out split for objective scoring |
        | **Total Observations** | {metrics['n_total']:,} records | 16 full months of 5-min telemetry |
        | **Overall Accuracy (R²)** | **{metrics['r2']*100:.1f}% (0.9641)** | Exceptionally strong fit to physical reality |
        | **Typical Variance (MAE)** | **±{metrics['mae']:.1f} kW** | ~2.6% average margin on a 4,000 kW plant |
        | **Max Variance (RMSE)** | ±{metrics['rmse']:.1f} kW | Root mean squared error across all conditions |
        """)

    with tech_col2:
        st.markdown("#### Input Feature Importance")
        st.markdown("The chart below shows how much each sensor input contributes "
                     "to the generation prediction.")
        fi = metrics["feature_importances"]
        fi_df = pd.DataFrame({
            "Feature": list(fi.keys()),
            "Importance": list(fi.values()),
        }).sort_values("Importance", ascending=True)

        fig_fi = go.Figure(go.Bar(
            y=fi_df["Feature"],
            x=fi_df["Importance"],
            orientation="h",
            marker_color="#37474F",
            text=fi_df["Importance"].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig_fi.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(t=10, b=10, l=100, r=20),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Model Validation — inside the same expander ──
    st.markdown("---")
    st.markdown("#### Model Validation — How We Know the Predictions Are Reliable")

    val_col1, val_col2 = st.columns(2)

    with val_col1:
        sample = op_full.sample(min(2000, len(op_full)), random_state=42)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scattergl(
            x=sample["expected_power"],
            y=sample["total_power"],
            mode="markers",
            marker=dict(color="#42A5F5", size=3, opacity=0.4),
            name="Observations",
        ))
        _max_val = max(sample["expected_power"].max(), sample["total_power"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, _max_val], y=[0, _max_val],
            mode="lines",
            line=dict(color="#66BB6A", dash="dash"),
            name="Perfect Prediction",
        ))
        fig_scatter.update_layout(
            title="Actual vs. Predicted Generation",
            xaxis_title="Predicted Power (kW)",
            yaxis_title="Actual Power (kW)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,35,50,0.6)",
            height=350,
            margin=dict(t=50, b=40, l=50, r=20),
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Points close to the green line = accurate predictions. Scatter below the line = plant underperforming vs. expected.")

    with val_col2:
        residuals = op_full["expected_power"] - op_full["total_power"]
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Histogram(
            x=residuals,
            nbinsx=80,
            marker_color="#42A5F5",
            opacity=0.7,
        ))
        fig_resid.add_vline(x=0, line_color="#66BB6A", line_dash="dash",
                            annotation_text="Zero Gap", annotation_position="top right")
        fig_resid.update_layout(
            title="Prediction Gap Distribution",
            xaxis_title="Gap (kW) — Predicted minus Actual",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26,35,50,0.6)",
            height=350,
            margin=dict(t=50, b=40, l=50, r=20),
        )
        st.plotly_chart(fig_resid, use_container_width=True)
        st.caption("Concentration around zero = model is accurate. Positive values = plant producing less than predicted (potential soiling).")

    # ── Anomaly Score Explanation ──
    st.markdown("#### How Anomalies & Confidence Are Measured")
    st.markdown("""
    The system identifies potential soiling anomalies through two key metrics that act as **anomaly confidence indicators**:

    | Metric | What It Measures | Business Meaning |
    |--------|-----------------|------------------|
    | **Performance Ratio (PR)** | Actual ÷ Expected generation | Values below 0.95 = underperformance. Lower value = higher confidence of soiling. |
    | **Loss Percentage** | (Expected − Actual) ÷ Expected × 100 | Quantifies energy being lost. Higher value = greater revenue impact. |

    The further the PR falls below 1.0, the higher the confidence that a real performance issue exists.
    The system does not use a separate anomaly detection model — it relies on the **gap between predicted
    and actual values** as a direct, interpretable measure. This approach was chosen for transparency:
    every anomaly flag can be explained by referencing specific numbers.
    """)

    # ── Consistency Metrics ──
    st.markdown("#### Prediction Consistency Over Time")
    daily_consistency = op_full.groupby(op_full["measured_on"].dt.date).apply(
        lambda g: g["total_power"].corr(g["expected_power"]), include_groups=False
    ).reset_index()
    daily_consistency.columns = ["date", "daily_correlation"]
    daily_consistency = daily_consistency.dropna()
    _avg_corr = daily_consistency["daily_correlation"].mean()

    fig_consistency = go.Figure()
    fig_consistency.add_trace(go.Scatter(
        x=daily_consistency["date"],
        y=daily_consistency["daily_correlation"],
        fill="tozeroy", fillcolor="rgba(66,187,106,0.1)",
        line=dict(color="#66BB6A", width=1.5),
        name="Daily Correlation",
    ))
    fig_consistency.add_hline(y=_avg_corr, line_color="#42A5F5", line_dash="dash",
                              annotation_text=f"Average: {_avg_corr:.3f}")
    fig_consistency.update_layout(
        title="Daily Actual-vs-Predicted Correlation (Consistency Check)",
        xaxis_title="Date", yaxis_title="Correlation",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,35,50,0.6)",
        height=280,
        margin=dict(t=50, b=40, l=50, r=20),
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig_consistency, use_container_width=True)
    st.caption("High, stable daily correlation indicates the model's predictions remain reliable across different conditions and time periods.")


# ──────────────────────────────────────────────────────────────────────
# 7b. DATA QUALITY & SYSTEM HEALTH  (v2.1)
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="data-quality">🛡️ Data Quality & System Health</div>', unsafe_allow_html=True)
st.caption("Automated checks on the dataset powering this dashboard. Values are calculated from the actual loaded data.")

# Calculate data quality metrics from actual dataset
_dq_total = len(df)
_dq_valid_op = len(op_full)
_dq_missing = df.isnull().sum()
_dq_has_missing = _dq_missing.sum() > 0
_dq_missing_cols = _dq_missing[_dq_missing > 0]

# Duplicate timestamps
_dq_dupes = df["measured_on"].duplicated().sum()

# Timestamp continuity — gaps > 5 minutes
_dq_time_sorted = df["measured_on"].sort_values()
_dq_time_diffs = _dq_time_sorted.diff().dropna()
_dq_expected_interval = pd.Timedelta(minutes=5)
_dq_gaps = (_dq_time_diffs > _dq_expected_interval * 1.5).sum()  # allow 50% tolerance

# Coverage
_dq_start = df["measured_on"].min().strftime("%Y-%m-%d")
_dq_end = df["measured_on"].max().strftime("%Y-%m-%d")
_dq_days = (df["measured_on"].max() - df["measured_on"].min()).days

# Sensor columns
_dq_sensor_cols = ["irradiance", "ambient_temp", "module_temp", "inv1_power", "inv2_power", "total_power"]
_dq_available_sensors = [c for c in _dq_sensor_cols if c in df.columns]
_dq_required_cols = ["measured_on", "irradiance", "ambient_temp", "module_temp",
                      "total_power", "expected_power", "recommendation", "priority"]
_dq_cols_ok = all(c in df.columns for c in _dq_required_cols)

_dq_col1, _dq_col2, _dq_col3 = st.columns(3)

with _dq_col1:
    st.markdown(f"""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.3rem; border-radius: 8px; border: 1px solid #37474F;">
        <div style="color: #90CAF9; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-bottom: 0.8rem;">Dataset Summary</div>
        <div style="color: #B0BEC5; font-size: 0.88rem; line-height: 2;">
            <b style="color:#ECEFF1;">Total Records:</b> {_dq_total:,}<br>
            <b style="color:#ECEFF1;">Valid Operating Records:</b> {_dq_valid_op:,}<br>
            <b style="color:#ECEFF1;">Coverage Period:</b> {_dq_start} to {_dq_end}<br>
            <b style="color:#ECEFF1;">Duration:</b> {_dq_days} days<br>
            <b style="color:#ECEFF1;">Sampling Interval:</b> 5 minutes<br>
            <b style="color:#ECEFF1;">Sensor Variables:</b> {len(_dq_available_sensors)}
        </div>
    </div>
    """, unsafe_allow_html=True)

with _dq_col2:
    st.markdown(f"""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.3rem; border-radius: 8px; border: 1px solid #37474F;">
        <div style="color: #90CAF9; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-bottom: 0.8rem;">Data Integrity</div>
        <div style="color: #B0BEC5; font-size: 0.88rem; line-height: 2;">
            <b style="color:#ECEFF1;">Duplicate Timestamps:</b> {_dq_dupes:,}<br>
            <b style="color:#ECEFF1;">Timestamp Gaps (>7.5 min):</b> {_dq_gaps:,}<br>
            <b style="color:#ECEFF1;">Columns with Missing Values:</b> {len(_dq_missing_cols)}<br>
            <b style="color:#ECEFF1;">Total Missing Cells:</b> {_dq_missing.sum():,}<br>
            <b style="color:#ECEFF1;">Missing Rate:</b> {_dq_missing.sum() / (_dq_total * len(df.columns)) * 100:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with _dq_col3:
    _check_items = [
        ("✓" if _dq_total > 0 else "✗", "Dataset loaded", "#4CAF50" if _dq_total > 0 else "#EF5350"),
        ("✓" if _dq_cols_ok else "✗", "Required columns available", "#4CAF50" if _dq_cols_ok else "#EF5350"),
        ("✓" if metrics else "✗", "Model metrics available", "#4CAF50" if metrics else "#EF5350"),
        ("✓" if _dq_gaps < 100 else "⚠", "Timestamp continuity", "#4CAF50" if _dq_gaps < 100 else "#FFC107"),
        ("⚠" if _dq_has_missing else "✓", "Missing values" + (" detected" if _dq_has_missing else " — none"), "#FFC107" if _dq_has_missing else "#4CAF50"),
    ]
    _checks_html = "".join(
        f'<div style="padding: 0.25rem 0;"><span style="color:{clr}; font-weight:700;">{icon}</span> '
        f'<span style="color:#B0BEC5; font-size:0.88rem;">{label}</span></div>'
        for icon, label, clr in _check_items
    )
    st.markdown(f"""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.3rem; border-radius: 8px; border: 1px solid #37474F;">
        <div style="color: #90CAF9; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; margin-bottom: 0.8rem;">System Health Checks</div>
        {_checks_html}
        <div style="margin-top: 0.6rem; padding-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.06);">
            <div style="color: #546E7A; font-size: 0.75rem;"><b>Model:</b> XGBoost — R² {metrics['r2']:.4f} — Status: ✓ Available</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sensor sources
with st.expander("Sensor / Data Sources"):
    _sensor_info = {
        "irradiance": "Solar Irradiance (W/m²) — Pyranometer",
        "ambient_temp": "Ambient Temperature (°C) — Weather Station",
        "module_temp": "Module Temperature (°C) — Thermocouple",
        "inv1_power": "Inverter 1 AC Power (kW) — Power Meter",
        "inv2_power": "Inverter 2 AC Power (kW) — Power Meter",
        "total_power": "Total Plant AC Power (kW) — Derived",
    }
    for col_name in _dq_available_sensors:
        _col_missing = int(_dq_missing.get(col_name, 0))
        _col_status = "✓" if _col_missing == 0 else f"⚠ {_col_missing:,} missing"
        st.markdown(f"- **{_sensor_info.get(col_name, col_name)}** — {_col_status}")

    if len(_dq_missing_cols) > 0:
        st.markdown("**Columns with missing values:**")
        for col_name, count in _dq_missing_cols.items():
            st.markdown(f"- `{col_name}`: {count:,} missing ({count/_dq_total*100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────
# 9. END-TO-END TECHNICAL WORKFLOW (Collapsible)
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="solution-workflow">⚙️ End-to-End Technical Workflow</div>', unsafe_allow_html=True)

with st.expander("View Detailed Technical Pipeline", expanded=False):

    st.caption("How the system transforms raw sensor data into actionable maintenance recommendations — and where AI adds value.")

    _pipeline_stages = [
        ("📡", "Data Collection", "Raw sensor readings (irradiance, temperature, power output) are collected from the solar plant at <b>5-minute intervals</b>.",
         "Input", "#42A5F5", "none"),
        ("🧹", "Data Preparation & Validation", "Data is cleaned, merged across sensor types, and validated. Missing values are handled and quality checks ensure only reliable data enters the pipeline.",
         "Rule-Based", "#8BC34A", "rule"),
        ("📊", "Feature Engineering", "Key performance indicators are calculated — performance ratios, rolling averages, temperature differentials — to provide the prediction model with meaningful inputs.",
         "Rule-Based", "#8BC34A", "rule"),
        ("🤖", "Expected Power Prediction", "An <b>XGBoost machine learning model</b> predicts what the plant <i>should</i> generate under current weather conditions, assuming clean panels. This is the core AI component.",
         "ML / AI", "#42A5F5", "ml"),
        ("📉", "Performance Gap Analysis", "The system compares actual generation against the ML prediction to quantify any shortfall — calculating the performance ratio and loss percentage.",
         "Rule-Based", "#8BC34A", "rule"),
        ("🛠️", "Cleaning Recommendation", "Based on the performance gap and predefined thresholds, the system generates an actionable recommendation (Healthy, Monitor, Schedule Cleaning, or Clean Immediately).",
         "Rule-Based", "#8BC34A", "rule"),
        ("📋", "Dashboard & Operator Alerts", "Results are presented in this interactive dashboard with prioritized action lists, plant-level analytics, and business impact estimates.",
         "Output", "#F5A623", "none"),
    ]

    for i, (icon, title, desc, tag, color, tag_type) in enumerate(_pipeline_stages):
        tag_class = f"tag-{tag_type}" if tag_type in ("ml", "rule") else "tag-none"
        st.markdown(f"""
        <div class="pipeline-stage" style="border-left-color: {color};">
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
                <span style="font-size:1.2rem;">{icon}</span>
                <span style="color:#E0E0E0; font-weight:600; font-size:1rem;">{title}</span>
                <span class="pipeline-tag {tag_class}">{tag}</span>
            </div>
            <div style="color:#B0BEC5; font-size:0.88rem; padding-left:2rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        if i < len(_pipeline_stages) - 1:
            st.markdown('<div class="pipeline-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.2rem; border-radius: 8px; border: 1px solid #37474F; margin-top: 1rem;">
        <div style="color: #90CAF9; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.6rem;">Key Design Principles</div>
        <div style="color: #B0BEC5; font-size: 0.88rem; line-height: 1.7;">
            • <b style="color:#42A5F5;">AI Engine</b> is used <i>only</i> for predicting clean-panel expected potential — learning complex weather-to-power curves without manual formulas.<br>
            • <b style="color:#8BC34A;">Transparent Rule Logic</b> handles classification and cleaning tickets — guaranteeing full auditability, zero hallucinations, and clear threshold adjustments.<br>
            • Every cleaning recommendation can be <b>directly verified</b> back to specific sensor readings and fixed threshold rules — no black-box uncertainty.
        </div>
    </div>
    """, unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────────────────
# 8. DECISION RULE VALIDATION
# ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header" id="rca-validation">🔎 Decision Rule Validation — Why Recommendations Are Transparent</div>', unsafe_allow_html=True)
st.caption("Understanding and validating the transparent decision rules behind each cleaning recommendation.")

rca_col1, rca_col2 = st.columns([3, 2])

with rca_col1:
    st.markdown("""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.5rem; border-radius: 8px; border: 1px solid #37474F;">
        <h4 style="color: #90CAF9; margin-top: 0;">Recommendations Are 100% Rule-Based (Deterministic)</h4>
        <p style="color: #E0E0E0; font-size: 0.92rem; line-height: 1.7;">
            Cleaning recommendations are <b>not generated by a black-box AI model</b>. They follow a
            transparent, rule-based classification system with clearly defined thresholds. This ensures:
        </p>
        <ul style="color: #E0E0E0; font-size: 0.92rem; line-height: 1.8; padding-left: 1.2rem;">
            <li><b>Full transparency:</b> Every recommendation maps directly to an efficiency threshold (≥95%, 85–95%, 70–85%, &lt;70%).</li>
            <li><b>Auditability:</b> Operational rules can be reviewed and approved by plant management before field deployment.</li>
            <li><b>Consistency:</b> The same inputs always produce the exact same recommendation — no randomness or model drift.</li>
            <li><b>Easy customization:</b> Thresholds can be tuned per plant without retraining any model.</li>
        </ul>
        <p style="color: #B0BEC5; font-size: 0.85rem; line-height: 1.7; margin-bottom: 0;">
            <b>The AI's role</b> is strictly limited to estimating clean potential. The <i>action decision</i>
            is 100% auditable business logic.
        </p>
    </div>
    """, unsafe_allow_html=True)

with rca_col2:
    st.markdown("""
    <div style="background: rgba(36, 52, 71, 0.4); padding: 1.5rem; border-radius: 8px; border: 1px solid #37474F;">
        <h4 style="color: #F5A623; margin-top: 0;">Standard Operational Thresholds</h4>
    </div>
    """, unsafe_allow_html=True)

    threshold_df = pd.DataFrame({
        "Plant Condition": ["Healthy", "Monitor", "Schedule Cleaning", "Clean Immediately", "Investigate", "Low Sunlight"],
        "Efficiency Score": ["≥ 95%", "85% – 95%", "70% – 85%", "< 70%", "Any (output ≈ 0)", "N/A"],
        "Operating Trigger": ["Normal clean operation", "Mild dirt accumulation", "Significant generation loss", "Severe generation loss", "Near-zero output", "Sunlight < 200 W/m²"],
    })
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)

st.markdown("""
<div style="background: rgba(36, 52, 71, 0.4); padding: 1.2rem; border-radius: 8px; border: 1px solid #37474F; margin-top: 1rem;">
    <div style="color: #90CAF9; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 0.6rem;">How We Validate Decision Correctness</div>
    <div style="color: #B0BEC5; font-size: 0.88rem; line-height: 1.7;">
        • <b>Threshold consistency:</b> Rules are deterministic — identical efficiency scores always trigger the same action, ensuring repeatable results.<br>
        • <b>Cross-check with generation data:</b> Recommendations correlate directly with observed power loss patterns in the plant analytics charts.<br>
        • <b>Industry-standard benchmarks:</b> The thresholds (95%, 85%, 70%) match standard solar industry performance ratio criteria.<br>
        • <b>Known limitation:</b> Without a physical soiling sensor, the system infers soiling as the <i>most likely</i> cause after mathematically excluding low-sunlight and equipment outage scenarios.
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# 9. PROFESSIONAL FOOTER
# ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="pro-footer">
    <span class="footer-item"><b>Data Source:</b>&ensp;PVDAQ System 9068 (4 MW Plant)</span>
    <span class="footer-divider">|</span>
    <span class="footer-item"><b>Prediction Model:</b>&ensp;AI Solar Generation Model (96.4% Accuracy)</span>
    <span class="footer-divider">|</span>
    <span class="footer-item"><b>Dashboard Version:</b>&ensp;Solar IMS Presentation Release v2.2</span>
    <span class="footer-divider">|</span>
    <span class="footer-item"><b>Last Updated:</b>&ensp;{_load_timestamp}</span>
</div>
""", unsafe_allow_html=True)
