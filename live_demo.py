"""
live_demo.py
============
Solar IMS — Standalone Interactive Soiling Intelligence & Cleaning Simulator
Standalone Streamlit entry point.

Opens directly into the interactive simulator without requiring navigation
through the full analytical dashboard. Reuses the unified simulation and
recommendation engine.

Usage:
    streamlit run live_demo.py
"""

import streamlit as st
from pathlib import Path
from simulator_ui import render_simulator_interface
from data_loader import load_processed_data, load_model_metrics

_BASE_DIR = Path(__file__).parent

# Page Configuration
st.set_page_config(
    page_title="Solar IMS — Live Soiling Simulator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional pre-load of historical data (cached) for historical replay tab
@st.cache_data
def get_historical_cache():
    try:
        df = load_processed_data(_BASE_DIR / "processed_output.csv")
        metrics = load_model_metrics(_BASE_DIR / "model_metrics.json")
        return df, metrics
    except Exception:
        return None, None

df_hist, metrics_hist = get_historical_cache()

# Sidebar: Standalone Controls, Info & Links
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.2rem 0 0.8rem 0; border-bottom: 1px solid #263238; margin-bottom: 0.8rem;">
        <div style="font-size: 1.35rem; font-weight: 700; color: #F5A623; letter-spacing: 0.5px; display: flex; align-items: center; gap: 0.5rem;">
            <span>☀️</span> SOLAR IMS
        </div>
        <div style="font-size: 0.85rem; color: #90CAF9; font-weight: 600; margin-top: 2px;">
            Standalone Live Simulator
        </div>
        <div style="font-size: 0.72rem; color: #78909C; margin-top: 3px;">
            Interactive Soiling Intelligence & Recommendation Prototype
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📌 About Standalone Mode")
    st.markdown("""
    This application provides direct, zero-friction access to the **Solar IMS Interactive Simulation Engine**.

    **Operational Workflow:**
    `MONITOR` → `DETECT` → `DIAGNOSE` → `RECOMMEND` → `INTERVENE` → `EVALUATE`
    """)

    st.markdown("---")
    st.markdown("### 🔗 System Navigation")
    st.info(
        "To explore the complete 16-month PVDAQ historical dataset, XGBoost model training ground truth, "
        "and plant generation analytics, run the full dashboard:\n\n"
        "`streamlit run app.py`"
    )

    st.markdown("---")
    st.markdown("""
    <div style="color: #546E7A; font-size: 0.72rem; line-height: 1.5;">
        <b>Facility Reference:</b> PVDAQ Plant 9068<br>
        <b>Rated Capacity:</b> 4,000 kW (4 MW)<br>
        <b>Data Mode:</b> Synthetic Physics & Diagnostics<br>
        <b>Currency Standard:</b> INR (₹)
    </div>
    """, unsafe_allow_html=True)

# Main Content: Render the unified simulator experience
render_simulator_interface(is_standalone=True, raw_df=df_hist, metrics=metrics_hist)
