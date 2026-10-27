# Solar IMS — Interactive Soiling Intelligence & Cleaning Recommendation Simulator

An industrial-grade, interactive solar operations prototype designed to monitor, detect, diagnose, recommend, intervene, and evaluate solar panel soiling degradation on utility-scale photovoltaic installations.

Built upon 16 consecutive months of 5-minute SCADA telemetry (**135,784 observations**) from **NREL PVDAQ System 9068** (a 4,000 kW / 4 MW utility-scale solar PV facility) and powered by an **XGBoost clean-panel benchmark model (R² = 0.9641, MAE = 103.58 kW)**.

---

## 🧭 Executive Overview & Target Audience

Solar photovoltaic (PV) facilities face significant energy generation shortfalls due to soiling (dust, dirt, and particulate accumulation). Conventional maintenance relies on fixed schedules or visual inspection—often leading to either unnecessary cleaning costs or prolonged uncaptured revenue loss.

**Solar IMS** provides solar asset managers and supervisors with an intelligent decision-support system executing the complete operational workflow:

$$\text{MONITOR} \longrightarrow \text{DETECT} \longrightarrow \text{DIAGNOSE} \longrightarrow \text{RECOMMEND} \longrightarrow \text{INTERVENE} \longrightarrow \text{EVALUATE}$$

The platform offers two entry points sharing the same simulation and recommendation logic:
1. **Full Solar IMS Dashboard (`app.py`)**: The complete operational suite containing all plant analytics, queue management, model explainability traces, sensor health audits, and an integrated simulation centre.
2. **Standalone Live Simulator (`live_demo.py`)**: A focused, zero-friction entry point opening directly into the interactive simulator—optimized for executive reviews and shareable URL deployment.

---

## 🏛️ System Architecture & Codebase Organization

The codebase is organized into modular components separating user interface presentation from core simulation, diagnostic, economic, and data access logic:

```
Solar IMS Project/
│
├── app.py                          # Full Solar IMS Dashboard (Analytical Suite + Integrated Simulator)
├── live_demo.py                    # Standalone Simulator Entry Point (Direct presentation mode)
├── simulator_ui.py                 # Shared, reusable Streamlit UI component for simulator experience
├── simulation_engine.py            # Simulation state, clock progression, physical formulas, fault injection
├── diagnostic_engine.py            # Transparent rule-based diagnostic engine & suspicion scoring
├── recommendation_engine.py        # Economic recovery calculations & explainable cleaning recommendations
├── data_loader.py                  # Validated CSV/JSON loading, caching, integrity checks, and baselines
│
├── processed_output.csv            # 135,784-row preprocessed telemetry dataset with XGBoost predictions
├── model_metrics.json              # Trained XGBoost model accuracy metrics (R², MAE, RMSE, importances)
├── prepare_data.py                 # Headless data preparation and model retraining pipeline
├── requirements.txt                # Python package dependencies
├── README.md                       # Comprehensive operational & technical documentation
│
├── .streamlit/
│   └── config.toml                 # Industrial dark theme configuration (#0f1923, #F5A623)
│
└── tests/                          # Automated test suite (35 automated tests, 100% pass rate)
    ├── run_tests.py                # Standalone test runner with detailed reporting
    ├── test_simulation.py          # Physics consistency, clock, reset, interventions, accumulation
    ├── test_diagnostics.py         # Diagnostic classifications, suspicion score bounds, low sun
    ├── test_recommendations.py     # Economic math, payback period, equipment fault overrides
    ├── test_data_loader.py         # File validation, missing column checks, baseline statistics
    └── test_integration.py         # End-to-end MONITOR → EVALUATE lifecycle verification
```

---

## 🎮 Interactive Simulator Modules (Modules A – J)

Both `app.py` and `live_demo.py` provide full access to the following 10 simulator modules:

### Module A — Live Simulation Control Centre
- **Playback Controls**: Start simulation, Pause simulation, Reset to baseline.
- **Clock Speed**: Toggle between `1x`, `5x`, and `10x` simulated time progression.
- **Time Navigation**: Step forward (`+5m`), step backward (`-5m`), or scrub smoothly across time using the 5-minute interval timeline slider.
- **Preset Scenarios**: Quick-loading presets including *Clean Panels – Sunny*, *Moderate Soiling – Sunny*, *Heavy Soiling – Hot Day*, *Cloudy Day – Light Soiling*, *Partial Shading*, *Inverter Issue + Soiling*, *Rain Cleaning*, and *Sensor Malfunction*.
- **Automatic Refresh Engine**: Utilizes `@st.fragment(run_every=2.0)` to auto-advance telemetry in real-time when running, without freezing the browser or causing full-page flicker.

### Module B — Synthetic Solar Operating Conditions
Manipulate environmental parameters in real time:
- **Solar Irradiance**: Slider from 0 to 1,200 W/m².
- **Ambient Air Temperature**: Slider from -10°C to 50°C.
- **Panel Module Temperature**: Dynamically modeled via NOCT heat-transfer offset plus direct controls.
- **Soiling Severity ($L_{\text{soiling}}$)**: Slider from 0% (Clean) to 40% (Heavy).
- **Accumulation Rate**: Slider from -0.5% (natural cleaning) to +1.0% per 5-minute step.
- **Weather Conditions**: *Sunny*, *Cloudy*, *Variable*, and *Rainy* (with natural rain washing dynamics).
- **Partial Shading Severity**: Slider from 0% to 80% array shading.
- **Equipment Loss Factor**: Slider from 0% to 80% electrical transmission/inverter loss.

### Module C — Soiling & Power-Loss Simulator
- **Clean-Reference Baseline ($P_{\text{clean}}$)**: Evaluates what clean panels would produce under matching irradiance and temperature.
- **Simulated Actual Output ($P_{\text{soiled}}$)**: Evaluates electrical output after soiling, temperature derate, shading, and inverter losses.
- **Dynamic Plotly Visualizations**:
  - *Telemetry Stream Curve*: Displays clean potential vs. actual generation, highlighting the uncaptured power gap with subtle red shading and diamond position indicators.
  - *Decomposed Loss Breakdown*: Real-time stacked area chart isolating losses attributable to soiling, elevated panel heat, and equipment/faults.

### Module D — Fault Injection & Transparent Diagnostics
- **Interactive Fault Matrix**: Inject one or multiple faults simultaneously (*Normal Operation*, *Gradual Soiling*, *Sudden Dust Event*, *Cloud Cover*, *Elevated Panel Temperature*, *Partial Shading*, *Inverter/Connection Loss*, *Sensor Anomaly/Missing Data*).
- **Transparent Rule Logic**: Compares actual vs. clean potential and analyzes weather consistency.
- **Rule-Based Suspicion Score**: Bounded 0–100 scale indicating diagnostic evidence strength (clearly distinguished from calibrated ML probabilities).
- **Full Evidence Audit**: Explains the exact telemetry facts supporting the primary diagnosis and ranks alternative hypotheses.
- **Limitations Disclosure**: Prominently warns operators when conditions (e.g., dawn/dusk, cloud cover) limit diagnostic confidence.

### Module E — Cleaning Recommendation & Economic Engine
Evaluates whether a physical washing intervention is economically justified:
- **Configurable Financial Parameters**:
  - Cleaning Cost (Default: ₹15,000 INR per array wash).
  - Electricity Tariff (Default: ₹5.50 INR / kWh).
  - Evaluation Horizon (7, 15, 30, or 60 days).
  - Cleaning Washing Effectiveness (Default: 85% particulate removal).
  - Daily Sunshine Hours & Rain Probability.
- **Economic Metrics**:
  - Recoverable Power (kW instantaneous).
  - Recoverable Energy (kWh / MWh over horizon).
  - Gross Recoverable Energy Value (₹ INR).
  - Net Economic Benefit ($\text{Value} - \text{Cost}$).
  - Benefit-Cost Ratio / ROI Multiplier ($\text{Value} / \text{Cost}$).
  - Payback Horizon (Hours/Days of operation to break even).
- **Explainable Recommendation**: Plain-language justification for why cleaning is recommended, delayed, or discouraged.
- **Fault Override Protection**: Prevents inappropriate cleaning tickets if the primary cause is weather or inverter outage!

### Module F — Simulated Interventions
- **Simulate Cleaning**: Reduces soiling according to configured effectiveness, recalculates plant output, updates the telemetry stream, and records pre/post intervention states.
- **Before-and-After Comparison Card**: Displays side-by-side metrics showing pre-clean vs. post-clean soiling, output gain (kW), energy recovery (MWh), and recommendation shift.
- **Delay Cleaning**: Logs the deferral decision and simulates further dust accumulation.
- **Investigate Fault**: Dispatches a simulated inspection ticket for inverters/sensors without altering panel surface dirt.

### Module G — Historical Replay vs. Synthetic Mode
- **Synthetic Simulation Mode**: Physics-based mathematical models and fault injection for interactive experimentation.
- **Historical Sensor Replay Mode**: Replays ground-truth 5-minute sensor readings from PVDAQ Plant 9068 across authentic operating days (*Severe Soiling Day*, *Moderate Soiling Day*, *Clean Day*).
- **Persistent Data-Source Indicator**: Ensures users always know whether they are looking at physical sensor records or synthetic simulation results.

### Module H — Scenario Comparison Laboratory
- Compare 3 operational scenarios side by side (e.g., *Clean Sunny* vs. *Moderate Soiling Sunny* vs. *Heavy Soiling Hot Day*).
- Comprehensive comparison table: Irradiance, Temperature, Soiling Severity, Clean Power, Measured Power, Generation Shortfall, Diagnostic Classification, Recommendation, and Net Benefit.
- Side-by-side Plotly charts for power outputs and net economic return.

### Module I — Simulation Event Log & Audit Export
- Chronological session log capturing: `simulation_started`, `simulation_paused`, `scenario_changed`, `fault_injected`, `threshold_crossed`, `cleaning_simulated`, `cleaning_delayed`, `simulation_reset`.
- Filter log by Event Type and Severity Level (*INFO*, *ACTION*, *WARNING*, *CRITICAL*).
- **One-Click CSV Export**: Download the complete event history as a timestamped CSV file for compliance audits.

### Module J — Executive Alerts & KPI Cards
- Real-time status cards: *Clean-Panel Reference (kW)*, *Simulated Measured Power (kW)*, *Uncaptured Power Gap (kW)*, *Simulated Plant PR (%)*, *Diagnostic Status*, and *Automated Recommendation*.
- Color-coded severity states: **Normal** (Green), **Advisory** (Yellow), **Warning** (Orange), and **Critical** (Red).

---

## 📊 Complete Dashboard Sections (`app.py`)

In addition to the integrated simulator, `app.py` preserves all original analytical sections:
1. **Header & Hero Status Container**: Executive plant health score, selected period status, recoverable MWh, and primary action.
2. **Operations Queue (Interactive)**: Prioritized maintenance tickets with interactive status lifecycle (`🔴 Open` → `🟡 Acknowledged` → `🔵 In Progress` → `🟢 Resolved`).
3. **Explainable AI Decision Trace**: Step-by-step 9-stage trace demonstrating how raw sensor readings progress through operational checks, AI clean benchmark, shortfall isolation, and rule classification.
4. **Prototype Workflow — How Decisions Are Made**: Interactive business architecture flow diagram showing Inputs, AI Prediction, Validation Gates, Severity Classification, Correction, and Operator Output.
5. **Operational Insights**: Programmatically derived takeaways summarizing generation gap, soiling severity distribution, and revenue recovery opportunities.
6. **Business & Revenue Impact**: Quantified financial value of targeted cleaning schedules versus calendar-based washing.
7. **Plant Generation Analytics**: Pie chart of maintenance effort distribution and historical daily generation trends over time.
8. **Model Training Baseline Ground Truth**: Exact sensor physical boundaries and statistical baselines across all 135,784 training records.
9. **AI Model Information & Validation**: Technical documentation of the XGBoost regression model, feature importances, Actual vs. Predicted scatter plot, residual error histogram, and daily correlation consistency check.
10. **Data Quality & System Health**: Automated dataset audit checking timestamp gaps, duplicate timestamps, missing values, and sensor continuity.
11. **End-to-End Technical Workflow**: Collapsible architecture map detailing ML vs. rule-based components.
12. **Decision Rule Validation (RCA)**: Explanation of deterministic threshold criteria and operational limits.
13. **Professional Audit Footer**: Data provenance, model accuracy, and load timestamps.

---

## 🧮 Physical & Mathematical Simulation Assumptions

All synthetic calculations use clearly documented physical relationships:

1. **Clean-Reference Output ($P_{\text{clean}}$)**:
   $$P_{\text{clean}} = P_{\text{rated}} \times \left(\frac{G_{\text{effective}}}{1000}\right) \times \left[1 + \gamma \times (T_{\text{module}} - T_{\text{ref}})\right]$$
   - $P_{\text{rated}} = 4,000 \text{ kW}$ (4 MW Plant Capacity).
   - $G_{\text{effective}}$: Solar irradiance arriving at array plane after sky variability and cloud attenuation.
   - $\gamma = -0.004 \text{ / °C}$ (Standard crystalline silicon temperature derating coefficient).
   - $T_{\text{ref}} = 25 \text{ °C}$ (Standard Test Conditions reference temperature).

2. **Panel Module Temperature Model**:
   $$T_{\text{module}} = T_{\text{ambient}} + \Delta T_{\text{offset}} \times \left(\frac{G_{\text{effective}}}{800}\right)$$
   - $\Delta T_{\text{offset}} = 15 \text{ °C}$ nominal operating cell temperature offset.

3. **Soiling Power Derate ($P_{\text{soiled}}$)**:
   $$P_{\text{soiled}} = P_{\text{clean}} \times (1 - L_{\text{soiling}}) \times (1 - L_{\text{equipment}}) \times (1 - L_{\text{shading}})$$
   - $L_{\text{soiling}}$: Fractional soiling loss ($0.0 \le L_{\text{soiling}} \le 1.0$).
   - $L_{\text{equipment}}$: Fractional equipment/inverter loss.
   - $L_{\text{shading}}$: Fractional partial shading loss.

4. **Cleaning Intervention Effect**:
   $$L_{\text{soiling, post}} = L_{\text{soiling, pre}} \times (1 - \eta_{\text{clean}})$$
   - $\eta_{\text{clean}}$: Cleaning effectiveness ($0.85$ default).

5. **Economic Benefit Projection**:
   $$E_{\text{recovered}} = P_{\text{recovered}} \times H_{\text{sun}} \times (1 - p_{\text{rain}}) \times \left(\frac{H_{\text{horizon}}}{24}\right)$$
   $$\text{Net Benefit} = (E_{\text{recovered}} \times \text{Tariff}) - C_{\text{cleaning}}$$
   - $H_{\text{sun}} = 6.0 \text{ hrs/day}$ (Average daily productive peak sun hours assumption).
   - $p_{\text{rain}} = 0.10$ (Assumed rain cleaning probability).
   - Tariff defaults to ₹5.50 INR/kWh; Cleaning Cost defaults to ₹15,000 INR.

---

## ⚠️ Known Limitations & Diagnostic Disclaimers

1. **Simulation vs. Telemetry**: The interactive simulator generates synthetic observations based on documented physical approximations. It does not claim to receive real-time streaming telemetry from physical sensors.
2. **Rule-Based Diagnostic Scores**: Suspicion scores are rule-based heuristics derived from deterministic thresholds. They do **not** represent calibrated machine learning probabilities.
3. **Root-Cause Confirmation**: In a real facility, soiling cannot be definitively distinguished from string disconnection or module degradation without physical array inspection or dedicated optical soiling sensors.
4. **Economic Uncertainty**: Energy recovery and financial net benefit are scenario-based estimates subject to weather variability and actual cleaning quality.

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run the Automated Test Suite
Run the comprehensive test suite to verify physical consistency, simulation rules, diagnostics, and economic calculations:
```bash
python tests/run_tests.py
```
*(Expected output: 35 tests run, 35 passed, 0 failed).*

---

### Step 3: Run the Applications

#### Option A: Run the Standalone Interactive Simulator
Opens directly into the simulator with zero friction:
```bash
python -m streamlit run live_demo.py
```
*Access in browser at:* `http://localhost:8501`

#### Option B: Run the Full Solar IMS Dashboard
Opens the full analytical suite with all 13 sections and the integrated simulator:
```bash
python -m streamlit run app.py
```
*Access in browser at:* `http://localhost:8501`

#### Option C: Run Both Simultaneously on Separate Ports
To evaluate both applications side by side on one computer:
```bash
# Terminal 1: Full Dashboard on port 8501
python -m streamlit run app.py --server.port 8501

# Terminal 2: Standalone Simulator on port 8502
python -m streamlit run live_demo.py --server.port 8502
```

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

Both `app.py` and `live_demo.py` can be deployed independently as separate shareable web applications from the same GitHub repository:

### Deploying the Standalone Simulator (`live_demo.py`)
1. Push this repository to GitHub.
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository and branch.
5. In **Main file path**, enter: `live_demo.py`
6. In **App URL**, choose a descriptive URL (e.g., `solar-ims-live-simulator`).
7. Click **Deploy!**

### Deploying the Full Solar IMS Dashboard (`app.py`)
1. In Streamlit Community Cloud, click **New app**.
2. Select the same repository and branch.
3. In **Main file path**, enter: `app.py`
4. In **App URL**, choose a descriptive URL (e.g., `solar-ims-full-dashboard`).
5. Click **Deploy!**

Both applications share the exact same underlying simulation code, requirements, and data artifacts without any code duplication.

---

## 📋 Acceptance Criteria Verification Matrix

| # | Acceptance Criterion | Status | Verification Detail |
|---|---|:---:|---|
| 1 | Existing dashboard still works | ✅ Verified | All 13 original sections in `app.py` load and function with full fidelity. |
| 2 | Standalone `live_demo.py` exists & runs | ✅ Verified | `live_demo.py` runs independently without importing `app.py`. |
| 3 | Both entry points share identical engine | ✅ Verified | Both import `simulator_ui.py`, `simulation_engine.py`, and `recommendation_engine.py`. |
| 4 | Interactive environmental sliders | ✅ Verified | Irradiance, temperature, soiling, and weather sliders dynamically alter physics state. |
| 5 | Conditions evolve over time | ✅ Verified | `step()` and accumulation rates advance states and update generation curves. |
| 6 | Start, Pause, Reset, Scrubber controls work | ✅ Verified | Clock ticks, scrubber navigates backwards/forwards, and reset restores clean state. |
| 7 | Fault injection changes readings | ✅ Verified | Inverter loss, shading, cloud cover, and sensor faults alter outputs predictably. |
| 8 | Diagnostic panel explains conclusions | ✅ Verified | Displays primary diagnosis, suspicion score (0-100), evidence list, and limitations. |
| 9 | Recommendations respond to assumptions | ✅ Verified | Cleaning costs, tariffs, and soiling severity dynamically shift recommendation states. |
| 10 | Simulated cleaning before/after comparison | ✅ Verified | Cleans dirt, restores power, and renders detailed pre/post intervention comparison table. |
| 11 | Scenario Comparison Laboratory works | ✅ Verified | Compares 3 scenarios side by side with comparison tables and Plotly bar charts. |
| 12 | Event log records actions & exports CSV | ✅ Verified | Logs all resets, faults, interventions; filterable by type/severity with CSV export. |
| 13 | Historical vs. synthetic clearly distinguished | ✅ Verified | Visible badges and dedicated Tab 4 prevent synthetic data from being mistaken for real SCADA. |
| 14 | Automated test suite passes | ✅ Verified | 35 automated tests passing (100% pass rate) in `tests/run_tests.py`. |
| 15 | Documented for independent deployment | ✅ Verified | Step-by-step instructions for local execution and Streamlit Community Cloud deployment. |

---

*Solar IMS — Soiling Intelligence & Cleaning Recommendation System • Built with Python & Streamlit*
