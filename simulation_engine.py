"""
simulation_engine.py
====================
Core simulation engine for Solar IMS Live Demo.

Responsible for:
- Simulation state management
- Synthetic sensor readings generation
- Simulation clock and time progression
- Environmental conditions modeling
- Soiling accumulation
- Fault injection
- Simulated cleaning interventions
- Deterministic scenario generation

All synthetic data is clearly labelled and never presented as actual sensor telemetry.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy


# ──────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
RATED_CAPACITY_KW = 4000
SIMULATION_INTERVAL_MINUTES = 5

# Default environmental baselines (documented assumptions)
DEFAULT_BASELINES = {
    "irradiance_peak": 1000.0,      # W/m² typical clear-sky peak
    "ambient_temp_base": 30.0,       # °C baseline ambient
    "module_temp_offset": 15.0,      # °C above ambient under load
    "temp_coefficient": -0.004,      # fractional loss per °C above 25°C (assumption)
    "ref_temperature": 25.0,         # °C STC reference
}

# Soiling severity labels and ranges (simulation assumptions, not validated)
SOILING_LABELS = {
    (0.00, 0.02): "Clean",
    (0.02, 0.05): "Light Soiling",
    (0.05, 0.15): "Moderate Soiling",
    (0.15, 1.00): "Heavy Soiling",
}

# Weather condition definitions
WEATHER_CONDITIONS = {
    "Sunny": {"irradiance_factor": 1.0, "variability": 0.03, "rain_cleaning": False},
    "Cloudy": {"irradiance_factor": 0.4, "variability": 0.10, "rain_cleaning": False},
    "Variable": {"irradiance_factor": 0.65, "variability": 0.20, "rain_cleaning": False},
    "Rainy": {"irradiance_factor": 0.25, "variability": 0.15, "rain_cleaning": True},
}

# Fault type definitions
FAULT_TYPES = {
    "none": {
        "label": "Normal Operation",
        "power_factor": 1.0,
        "irradiance_factor": 1.0,
        "description": "No faults injected. System operating normally.",
    },
    "gradual_soiling": {
        "label": "Gradual Soiling Accumulation",
        "power_factor": 1.0,  # Applied through soiling, not power factor
        "irradiance_factor": 1.0,
        "description": "Simulates progressive dirt accumulation on panels over time.",
    },
    "sudden_soiling": {
        "label": "Sudden Soiling Event",
        "power_factor": 1.0,
        "irradiance_factor": 1.0,
        "description": "Simulates a rapid soiling event (e.g. dust storm, construction nearby).",
    },
    "cloud_cover": {
        "label": "Cloud-Induced Irradiance Drop",
        "power_factor": 1.0,
        "irradiance_factor": 0.35,
        "description": "Simulates sudden cloud cover reducing available irradiance.",
    },
    "high_temperature": {
        "label": "Elevated Panel Temperature",
        "power_factor": 1.0,
        "irradiance_factor": 1.0,
        "description": "Simulates abnormally high panel temperature causing efficiency loss.",
    },
    "partial_shading": {
        "label": "Partial Shading",
        "power_factor": 0.70,
        "irradiance_factor": 1.0,
        "description": "Simulates partial shading from nearby structures or vegetation.",
    },
    "inverter_loss": {
        "label": "Inverter / Connection Loss",
        "power_factor": 0.50,
        "irradiance_factor": 1.0,
        "description": "Simulates partial inverter failure or connection-related power loss.",
    },
    "sensor_anomaly": {
        "label": "Sensor Anomaly / Missing Data",
        "power_factor": 1.0,
        "irradiance_factor": 1.0,
        "description": "Simulates sensor malfunction producing invalid or missing readings.",
    },
}


# ──────────────────────────────────────────────────────────────────────
# SCENARIO PRESETS
# ──────────────────────────────────────────────────────────────────────
SCENARIO_PRESETS = {
    "Clean Panels – Sunny": {
        "irradiance": 850.0,
        "ambient_temp": 28.0,
        "weather": "Sunny",
        "soiling_severity": 0.01,
        "accumulation_rate": 0.0,
        "active_faults": [],
        "description": "Optimal conditions: clean panels under clear skies.",
    },
    "Moderate Soiling – Sunny": {
        "irradiance": 800.0,
        "ambient_temp": 32.0,
        "weather": "Sunny",
        "soiling_severity": 0.10,
        "accumulation_rate": 0.002,
        "active_faults": ["gradual_soiling"],
        "description": "Moderate dirt accumulation reducing output under good irradiance.",
    },
    "Heavy Soiling – Hot Day": {
        "irradiance": 950.0,
        "ambient_temp": 40.0,
        "weather": "Sunny",
        "soiling_severity": 0.22,
        "accumulation_rate": 0.003,
        "active_faults": ["gradual_soiling"],
        "description": "Heavy soiling compounded by high temperature losses.",
    },
    "Cloudy Day – Light Soiling": {
        "irradiance": 400.0,
        "ambient_temp": 22.0,
        "weather": "Cloudy",
        "soiling_severity": 0.04,
        "accumulation_rate": 0.001,
        "active_faults": ["cloud_cover"],
        "description": "Overcast conditions with minor soiling—diagnostic challenge.",
    },
    "Partial Shading Event": {
        "irradiance": 750.0,
        "ambient_temp": 30.0,
        "weather": "Sunny",
        "soiling_severity": 0.03,
        "accumulation_rate": 0.0,
        "active_faults": ["partial_shading"],
        "description": "Nearby structure casting shadow on part of the array.",
    },
    "Inverter Issue + Soiling": {
        "irradiance": 820.0,
        "ambient_temp": 33.0,
        "weather": "Sunny",
        "soiling_severity": 0.08,
        "accumulation_rate": 0.001,
        "active_faults": ["inverter_loss", "gradual_soiling"],
        "description": "Equipment loss combined with moderate soiling.",
    },
    "Rain Cleaning Effect": {
        "irradiance": 300.0,
        "ambient_temp": 20.0,
        "weather": "Rainy",
        "soiling_severity": 0.12,
        "accumulation_rate": -0.005,
        "active_faults": [],
        "description": "Rainfall naturally reducing soiling. Observe automatic cleaning.",
    },
    "Sensor Malfunction": {
        "irradiance": 700.0,
        "ambient_temp": 29.0,
        "weather": "Sunny",
        "soiling_severity": 0.05,
        "accumulation_rate": 0.0,
        "active_faults": ["sensor_anomaly"],
        "description": "Simulated sensor failure producing invalid readings.",
    },
}


# ──────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SimulationReading:
    """A single simulated observation at a point in time."""
    timestamp: datetime
    step: int
    irradiance: float              # W/m²
    ambient_temp: float            # °C
    module_temp: float             # °C
    soiling_severity: float        # 0.0 to 1.0 fractional
    weather: str
    clean_power: float             # kW — expected under these conditions without soiling
    measured_power: float          # kW — simulated actual output after losses
    performance_ratio: float       # measured / clean
    soiling_loss_pct: float        # percentage loss due to soiling
    total_loss_pct: float          # total percentage loss (soiling + faults + temp)
    active_faults: list            # list of active fault type keys
    scenario_name: str
    is_synthetic: bool = True      # Always True for simulation readings
    sensor_valid: bool = True      # False during sensor anomaly faults
    
    # Decomposed losses for transparency
    soiling_loss_kw: float = 0.0
    temperature_loss_kw: float = 0.0
    fault_loss_kw: float = 0.0
    equipment_loss_kw: float = 0.0


@dataclass
class InterventionRecord:
    """Record of a simulated intervention."""
    timestamp: datetime
    intervention_type: str         # "cleaning", "delay", "investigate"
    pre_soiling: float
    post_soiling: float
    pre_power: float
    post_power: float
    pre_clean_power: float
    cleaning_effectiveness: float
    estimated_recovery_kw: float
    notes: str = ""


@dataclass
class SimulationState:
    """Complete simulation state — managed via session state."""
    # Time
    current_step: int = 0
    start_time: datetime = field(default_factory=lambda: datetime(2025, 6, 15, 6, 0))
    current_time: datetime = field(default_factory=lambda: datetime(2025, 6, 15, 6, 0))
    
    # Configuration
    scenario_name: str = "Clean Panels – Sunny"
    speed_multiplier: int = 1
    is_running: bool = False
    mode: str = "manual"           # "manual" or "auto"
    
    # Environmental state
    irradiance: float = 850.0
    ambient_temp: float = 28.0
    weather: str = "Sunny"
    
    # Soiling state
    soiling_severity: float = 0.01
    accumulation_rate: float = 0.0  # per simulation step
    
    # Fault state
    active_faults: list = field(default_factory=list)
    
    # Additional modifiers
    shading_severity: float = 0.0
    equipment_loss_factor: float = 0.0
    
    # History
    readings: list = field(default_factory=list)
    interventions: list = field(default_factory=list)
    event_log: list = field(default_factory=list)
    
    # Random seed for reproducibility
    random_seed: int = 42


# ──────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE
# ──────────────────────────────────────────────────────────────────────
class SimulationEngine:
    """
    Core simulation engine for the Solar IMS Live Demo.
    
    All calculations use documented simplified models.
    Results are labelled as synthetic simulation outputs.
    """

    def __init__(self, state: SimulationState | None = None):
        self.state = state or SimulationState()
        self._rng = np.random.RandomState(self.state.random_seed)

    def reset(self, scenario_name: str | None = None) -> SimulationState:
        """Reset simulation to a clean baseline state."""
        seed = self.state.random_seed
        scenario = scenario_name or self.state.scenario_name
        
        self.state = SimulationState(random_seed=seed)
        self._rng = np.random.RandomState(seed)
        
        if scenario in SCENARIO_PRESETS:
            self.load_scenario(scenario)
        
        self._log_event("simulation_reset", "INFO",
                        f"Simulation reset to scenario: {self.state.scenario_name}")
        return self.state

    def load_scenario(self, name: str) -> None:
        """Load a preset scenario configuration."""
        if name not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_PRESETS.keys())}")
        
        preset = SCENARIO_PRESETS[name]
        self.state.scenario_name = name
        self.state.irradiance = preset["irradiance"]
        self.state.ambient_temp = preset["ambient_temp"]
        self.state.weather = preset["weather"]
        self.state.soiling_severity = preset["soiling_severity"]
        self.state.accumulation_rate = preset["accumulation_rate"]
        self.state.active_faults = list(preset.get("active_faults", []))
        
        # Reset simulation position but keep history
        self.state.current_step = 0
        self.state.current_time = self.state.start_time
        self.state.readings = []
        self.state.interventions = []
        
        self._log_event("scenario_changed", "INFO",
                        f"Loaded scenario: {name}")

    def step(self) -> SimulationReading:
        """
        Advance the simulation by one time step and return the new reading.
        
        This is the main simulation loop method.
        """
        # Advance time
        self.state.current_step += 1
        self.state.current_time = (
            self.state.start_time
            + timedelta(minutes=SIMULATION_INTERVAL_MINUTES * self.state.current_step)
        )
        
        # Apply soiling accumulation
        self._apply_accumulation()
        
        # Generate the reading
        reading = self._generate_reading()
        self.state.readings.append(reading)
        
        # Check thresholds and generate events
        self._check_thresholds(reading)
        
        return reading

    def step_to(self, target_step: int) -> SimulationReading | None:
        """
        Advance or rewind simulation to a specific step.
        For rewind, regenerates all readings from the start.
        """
        if target_step < 0:
            target_step = 0
        
        if target_step <= len(self.state.readings):
            # Rewind: trim readings
            self.state.readings = self.state.readings[:target_step]
            self.state.current_step = target_step
            self.state.current_time = (
                self.state.start_time
                + timedelta(minutes=SIMULATION_INTERVAL_MINUTES * target_step)
            )
            if self.state.readings:
                return self.state.readings[-1]
            return None
        else:
            # Forward: generate missing steps
            while self.state.current_step < target_step:
                reading = self.step()
            return reading

    def get_current_reading(self) -> SimulationReading | None:
        """Get the reading corresponding to the current simulation step."""
        if 0 < self.state.current_step <= len(self.state.readings):
            return self.state.readings[self.state.current_step - 1]
        elif self.state.readings:
            return self.state.readings[-1]
        return self._generate_reading_at_step(self.state.current_step)

    def simulate_cleaning(self, effectiveness: float = 0.85) -> InterventionRecord:
        """
        Simulate a cleaning intervention.
        
        Parameters
        ----------
        effectiveness : float
            Fraction of soiling removed (0 to 1). Default 0.85 (85%).
            Labelled as a simulation assumption.
        
        Returns
        -------
        InterventionRecord
            Record of the before/after state.
        """
        effectiveness = max(0.0, min(1.0, effectiveness))
        
        pre_soiling = self.state.soiling_severity
        pre_reading = self.get_current_reading()
        pre_power = pre_reading.measured_power if pre_reading else 0.0
        pre_clean_power = pre_reading.clean_power if pre_reading else 0.0
        
        # Apply cleaning
        removed = self.state.soiling_severity * effectiveness
        self.state.soiling_severity = max(0.0, self.state.soiling_severity - removed)
        
        # Generate post-cleaning reading
        post_reading = self._generate_reading()
        post_power = post_reading.measured_power
        
        # Update current reading in buffer so get_current_reading() reflects post-cleaning immediately
        if self.state.readings:
            self.state.readings[-1] = post_reading
        else:
            self.state.readings.append(post_reading)
        
        recovery = post_power - pre_power
        
        record = InterventionRecord(
            timestamp=self.state.current_time,
            intervention_type="cleaning",
            pre_soiling=pre_soiling,
            post_soiling=self.state.soiling_severity,
            pre_power=pre_power,
            post_power=post_power,
            pre_clean_power=pre_clean_power,
            cleaning_effectiveness=effectiveness,
            estimated_recovery_kw=max(0, recovery),
            notes=f"Cleaning reduced soiling from {pre_soiling:.1%} to {self.state.soiling_severity:.1%}",
        )
        self.state.interventions.append(record)
        
        self._log_event("cleaning_simulated", "ACTION",
                        f"Cleaning applied (effectiveness={effectiveness:.0%}). "
                        f"Soiling: {pre_soiling:.1%} → {self.state.soiling_severity:.1%}. "
                        f"Power recovery: +{max(0, recovery):.0f} kW")
        
        return record

    def delay_cleaning(self) -> None:
        """Record a decision to delay cleaning."""
        self._log_event("cleaning_delayed", "ACTION",
                        f"Cleaning delayed. Soiling at {self.state.soiling_severity:.1%}. "
                        f"Accumulation rate: {self.state.accumulation_rate:.4f}/step.")

    def set_environmental_conditions(self, irradiance: float | None = None,
                                      ambient_temp: float | None = None,
                                      weather: str | None = None,
                                      soiling_severity: float | None = None,
                                      accumulation_rate: float | None = None,
                                      shading: float | None = None,
                                      equipment_loss: float | None = None) -> None:
        """Update one or more environmental conditions."""
        if irradiance is not None:
            self.state.irradiance = max(0.0, irradiance)
        if ambient_temp is not None:
            self.state.ambient_temp = ambient_temp
        if weather is not None and weather in WEATHER_CONDITIONS:
            self.state.weather = weather
        if soiling_severity is not None:
            self.state.soiling_severity = max(0.0, min(1.0, soiling_severity))
        if accumulation_rate is not None:
            self.state.accumulation_rate = accumulation_rate
        if shading is not None:
            self.state.shading_severity = max(0.0, min(1.0, shading))
        if equipment_loss is not None:
            self.state.equipment_loss_factor = max(0.0, min(1.0, equipment_loss))

        # Refresh current reading in history so changes are immediately visible
        if 0 < self.state.current_step <= len(self.state.readings):
            self.state.readings[self.state.current_step - 1] = self._generate_reading_at_step(self.state.current_step)

    def set_faults(self, faults: list[str]) -> None:
        """Set the list of active faults."""
        valid = [f for f in faults if f in FAULT_TYPES]
        old_faults = list(self.state.active_faults)
        self.state.active_faults = valid
        
        added = set(valid) - set(old_faults)
        removed = set(old_faults) - set(valid)
        
        for f in added:
            self._log_event("fault_injected", "WARNING",
                            f"Fault injected: {FAULT_TYPES[f]['label']}")
        for f in removed:
            self._log_event("fault_cleared", "INFO",
                            f"Fault cleared: {FAULT_TYPES[f]['label']}")

        # Refresh current reading in history so changes are immediately visible
        if 0 < self.state.current_step <= len(self.state.readings):
            self.state.readings[self.state.current_step - 1] = self._generate_reading_at_step(self.state.current_step)

    # ── Private Methods ──────────────────────────────────────────────

    def _apply_accumulation(self) -> None:
        """Apply soiling accumulation for one time step."""
        rate = self.state.accumulation_rate
        
        # Adjust rate for weather effects
        weather_cfg = WEATHER_CONDITIONS.get(self.state.weather, WEATHER_CONDITIONS["Sunny"])
        if weather_cfg["rain_cleaning"]:
            # Rain reduces soiling
            rate = min(rate, -0.003)
        
        # Apply faults that affect accumulation
        if "sudden_soiling" in self.state.active_faults:
            rate = max(rate, 0.015)  # Rapid accumulation
        elif "gradual_soiling" in self.state.active_faults:
            rate = max(rate, 0.002)
        
        self.state.soiling_severity = max(0.0, min(1.0,
            self.state.soiling_severity + rate))

    def _generate_reading(self) -> SimulationReading:
        """Generate a reading at the current simulation state."""
        return self._generate_reading_at_step(self.state.current_step)

    def _generate_reading_at_step(self, step: int) -> SimulationReading:
        """
        Generate a consistent simulated reading for a given step.
        
        Power model (documented assumption):
            P_clean = RATED_CAPACITY × (irradiance / 1000) × temp_factor × weather_variability
            P_soiled = P_clean × (1 - soiling_severity) × fault_factors
        
        This is a simplified model. It does not account for:
        - Non-linear panel response curves
        - Spectral effects
        - Angle of incidence
        - Detailed inverter efficiency curves
        """
        # Time of day factor (simplified solar curve)
        timestamp = (self.state.start_time
                     + timedelta(minutes=SIMULATION_INTERVAL_MINUTES * step))
        hour = timestamp.hour + timestamp.minute / 60.0
        time_factor = self._solar_time_factor(hour)
        
        # Base irradiance with weather and time effects
        weather_cfg = WEATHER_CONDITIONS.get(self.state.weather, WEATHER_CONDITIONS["Sunny"])
        base_irradiance = self.state.irradiance * time_factor
        effective_irradiance = base_irradiance * weather_cfg["irradiance_factor"]
        
        # Add realistic variability (deterministic based on step)
        variability = weather_cfg["variability"]
        step_rng = np.random.RandomState((self.state.random_seed * 10007 + step) % (2**31 - 1))
        noise = step_rng.normal(0, variability * effective_irradiance * 0.3)
        effective_irradiance = max(0, effective_irradiance + noise)
        
        # Apply fault effects on irradiance
        for fault_key in self.state.active_faults:
            if fault_key in FAULT_TYPES:
                effective_irradiance *= FAULT_TYPES[fault_key]["irradiance_factor"]
        
        # Temperature modeling
        ambient = self.state.ambient_temp
        # Module temp increases with irradiance (simplified NOCT model)
        module_temp = ambient + DEFAULT_BASELINES["module_temp_offset"] * (effective_irradiance / 800)
        
        # High temperature fault
        if "high_temperature" in self.state.active_faults:
            module_temp += 20.0  # Additional +20°C above normal
        
        # Temperature derating factor
        temp_factor = 1.0 + DEFAULT_BASELINES["temp_coefficient"] * (
            module_temp - DEFAULT_BASELINES["ref_temperature"])
        temp_factor = max(0.5, min(1.1, temp_factor))
        
        # Clean power estimate (what would be produced without soiling)
        clean_power = RATED_CAPACITY_KW * (effective_irradiance / 1000) * temp_factor
        clean_power = max(0, clean_power)
        
        # Soiling loss
        soiling_loss_fraction = self.state.soiling_severity
        soiling_loss_kw = clean_power * soiling_loss_fraction
        
        # Equipment/fault power factor
        equipment_factor = 1.0
        for fault_key in self.state.active_faults:
            if fault_key in FAULT_TYPES:
                equipment_factor *= FAULT_TYPES[fault_key]["power_factor"]
        
        # Shading effect
        shading_factor = 1.0 - self.state.shading_severity
        equipment_factor *= shading_factor
        
        # Additional equipment loss
        equipment_factor *= (1.0 - self.state.equipment_loss_factor)
        
        equipment_loss_kw = clean_power * (1.0 - equipment_factor)
        
        # Temperature loss (relative to clean_power at ref temp)
        temperature_loss_kw = 0.0
        if module_temp > DEFAULT_BASELINES["ref_temperature"]:
            # temp_factor already accounts for this, but compute for reporting
            no_temp_power = RATED_CAPACITY_KW * (effective_irradiance / 1000)
            temperature_loss_kw = max(0, no_temp_power - clean_power)
        
        # Final measured power
        measured_power = clean_power * (1 - soiling_loss_fraction) * equipment_factor
        
        # Sensor anomaly handling
        sensor_valid = True
        if "sensor_anomaly" in self.state.active_faults:
            sensor_valid = False
            # Produce invalid reading
            if self._rng.random() < 0.3:
                measured_power = np.nan
            else:
                measured_power *= (1 + self._rng.uniform(-0.5, 0.5))
        
        measured_power = max(0, measured_power) if not np.isnan(measured_power) else np.nan
        
        # Performance ratio
        if clean_power > 10 and not np.isnan(measured_power):
            pr = measured_power / clean_power
        else:
            pr = np.nan
        
        # Loss percentages
        if clean_power > 10:
            soiling_loss_pct = soiling_loss_fraction * 100
            total_gap = clean_power - (measured_power if not np.isnan(measured_power) else 0)
            total_loss_pct = (total_gap / clean_power) * 100 if total_gap > 0 else 0
        else:
            soiling_loss_pct = 0.0
            total_loss_pct = 0.0
        
        return SimulationReading(
            timestamp=timestamp,
            step=step,
            irradiance=round(effective_irradiance, 1),
            ambient_temp=round(ambient, 1),
            module_temp=round(module_temp, 1),
            soiling_severity=round(self.state.soiling_severity, 4),
            weather=self.state.weather,
            clean_power=round(clean_power, 1),
            measured_power=round(measured_power, 1) if not np.isnan(measured_power) else np.nan,
            performance_ratio=round(pr, 4) if not np.isnan(pr) else np.nan,
            soiling_loss_pct=round(soiling_loss_pct, 1),
            total_loss_pct=round(total_loss_pct, 1),
            active_faults=list(self.state.active_faults),
            scenario_name=self.state.scenario_name,
            sensor_valid=sensor_valid,
            soiling_loss_kw=round(soiling_loss_kw, 1),
            temperature_loss_kw=round(temperature_loss_kw, 1),
            fault_loss_kw=round(equipment_loss_kw, 1),
            equipment_loss_kw=round(equipment_loss_kw, 1),
        )

    def _solar_time_factor(self, hour: float) -> float:
        """
        Simplified solar irradiance time-of-day curve.
        Peak at solar noon (~12:30), zero before sunrise (~6:00) and after sunset (~18:30).
        
        This is a simulation assumption, not a validated solar position model.
        """
        solar_noon = 12.5
        day_length = 13.0  # hours of useful sunlight
        sunrise = solar_noon - day_length / 2
        sunset = solar_noon + day_length / 2
        
        if hour < sunrise or hour > sunset:
            return 0.0
        
        # Sinusoidal approximation
        phase = np.pi * (hour - sunrise) / (sunset - sunrise)
        return max(0.0, np.sin(phase))

    def _check_thresholds(self, reading: SimulationReading) -> None:
        """Check if any alert thresholds have been crossed."""
        if reading.soiling_loss_pct > 20:
            self._log_event("threshold_crossed", "CRITICAL",
                            f"Soiling loss exceeds 20%: {reading.soiling_loss_pct:.1f}%")
        elif reading.soiling_loss_pct > 10:
            self._log_event("threshold_crossed", "WARNING",
                            f"Soiling loss exceeds 10%: {reading.soiling_loss_pct:.1f}%")
        
        if not reading.sensor_valid:
            self._log_event("sensor_anomaly", "WARNING",
                            "Sensor anomaly detected: reading may be invalid")

    def _log_event(self, event_type: str, severity: str, description: str) -> None:
        """Add an event to the simulation log."""
        self.state.event_log.append({
            "timestamp": self.state.current_time.isoformat(),
            "step": self.state.current_step,
            "event_type": event_type,
            "severity": severity,
            "description": description,
            "scenario": self.state.scenario_name,
            "soiling": self.state.soiling_severity,
            "irradiance": self.state.irradiance,
        })

    def get_readings_dataframe(self) -> pd.DataFrame:
        """Convert simulation readings to a DataFrame."""
        if not self.state.readings:
            return pd.DataFrame()
        
        records = []
        for r in self.state.readings:
            records.append({
                "timestamp": r.timestamp,
                "step": r.step,
                "irradiance": r.irradiance,
                "ambient_temp": r.ambient_temp,
                "module_temp": r.module_temp,
                "soiling_severity": r.soiling_severity,
                "weather": r.weather,
                "clean_power": r.clean_power,
                "measured_power": r.measured_power,
                "performance_ratio": r.performance_ratio,
                "soiling_loss_pct": r.soiling_loss_pct,
                "total_loss_pct": r.total_loss_pct,
                "sensor_valid": r.sensor_valid,
                "scenario": r.scenario_name,
                "soiling_loss_kw": r.soiling_loss_kw,
                "temperature_loss_kw": r.temperature_loss_kw,
                "fault_loss_kw": r.fault_loss_kw,
            })
        return pd.DataFrame(records)

    def get_event_log_dataframe(self) -> pd.DataFrame:
        """Convert event log to a DataFrame."""
        if not self.state.event_log:
            return pd.DataFrame(columns=[
                "timestamp", "step", "event_type", "severity",
                "description", "scenario",
            ])
        return pd.DataFrame(self.state.event_log)


def get_soiling_label(severity: float) -> str:
    """Get a human-readable soiling label for a severity value."""
    for (low, high), label in SOILING_LABELS.items():
        if low <= severity < high:
            return label
    return "Extreme Soiling" if severity >= 1.0 else "Clean"


def create_comparison_scenarios(names: list[str] | None = None) -> list[dict]:
    """
    Create a set of scenario configurations for comparison.
    
    Returns a list of scenario configuration dicts.
    """
    if names is None:
        names = ["Clean Panels – Sunny", "Moderate Soiling – Sunny",
                 "Cloudy Day – Light Soiling"]
    
    results = []
    for name in names:
        if name in SCENARIO_PRESETS:
            engine = SimulationEngine()
            engine.load_scenario(name)
            # Generate a mid-day reading
            engine.state.current_step = 0
            engine.state.current_time = datetime(2025, 6, 15, 12, 0)
            reading = engine._generate_reading_at_step(72)  # ~12:00 noon
            
            results.append({
                "scenario": name,
                "description": SCENARIO_PRESETS[name]["description"],
                "irradiance": reading.irradiance,
                "ambient_temp": reading.ambient_temp,
                "module_temp": reading.module_temp,
                "soiling_severity": reading.soiling_severity,
                "soiling_label": get_soiling_label(reading.soiling_severity),
                "clean_power": reading.clean_power,
                "measured_power": reading.measured_power if not np.isnan(reading.measured_power) else 0,
                "performance_ratio": reading.performance_ratio if not np.isnan(reading.performance_ratio) else 0,
                "soiling_loss_pct": reading.soiling_loss_pct,
                "total_loss_pct": reading.total_loss_pct,
                "weather": reading.weather,
                "active_faults": ", ".join(
                    FAULT_TYPES[f]["label"] for f in reading.active_faults if f in FAULT_TYPES
                ) or "None",
            })
    
    return results
