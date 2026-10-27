"""
diagnostic_engine.py
====================
Rule-based diagnostic engine for Solar IMS.

Responsible for:
- Comparing observed/simulated output against expected output
- Evaluating deviations
- Generating diagnostic hypotheses
- Calculating rule-based suspicion scores
- Explaining the evidence supporting a diagnosis

All scores are labelled as "rule-based suspicion scores" — NOT ML confidence.
"""

import numpy as np
from dataclasses import dataclass, field
from simulation_engine import SimulationReading, FAULT_TYPES, get_soiling_label


# ──────────────────────────────────────────────────────────────────────
# DIAGNOSTIC THRESHOLDS (Simulation assumptions — not validated)
# ──────────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "pr_healthy": 0.95,
    "pr_monitor": 0.85,
    "pr_schedule_cleaning": 0.70,
    "irradiance_low": 200.0,        # W/m² — below this, diagnostics are unreliable
    "irradiance_cloudy": 400.0,     # W/m² — cloud cover likely
    "temp_high": 45.0,              # °C module temp — elevated
    "temp_very_high": 55.0,         # °C module temp — critical
    "power_near_zero": 50.0,        # kW — effectively zero output
    "loss_minor": 5.0,              # % — minor loss
    "loss_moderate": 15.0,          # % — moderate loss
    "loss_severe": 30.0,            # % — severe loss
}

# Severity levels
SEVERITY_LEVELS = {
    "NORMAL": {"label": "Normal", "color": "#4CAF50", "priority": 4},
    "ADVISORY": {"label": "Advisory", "color": "#FFC107", "priority": 3},
    "WARNING": {"label": "Warning", "color": "#FF9800", "priority": 2},
    "CRITICAL": {"label": "Critical", "color": "#EF5350", "priority": 1},
}


@dataclass
class DiagnosticHypothesis:
    """A single diagnostic hypothesis with supporting evidence."""
    cause: str
    suspicion_score: float      # 0-100, rule-based
    evidence: list[str]
    description: str
    is_primary: bool = False
    fault_type: str = ""        # matching fault type key


@dataclass
class DiagnosticResult:
    """Complete diagnostic assessment for a reading."""
    # Overall assessment
    severity: str               # NORMAL, ADVISORY, WARNING, CRITICAL
    severity_label: str
    severity_color: str
    
    # Primary diagnosis
    primary_cause: str
    primary_description: str
    primary_score: float
    
    # All hypotheses ranked by score
    hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)
    
    # Evidence and limitations
    evidence_summary: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    
    # Recommended action
    recommended_action: str = ""
    action_explanation: str = ""
    
    # Data quality
    data_quality: str = "Valid"
    is_sufficient_data: bool = True
    
    # Key metrics used
    performance_ratio: float = 0.0
    total_loss_pct: float = 0.0
    soiling_loss_pct: float = 0.0


class DiagnosticEngine:
    """
    Transparent rule-based diagnostic engine.
    
    This engine does NOT use machine learning for diagnostics.
    All assessments are derived from transparent rules and clearly
    documented thresholds.
    
    Scores are labelled as "Simulation-based suspicion scores"
    and do NOT represent calibrated probabilities.
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**THRESHOLDS, **(thresholds or {})}

    def diagnose(self, reading: SimulationReading) -> DiagnosticResult:
        """
        Perform a complete diagnostic assessment on a simulation reading.
        
        Parameters
        ----------
        reading : SimulationReading
            The current simulated observation.
        
        Returns
        -------
        DiagnosticResult
            Complete diagnostic assessment with hypotheses and evidence.
        """
        # Check data quality first
        if not reading.sensor_valid:
            return self._sensor_anomaly_result(reading)
        
        if np.isnan(reading.measured_power) or np.isnan(reading.clean_power):
            return self._insufficient_data_result(reading)
        
        if reading.irradiance < self.thresholds["irradiance_low"]:
            # If during daytime and cloudy/rainy weather or cloud fault active, diagnose cloud cover
            if reading.weather in ("Cloudy", "Rainy") or "cloud_cover" in reading.active_faults:
                pass
            else:
                return self._low_irradiance_result(reading)
        
        # Generate all hypotheses
        hypotheses = self._generate_hypotheses(reading)
        
        # Sort by suspicion score
        hypotheses.sort(key=lambda h: h.suspicion_score, reverse=True)
        
        # Mark primary
        if hypotheses:
            hypotheses[0].is_primary = True
        
        # Determine severity
        severity = self._determine_severity(reading)
        sev_info = SEVERITY_LEVELS[severity]
        
        # Build evidence summary
        evidence = self._build_evidence_summary(reading)
        
        # Build limitations
        limitations = self._build_limitations(reading)
        
        # Determine recommended action
        action, explanation = self._recommend_action(reading, hypotheses)
        
        primary = hypotheses[0] if hypotheses else None
        
        return DiagnosticResult(
            severity=severity,
            severity_label=sev_info["label"],
            severity_color=sev_info["color"],
            primary_cause=primary.cause if primary else "Unknown",
            primary_description=primary.description if primary else "",
            primary_score=primary.suspicion_score if primary else 0.0,
            hypotheses=hypotheses,
            evidence_summary=evidence,
            limitations=limitations,
            recommended_action=action,
            action_explanation=explanation,
            data_quality="Valid" if reading.sensor_valid else "Anomalous",
            is_sufficient_data=True,
            performance_ratio=reading.performance_ratio if not np.isnan(reading.performance_ratio) else 0.0,
            total_loss_pct=reading.total_loss_pct,
            soiling_loss_pct=reading.soiling_loss_pct,
        )

    def _generate_hypotheses(self, reading: SimulationReading) -> list[DiagnosticHypothesis]:
        """Generate all possible diagnostic hypotheses for a reading."""
        hypotheses = []
        
        pr = reading.performance_ratio if not np.isnan(reading.performance_ratio) else 0.0
        loss = reading.total_loss_pct
        soiling = reading.soiling_severity
        irr = reading.irradiance
        mod_temp = reading.module_temp
        
        # 1. Soiling hypothesis
        soiling_score = self._score_soiling(reading)
        soiling_label = get_soiling_label(soiling)
        hypotheses.append(DiagnosticHypothesis(
            cause="Soiling-Related Degradation",
            suspicion_score=soiling_score,
            evidence=self._soiling_evidence(reading),
            description=(
                f"Soiling suspected: simulated output remains below the clean-reference "
                f"estimate ({reading.soiling_loss_pct:.1f}% soiling loss, "
                f"severity: {soiling_label}). "
                f"Irradiance is {'relatively stable' if irr > 400 else 'low'}, "
                f"suggesting the gap is not primarily weather-related."
            ),
            fault_type="gradual_soiling",
        ))
        
        # 2. Cloud/weather hypothesis
        cloud_score = self._score_cloud(reading)
        hypotheses.append(DiagnosticHypothesis(
            cause="Cloud / Weather-Related Reduction",
            suspicion_score=cloud_score,
            evidence=self._cloud_evidence(reading),
            description=(
                f"Cloud-related reduction suspected: irradiance ({irr:.0f} W/m²) "
                f"and output have decreased together under {reading.weather} conditions."
            ),
            fault_type="cloud_cover",
        ))
        
        # 3. Temperature hypothesis
        temp_score = self._score_temperature(reading)
        hypotheses.append(DiagnosticHypothesis(
            cause="Temperature-Related Efficiency Loss",
            suspicion_score=temp_score,
            evidence=self._temperature_evidence(reading),
            description=(
                f"Elevated panel temperature ({mod_temp:.1f}°C) reducing "
                f"conversion efficiency beyond normal thermal derating."
            ),
            fault_type="high_temperature",
        ))
        
        # 4. Equipment hypothesis
        equip_score = self._score_equipment(reading)
        hypotheses.append(DiagnosticHypothesis(
            cause="Equipment / Connection-Related Loss",
            suspicion_score=equip_score,
            evidence=self._equipment_evidence(reading),
            description=(
                f"Possible equipment-related loss: output is below the reference "
                f"estimate despite adequate simulated irradiance ({irr:.0f} W/m²)."
            ),
            fault_type="inverter_loss",
        ))
        
        # 5. Shading hypothesis
        shading_score = self._score_shading(reading)
        hypotheses.append(DiagnosticHypothesis(
            cause="Partial Shading",
            suspicion_score=shading_score,
            evidence=self._shading_evidence(reading),
            description=(
                f"Partial shading suspected: output reduction pattern "
                f"consistent with physical obstruction."
            ),
            fault_type="partial_shading",
        ))
        
        # 6. Normal operation
        normal_score = max(0, 100 - max(h.suspicion_score for h in hypotheses))
        hypotheses.append(DiagnosticHypothesis(
            cause="Normal Operation",
            suspicion_score=normal_score,
            evidence=[
                f"Performance ratio: {pr:.3f}",
                f"Total loss: {loss:.1f}%",
            ],
            description="No significant anomaly detected in current conditions.",
            fault_type="none",
        ))
        
        return hypotheses

    # ── Scoring functions (rule-based, 0-100 scale) ──────────────────

    def _score_soiling(self, r: SimulationReading) -> float:
        """Score soiling hypothesis based on evidence."""
        score = 0.0
        soiling = r.soiling_severity
        
        # Direct soiling severity (primary factor)
        if soiling > 0.15:
            score += 50
        elif soiling > 0.05:
            score += 30
        elif soiling > 0.02:
            score += 15
        
        # Stable irradiance + low PR = soiling more likely
        if r.irradiance > 400 and not np.isnan(r.performance_ratio):
            if r.performance_ratio < 0.70:
                score += 25
            elif r.performance_ratio < 0.85:
                score += 15
            elif r.performance_ratio < 0.95:
                score += 5
        
        # Active soiling faults increase score
        soiling_faults = {"gradual_soiling", "sudden_soiling"}
        if soiling_faults & set(r.active_faults):
            score += 10
        
        # Weather is clear but output is low
        if r.weather == "Sunny" and r.soiling_loss_pct > 5:
            score += 10
        
        return min(100, score)

    def _score_cloud(self, r: SimulationReading) -> float:
        """Score cloud/weather hypothesis."""
        score = 0.0
        
        if r.weather in ("Cloudy", "Rainy"):
            score += 40
        elif r.weather == "Variable":
            score += 25
        
        if r.irradiance < self.thresholds["irradiance_cloudy"]:
            score += 20
        elif r.irradiance < 600:
            score += 10
        
        if "cloud_cover" in r.active_faults:
            score += 20
        
        # If irradiance is low AND soiling is also present, both share blame
        if r.soiling_severity > 0.05 and r.irradiance < 400:
            score -= 10  # Reduce cloud score when soiling also active
        
        return max(0, min(100, score))

    def _score_temperature(self, r: SimulationReading) -> float:
        """Score temperature hypothesis."""
        score = 0.0
        
        if r.module_temp > self.thresholds["temp_very_high"]:
            score += 50
        elif r.module_temp > self.thresholds["temp_high"]:
            score += 30
        elif r.module_temp > 38:
            score += 10
        
        if "high_temperature" in r.active_faults:
            score += 25
        
        if r.temperature_loss_kw > 100:
            score += 15
        
        return min(100, score)

    def _score_equipment(self, r: SimulationReading) -> float:
        """Score equipment-related hypothesis."""
        score = 0.0
        
        equip_faults = {"inverter_loss", "partial_shading"}
        active_equip = equip_faults & set(r.active_faults)
        
        if active_equip:
            score += 40
        
        if r.fault_loss_kw > 200:
            score += 25
        elif r.fault_loss_kw > 50:
            score += 15
        
        # High irradiance but very low PR (not explained by soiling alone)
        if (r.irradiance > 600 and not np.isnan(r.performance_ratio)
                and r.performance_ratio < 0.5 and r.soiling_severity < 0.1):
            score += 20
        
        return min(100, score)

    def _score_shading(self, r: SimulationReading) -> float:
        """Score shading hypothesis."""
        score = 0.0
        
        if "partial_shading" in r.active_faults:
            score += 50
        
        # Time-based: shading more likely at certain hours
        hour = r.timestamp.hour
        if hour < 8 or hour > 16:
            score += 10
        
        if r.fault_loss_kw > 100 and r.soiling_severity < 0.05:
            score += 15
        
        return min(100, score)

    # ── Evidence builders ────────────────────────────────────────────

    def _soiling_evidence(self, r: SimulationReading) -> list[str]:
        evidence = []
        evidence.append(f"Soiling severity: {r.soiling_severity:.1%} ({get_soiling_label(r.soiling_severity)})")
        evidence.append(f"Soiling loss: {r.soiling_loss_pct:.1f}% ({r.soiling_loss_kw:.0f} kW)")
        if r.irradiance > 400:
            evidence.append(f"Irradiance adequate ({r.irradiance:.0f} W/m²) — loss not weather-driven")
        if not np.isnan(r.performance_ratio):
            evidence.append(f"Performance ratio: {r.performance_ratio:.3f}")
        return evidence

    def _cloud_evidence(self, r: SimulationReading) -> list[str]:
        evidence = []
        evidence.append(f"Weather condition: {r.weather}")
        evidence.append(f"Effective irradiance: {r.irradiance:.0f} W/m²")
        if r.weather in ("Cloudy", "Rainy", "Variable"):
            evidence.append("Weather conditions consistent with cloud cover")
        if r.irradiance < 400:
            evidence.append("Low irradiance consistent with atmospheric obstruction")
        return evidence

    def _temperature_evidence(self, r: SimulationReading) -> list[str]:
        evidence = []
        evidence.append(f"Module temperature: {r.module_temp:.1f}°C")
        evidence.append(f"Ambient temperature: {r.ambient_temp:.1f}°C")
        evidence.append(f"Temperature loss: {r.temperature_loss_kw:.0f} kW")
        if r.module_temp > 45:
            evidence.append("Panel temperature exceeds optimal range (>45°C)")
        return evidence

    def _equipment_evidence(self, r: SimulationReading) -> list[str]:
        evidence = []
        active_faults_labels = [
            FAULT_TYPES[f]["label"] for f in r.active_faults
            if f in FAULT_TYPES and f not in ("none", "gradual_soiling", "sudden_soiling")
        ]
        if active_faults_labels:
            evidence.append(f"Active faults: {', '.join(active_faults_labels)}")
        evidence.append(f"Equipment-related loss: {r.fault_loss_kw:.0f} kW")
        if r.fault_loss_kw > 50:
            evidence.append("Significant power loss not explained by soiling or weather")
        return evidence

    def _shading_evidence(self, r: SimulationReading) -> list[str]:
        evidence = []
        if "partial_shading" in r.active_faults:
            evidence.append("Partial shading fault is active")
        hour = r.timestamp.hour
        evidence.append(f"Time of day: {hour:02d}:00 — {'low sun angle' if hour < 8 or hour > 16 else 'mid-day'}")
        return evidence

    # ── Result builders ──────────────────────────────────────────────

    def _determine_severity(self, reading: SimulationReading) -> str:
        """Determine overall severity from reading metrics."""
        pr = reading.performance_ratio if not np.isnan(reading.performance_ratio) else 1.0
        loss = reading.total_loss_pct
        
        if pr < self.thresholds["pr_schedule_cleaning"] or loss > self.thresholds["loss_severe"]:
            return "CRITICAL"
        elif pr < self.thresholds["pr_monitor"] or loss > self.thresholds["loss_moderate"]:
            return "WARNING"
        elif pr < self.thresholds["pr_healthy"] or loss > self.thresholds["loss_minor"]:
            return "ADVISORY"
        return "NORMAL"

    def _build_evidence_summary(self, reading: SimulationReading) -> list[str]:
        """Build a concise evidence summary."""
        ev = []
        pr = reading.performance_ratio
        if not np.isnan(pr):
            ev.append(f"Performance ratio: {pr:.3f} "
                      f"({'normal' if pr >= 0.95 else 'degraded'})")
        ev.append(f"Total estimated loss: {reading.total_loss_pct:.1f}%")
        ev.append(f"Soiling severity: {reading.soiling_severity:.1%}")
        ev.append(f"Weather: {reading.weather} | Irradiance: {reading.irradiance:.0f} W/m²")
        ev.append(f"Module temp: {reading.module_temp:.1f}°C")
        if reading.active_faults:
            fault_names = [FAULT_TYPES[f]["label"] for f in reading.active_faults if f in FAULT_TYPES]
            if fault_names:
                ev.append(f"Active conditions: {', '.join(fault_names)}")
        return ev

    def _build_limitations(self, reading: SimulationReading) -> list[str]:
        """Build list of diagnostic limitations."""
        limits = [
            "All values are from a simulation — not actual sensor telemetry",
            "Scores are rule-based suspicion indicators, not calibrated probabilities",
            "Soiling cannot be conclusively distinguished from all other causes without physical inspection",
            "No string-level or inverter fault code monitoring available",
        ]
        if reading.weather in ("Cloudy", "Variable", "Rainy"):
            limits.append("Weather variability reduces diagnostic precision")
        if reading.irradiance < 400:
            limits.append("Low irradiance conditions reduce diagnostic reliability")
        return limits

    def _recommend_action(self, reading: SimulationReading,
                          hypotheses: list[DiagnosticHypothesis]) -> tuple[str, str]:
        """Determine recommended action based on diagnostics."""
        if not hypotheses:
            return "Continue Monitoring", "Insufficient evidence for a specific recommendation."
        
        primary = hypotheses[0]
        pr = reading.performance_ratio if not np.isnan(reading.performance_ratio) else 1.0
        
        # Sensor anomaly takes priority
        if not reading.sensor_valid:
            return ("Investigate Sensor",
                    "Sensor anomaly detected. Investigate sensor hardware before taking "
                    "maintenance action based on potentially invalid readings.")
        
        # Equipment fault suspected
        if primary.fault_type in ("inverter_loss",) and primary.suspicion_score > 40:
            return ("Investigate Equipment",
                    "Equipment-related loss suspected. Inspect inverter and connections "
                    "before scheduling panel cleaning, as cleaning will not resolve "
                    "equipment faults.")
        
        # Soiling is primary cause
        if primary.fault_type in ("gradual_soiling", "sudden_soiling"):
            if pr < 0.70:
                return ("Cleaning Recommended",
                        f"Soiling is the primary suspected cause (score: {primary.suspicion_score:.0f}/100). "
                        f"Performance ratio ({pr:.3f}) is below the critical threshold (0.70). "
                        f"Confirm panel condition before proceeding.")
            elif pr < 0.85:
                return ("Schedule Cleaning",
                        f"Soiling is the primary suspected cause (score: {primary.suspicion_score:.0f}/100). "
                        f"Performance ratio ({pr:.3f}) indicates moderate degradation.")
            elif pr < 0.95:
                return ("Monitor — Cleaning May Be Beneficial",
                        f"Mild soiling detected (score: {primary.suspicion_score:.0f}/100). "
                        f"Continue monitoring; schedule cleaning if conditions worsen.")
        
        # Cloud cover is primary
        if primary.fault_type == "cloud_cover" and primary.suspicion_score > 40:
            return ("Continue Monitoring",
                    "Performance reduction is primarily attributed to weather conditions. "
                    "Cleaning is unlikely to improve output under current cloud cover.")
        
        # Normal operation
        if pr >= 0.95:
            return ("Continue Monitoring",
                    "System operating within normal parameters. No immediate action required.")
        
        return ("Continue Monitoring",
                "No single cause clearly identified. Continue monitoring and "
                "reassess when conditions change.")

    def _sensor_anomaly_result(self, reading: SimulationReading) -> DiagnosticResult:
        """Generate a diagnostic result for sensor anomaly."""
        return DiagnosticResult(
            severity="WARNING",
            severity_label="Warning",
            severity_color="#FF9800",
            primary_cause="Sensor Anomaly",
            primary_description=(
                "Sensor anomaly detected: the simulated measurement is missing or "
                "outside the configured plausible range. Diagnostic assessment "
                "is unreliable with invalid sensor data."
            ),
            primary_score=0.0,
            hypotheses=[DiagnosticHypothesis(
                cause="Sensor Anomaly",
                suspicion_score=90.0,
                evidence=["Sensor readings flagged as invalid",
                           "Measurement may be missing or out of range"],
                description="Simulated sensor malfunction. Readings are unreliable.",
                is_primary=True,
                fault_type="sensor_anomaly",
            )],
            evidence_summary=["Sensor data flagged as anomalous"],
            limitations=["All diagnostics unreliable with invalid sensor data"],
            recommended_action="Investigate Sensor",
            action_explanation="Check sensor hardware and data pipeline. Do not take "
                               "maintenance action based on anomalous readings.",
            data_quality="Anomalous",
            is_sufficient_data=False,
            performance_ratio=0.0,
            total_loss_pct=0.0,
            soiling_loss_pct=0.0,
        )

    def _insufficient_data_result(self, reading: SimulationReading) -> DiagnosticResult:
        """Generate a result when data is insufficient."""
        return DiagnosticResult(
            severity="ADVISORY",
            severity_label="Advisory",
            severity_color="#FFC107",
            primary_cause="Insufficient Data",
            primary_description="Cannot compute performance metrics due to missing or invalid values.",
            primary_score=0.0,
            evidence_summary=["One or more critical measurements are missing"],
            limitations=["Diagnostic assessment cannot be performed"],
            recommended_action="Continue Monitoring",
            action_explanation="Insufficient data for a reliable recommendation.",
            data_quality="Incomplete",
            is_sufficient_data=False,
        )

    def _low_irradiance_result(self, reading: SimulationReading) -> DiagnosticResult:
        """Generate a result for low irradiance conditions."""
        return DiagnosticResult(
            severity="NORMAL",
            severity_label="Normal",
            severity_color="#4CAF50",
            primary_cause="Low Irradiance",
            primary_description=(
                f"Irradiance ({reading.irradiance:.0f} W/m²) is below the diagnostic "
                f"threshold ({self.thresholds['irradiance_low']:.0f} W/m²). "
                f"This is typical for dawn/dusk or heavily overcast conditions."
            ),
            primary_score=0.0,
            evidence_summary=[
                f"Irradiance: {reading.irradiance:.0f} W/m² (below {self.thresholds['irradiance_low']:.0f})",
                "Low-irradiance period — diagnostics not meaningful",
            ],
            limitations=["Diagnostic precision is low under minimal irradiance"],
            recommended_action="Continue Monitoring",
            action_explanation="Low irradiance period. Diagnostics will be more reliable "
                               "during peak sunlight hours.",
            data_quality="Valid — Low Irradiance",
            is_sufficient_data=True,
        )
