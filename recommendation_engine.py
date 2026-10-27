"""
recommendation_engine.py
========================
Cleaning recommendation engine for Solar IMS.

Responsible for:
- Evaluating cleaning recommendations
- Estimating potential energy recovery
- Estimating financial benefits using user-defined assumptions
- Accounting for cleaning cost, effectiveness, weather, and evaluation horizon
- Explaining why an action is recommended

All estimates are scenario-based and not guaranteed real-world savings.
Currency defaults to INR (Indian Rupees) as this is an India-oriented prototype.
"""

import numpy as np
from dataclasses import dataclass, field
from simulation_engine import SimulationReading, RATED_CAPACITY_KW, SIMULATION_INTERVAL_MINUTES


# ──────────────────────────────────────────────────────────────────────
# RECOMMENDATION CONFIGURATION (defaults — all user-adjustable)
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "cleaning_cost": 15000.0,           # INR per cleaning
    "energy_rate": 5.5,                 # INR per kWh (Indian grid rate assumption)
    "currency": "INR",
    "currency_symbol": "₹",
    "evaluation_horizon_hours": 720,    # 30 days
    "cleaning_effectiveness": 0.85,     # 85% of soiling removed
    "min_benefit_ratio": 1.2,           # Benefit must exceed cost by 20% to recommend
    "expected_sunshine_hours": 6.0,     # Avg daily sunshine hours (assumption)
    "rain_probability": 0.1,            # 10% chance of rain reducing soiling naturally
}


@dataclass
class RecommendationConfig:
    """User-configurable recommendation parameters."""
    cleaning_cost: float = DEFAULT_CONFIG["cleaning_cost"]
    energy_rate: float = DEFAULT_CONFIG["energy_rate"]
    currency: str = DEFAULT_CONFIG["currency"]
    currency_symbol: str = DEFAULT_CONFIG["currency_symbol"]
    evaluation_horizon_hours: float = DEFAULT_CONFIG["evaluation_horizon_hours"]
    cleaning_effectiveness: float = DEFAULT_CONFIG["cleaning_effectiveness"]
    min_benefit_ratio: float = DEFAULT_CONFIG["min_benefit_ratio"]
    expected_sunshine_hours: float = DEFAULT_CONFIG["expected_sunshine_hours"]
    rain_probability: float = DEFAULT_CONFIG["rain_probability"]


@dataclass
class EconomicAnalysis:
    """Economic analysis results."""
    recoverable_power_kw: float = 0.0       # Instantaneous recoverable power
    recoverable_energy_kwh: float = 0.0     # Energy over evaluation horizon
    energy_value: float = 0.0               # Monetary value of recovered energy
    cleaning_cost: float = 0.0              # Cost of cleaning
    net_benefit: float = 0.0                # Value minus cost
    benefit_ratio: float = 0.0              # Value / Cost
    payback_hours: float = 0.0              # Hours to recover cleaning cost
    currency: str = "INR"
    currency_symbol: str = "₹"
    assumptions: list[str] = field(default_factory=list)


@dataclass
class CleaningRecommendation:
    """Complete cleaning recommendation with explanation."""
    # Decision
    action: str                     # "Continue Monitoring", "Cleaning May Be Beneficial", etc.
    action_color: str
    priority: str                   # "Low", "Medium", "High", "Critical"
    priority_color: str
    
    # Economic analysis
    economics: EconomicAnalysis
    
    # Explanation
    explanation: str
    factors_for: list[str] = field(default_factory=list)      # Factors supporting cleaning
    factors_against: list[str] = field(default_factory=list)   # Factors against cleaning
    
    # Context
    soiling_severity: float = 0.0
    soiling_label: str = ""
    current_loss_pct: float = 0.0
    weather_condition: str = ""
    active_faults: list[str] = field(default_factory=list)
    
    # Caveats
    caveats: list[str] = field(default_factory=list)


class RecommendationEngine:
    """
    Interactive cleaning recommendation engine.
    
    Turns simulated observations into actionable recommendations
    with full economic analysis and transparent explanations.
    """

    def __init__(self, config: RecommendationConfig | None = None):
        self.config = config or RecommendationConfig()

    def recommend(self, reading: SimulationReading,
                  diagnostic_severity: str = "NORMAL",
                  primary_cause: str = "Normal Operation",
                  config: RecommendationConfig | None = None) -> CleaningRecommendation:
        """
        Generate a cleaning recommendation based on current conditions.
        
        Parameters
        ----------
        reading : SimulationReading
            Current simulated observation.
        diagnostic_severity : str
            Severity from the diagnostic engine.
        primary_cause : str
            Primary diagnosed cause from the diagnostic engine.
        config : RecommendationConfig, optional
            Override configuration for this recommendation.
        
        Returns
        -------
        CleaningRecommendation
            Complete recommendation with explanation and economics.
        """
        cfg = config or self.config
        
        # Handle invalid data
        if not reading.sensor_valid or np.isnan(reading.measured_power):
            return self._insufficient_data_recommendation(reading, cfg)
        
        # Calculate economics
        economics = self._calculate_economics(reading, cfg)
        
        # Build factors
        factors_for, factors_against = self._evaluate_factors(
            reading, economics, diagnostic_severity, primary_cause, cfg)
        
        # Make decision
        action, action_color, priority, priority_color, explanation = (
            self._make_decision(reading, economics, factors_for, factors_against,
                                diagnostic_severity, primary_cause, cfg))
        
        # Build caveats
        caveats = self._build_caveats(reading, cfg)
        
        from simulation_engine import get_soiling_label
        
        return CleaningRecommendation(
            action=action,
            action_color=action_color,
            priority=priority,
            priority_color=priority_color,
            economics=economics,
            explanation=explanation,
            factors_for=factors_for,
            factors_against=factors_against,
            soiling_severity=reading.soiling_severity,
            soiling_label=get_soiling_label(reading.soiling_severity),
            current_loss_pct=reading.soiling_loss_pct,
            weather_condition=reading.weather,
            active_faults=list(reading.active_faults),
            caveats=caveats,
        )

    def _calculate_economics(self, reading: SimulationReading,
                              cfg: RecommendationConfig) -> EconomicAnalysis:
        """
        Calculate economic analysis for a potential cleaning intervention.
        
        Energy recovery model (documented assumption):
            recoverable_power = clean_power × soiling_severity × cleaning_effectiveness
            recoverable_energy = recoverable_power × sunshine_hours × evaluation_days
        
        This is a simplified estimate. Real-world recovery depends on:
        - Actual soiling distribution across the array
        - Time-varying irradiance profile
        - Partial cleaning effectiveness
        """
        # Recoverable instantaneous power
        if reading.clean_power > 10:
            recoverable_power = (reading.clean_power 
                                 * reading.soiling_severity 
                                 * cfg.cleaning_effectiveness)
        else:
            recoverable_power = 0.0
        
        # Project energy recovery over the evaluation horizon
        evaluation_days = cfg.evaluation_horizon_hours / 24
        
        # Account for sunshine hours and rain probability
        effective_sunshine = cfg.expected_sunshine_hours * (1 - cfg.rain_probability)
        recoverable_energy = recoverable_power * effective_sunshine * evaluation_days
        
        # Monetary value
        energy_value = recoverable_energy * cfg.energy_rate
        
        # Net benefit
        net_benefit = energy_value - cfg.cleaning_cost
        
        # Benefit ratio
        benefit_ratio = (energy_value / cfg.cleaning_cost 
                         if cfg.cleaning_cost > 0 else float('inf'))
        
        # Payback period
        if recoverable_power > 0 and cfg.energy_rate > 0:
            hourly_value = recoverable_power * cfg.energy_rate
            payback_hours = cfg.cleaning_cost / hourly_value if hourly_value > 0 else float('inf')
        else:
            payback_hours = float('inf')
        
        assumptions = [
            f"Cleaning effectiveness: {cfg.cleaning_effectiveness:.0%}",
            f"Evaluation horizon: {evaluation_days:.0f} days",
            f"Expected daily sunshine: {cfg.expected_sunshine_hours:.1f} hours",
            f"Rain probability: {cfg.rain_probability:.0%}",
            f"Energy rate: {cfg.currency_symbol}{cfg.energy_rate:.2f}/kWh",
            f"Cleaning cost: {cfg.currency_symbol}{cfg.cleaning_cost:,.0f}",
            "Assumes constant soiling rate without intervention",
            "Does not account for further soiling accumulation during evaluation period",
        ]
        
        return EconomicAnalysis(
            recoverable_power_kw=round(recoverable_power, 1),
            recoverable_energy_kwh=round(recoverable_energy, 1),
            energy_value=round(energy_value, 2),
            cleaning_cost=cfg.cleaning_cost,
            net_benefit=round(net_benefit, 2),
            benefit_ratio=round(benefit_ratio, 2),
            payback_hours=round(payback_hours, 1),
            currency=cfg.currency,
            currency_symbol=cfg.currency_symbol,
            assumptions=assumptions,
        )

    def _evaluate_factors(self, reading: SimulationReading,
                           economics: EconomicAnalysis,
                           diagnostic_severity: str,
                           primary_cause: str,
                           cfg: RecommendationConfig) -> tuple[list[str], list[str]]:
        """Evaluate factors for and against cleaning."""
        factors_for = []
        factors_against = []
        
        # Economic factors
        if economics.net_benefit > 0:
            factors_for.append(
                f"Positive net benefit: {cfg.currency_symbol}{economics.net_benefit:,.0f} "
                f"over {cfg.evaluation_horizon_hours / 24:.0f} days")
        else:
            factors_against.append(
                f"Negative net benefit: {cfg.currency_symbol}{economics.net_benefit:,.0f}")
        
        if economics.benefit_ratio > cfg.min_benefit_ratio:
            factors_for.append(
                f"Benefit-cost ratio ({economics.benefit_ratio:.1f}x) exceeds "
                f"minimum threshold ({cfg.min_benefit_ratio:.1f}x)")
        elif economics.benefit_ratio < 1.0:
            factors_against.append(
                f"Cleaning cost exceeds estimated recovery value")
        
        # Soiling factors
        if reading.soiling_severity > 0.15:
            factors_for.append(
                f"Heavy soiling ({reading.soiling_severity:.1%}) — significant recovery potential")
        elif reading.soiling_severity > 0.05:
            factors_for.append(
                f"Moderate soiling ({reading.soiling_severity:.1%}) — recovery may be worthwhile")
        elif reading.soiling_severity < 0.02:
            factors_against.append(
                f"Minimal soiling ({reading.soiling_severity:.1%}) — limited recovery potential")
        
        # Weather factors
        if reading.weather == "Rainy":
            factors_against.append(
                "Rain may naturally reduce soiling — consider waiting")
        if reading.weather in ("Cloudy", "Rainy"):
            factors_against.append(
                "Current generation is weather-limited — cleaning benefit is reduced")
        
        # Diagnostic factors
        if primary_cause in ("Equipment / Connection-Related Loss", "Sensor Anomaly"):
            factors_against.append(
                f"Primary issue appears to be {primary_cause} — cleaning may not resolve it")
        
        if primary_cause == "Cloud / Weather-Related Reduction" and reading.soiling_severity < 0.05:
            factors_against.append(
                "Performance reduction is primarily weather-related, not soiling")
        
        # Recoverable power
        if economics.recoverable_power_kw > 100:
            factors_for.append(
                f"Significant recoverable power: {economics.recoverable_power_kw:.0f} kW")
        
        return factors_for, factors_against

    def _make_decision(self, reading: SimulationReading,
                       economics: EconomicAnalysis,
                       factors_for: list[str],
                       factors_against: list[str],
                       diagnostic_severity: str,
                       primary_cause: str,
                       cfg: RecommendationConfig) -> tuple[str, str, str, str, str]:
        """
        Make the cleaning decision.
        
        Returns: (action, action_color, priority, priority_color, explanation)
        
        Decision logic:
        1. If equipment fault is primary cause → Investigate first
        2. If sensor anomaly → Insufficient data
        3. If economics positive AND soiling significant → Recommend cleaning
        4. If economics positive but soiling mild → May be beneficial
        5. Otherwise → Continue monitoring
        """
        # Equipment fault overrides cleaning recommendation
        if primary_cause in ("Equipment / Connection-Related Loss",):
            return (
                "Investigate Equipment First",
                "#FF9800",
                "Medium",
                "#FF9800",
                f"Equipment-related loss is the primary suspected issue. "
                f"Cleaning may not resolve the {economics.recoverable_power_kw:.0f} kW shortfall. "
                f"Investigate equipment before scheduling cleaning."
            )
        
        # Cloud cover primary and minimal soiling
        if (primary_cause == "Cloud / Weather-Related Reduction"
                and reading.soiling_severity < 0.03):
            return (
                "Continue Monitoring",
                "#4CAF50",
                "Low",
                "#4CAF50",
                f"Performance reduction is primarily attributed to weather conditions "
                f"({reading.weather}). Panel soiling is minimal ({reading.soiling_severity:.1%}). "
                f"Cleaning would not significantly improve output."
            )
        
        # Positive economics — check benefit ratio
        if (economics.net_benefit > 0 
                and economics.benefit_ratio >= cfg.min_benefit_ratio
                and reading.soiling_severity > 0.05):
            severity_word = "Critical" if reading.soiling_severity > 0.15 else "Moderate"
            priority = "High" if reading.soiling_severity > 0.15 else "Medium"
            p_color = "#EF5350" if priority == "High" else "#FF9800"
            return (
                "Cleaning Recommended",
                "#EF5350" if priority == "High" else "#FF9800",
                priority,
                p_color,
                f"Cleaning is recommended in this simulated scenario because estimated "
                f"recoverable energy ({economics.recoverable_energy_kwh:,.0f} kWh) over "
                f"the selected evaluation horizon ({cfg.evaluation_horizon_hours / 24:.0f} days) "
                f"exceeds the configured cleaning cost "
                f"({cfg.currency_symbol}{cfg.cleaning_cost:,.0f}). "
                f"Net benefit: {cfg.currency_symbol}{economics.net_benefit:,.0f}. "
                f"Confirm the panel condition before acting on the recommendation."
            )
        
        # Marginal benefit
        if economics.net_benefit > 0 and reading.soiling_severity > 0.02:
            return (
                "Cleaning May Be Beneficial",
                "#FFC107",
                "Low",
                "#FFC107",
                f"Cleaning could recover an estimated "
                f"{economics.recoverable_energy_kwh:,.0f} kWh "
                f"({cfg.currency_symbol}{economics.energy_value:,.0f}) over "
                f"{cfg.evaluation_horizon_hours / 24:.0f} days, but the benefit-to-cost "
                f"ratio ({economics.benefit_ratio:.1f}x) is below the recommended "
                f"threshold ({cfg.min_benefit_ratio:.1f}x). "
                f"Consider scheduling during the next maintenance window."
            )
        
        # Negative economics or minimal soiling
        if economics.net_benefit <= 0:
            return (
                "Continue Monitoring",
                "#4CAF50",
                "Low",
                "#4CAF50",
                f"Estimated cleaning cost ({cfg.currency_symbol}{cfg.cleaning_cost:,.0f}) "
                f"exceeds the projected energy recovery value "
                f"({cfg.currency_symbol}{economics.energy_value:,.0f}). "
                f"Cleaning is not economically justified under current assumptions."
            )
        
        return (
            "Continue Monitoring",
            "#4CAF50",
            "Low",
            "#4CAF50",
            "System operating within acceptable parameters. Continue standard monitoring."
        )

    def _insufficient_data_recommendation(self, reading: SimulationReading,
                                           cfg: RecommendationConfig) -> CleaningRecommendation:
        """Generate recommendation when data is insufficient."""
        from simulation_engine import get_soiling_label
        return CleaningRecommendation(
            action="Insufficient Data",
            action_color="#78909C",
            priority="Low",
            priority_color="#78909C",
            economics=EconomicAnalysis(currency=cfg.currency, currency_symbol=cfg.currency_symbol),
            explanation="Sensor data is invalid or missing. Cannot generate a reliable "
                        "cleaning recommendation. Investigate sensor health first.",
            soiling_severity=reading.soiling_severity,
            soiling_label=get_soiling_label(reading.soiling_severity),
            current_loss_pct=0.0,
            weather_condition=reading.weather,
            active_faults=list(reading.active_faults),
            caveats=["Recommendation unreliable due to missing or invalid sensor data"],
        )

    def _build_caveats(self, reading: SimulationReading,
                       cfg: RecommendationConfig) -> list[str]:
        """Build list of caveats for the recommendation."""
        caveats = [
            "This is a scenario-based estimate, not a guaranteed real-world saving",
            "Actual recovery depends on true soiling distribution and cleaning quality",
        ]
        if reading.weather in ("Rainy",):
            caveats.append("Expected rainfall may naturally reduce soiling")
        if reading.active_faults:
            caveats.append("Active fault conditions may affect the accuracy of recovery estimates")
        if cfg.cleaning_effectiveness < 0.5:
            caveats.append("Low cleaning effectiveness significantly reduces projected benefits")
        return caveats
