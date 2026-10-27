"""
prepare_data.py
===============
Replicates the PoC2.ipynb pipeline headlessly and exports the final
processed dataset (with XGBoost predictions and cleaning recommendations)
so the Streamlit dashboard can consume it without retraining.

Usage:
    python prepare_data.py

Outputs:
    processed_output.csv   — full merged dataset with predictions
    model_metrics.json     — MAE, RMSE, R², feature importances
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore")

RATED_CAPACITY_KW = 4000


# ─── PREPROCESSING ───────────────────────────────────────────────────
def preprocess_ac(df):
    df = df.copy()
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    df = df.rename(columns={
        "inverter_1_ac_power_(kw)_inv_150143": "inv1_power",
        "inverter_2_ac_power_(kw)_inv_150144": "inv2_power",
    })
    df["total_power"] = df["inv1_power"].fillna(0) + df["inv2_power"].fillna(0)
    return df[["measured_on", "inv1_power", "inv2_power", "total_power"]]


def preprocess_irr(df):
    df = df.copy()
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    pad1 = "pyranometer_(class_a)_pad_1_poa_irradiance_temp_compensated_(w/m2)_o_149724"
    pad2 = "pyranometer_(class_a)_pad_2_poa_irradiance_temp_compensated_(w/m2)_o_149726"
    df["irradiance"] = df[[pad1, pad2]].mean(axis=1)
    return df[["measured_on", "irradiance"]]


def preprocess_env(df):
    df = df.copy()
    df["measured_on"] = pd.to_datetime(df["measured_on"])
    df = df.rename(columns={
        "weather_station_ambient_temperature_(c)_o_149727": "ambient_temp",
        "thermocouple_pad_1_back-of-module_temperature_1_(c)_o_149728": "module_temp_1",
        "thermocouple_pad_2_back-of-module_temperature_1_(c)_o_149730": "module_temp_2",
    })
    df["module_temp"] = df[["module_temp_1", "module_temp_2"]].mean(axis=1)
    return df[["measured_on", "ambient_temp", "module_temp"]]


# ─── FEATURE ENGINEERING ─────────────────────────────────────────────
def engineer_features(df, rated_kw=RATED_CAPACITY_KW):
    df = df.copy()

    df["performance_ratio"] = np.where(
        df["irradiance"] > 50,
        df["total_power"] / (rated_kw * df["irradiance"] / 1000),
        np.nan,
    )
    df["est_power_loss"] = np.where(
        df["irradiance"] > 50,
        (rated_kw * df["irradiance"] / 1000) - df["total_power"],
        0,
    )
    df["est_power_loss"] = df["est_power_loss"].clip(lower=0)

    df["high_irradiance_flag"] = (df["irradiance"] > 800).astype(int)

    df["rolling_avg_power"] = (
        df["total_power"].rolling(window=12, min_periods=1).mean()
    )
    df["rolling_perf_ratio"] = (
        df["performance_ratio"].rolling(window=12, min_periods=1).mean()
    )
    df["temp_diff"] = df["module_temp"] - df["ambient_temp"]

    df["date"] = df["measured_on"].dt.date
    daily_energy = (
        df.groupby("date")["total_power"].sum().mul(5 / 60).rename("daily_energy_kwh")
    )
    df = df.merge(daily_energy, on="date", how="left")
    return df


# ─── RECOMMENDATION ENGINE ───────────────────────────────────────────
def classify_condition(row):
    pr = row.get("model_perf_ratio", np.nan)
    loss = row.get("loss_pct", 0)
    irr = row.get("irradiance", 0)
    act_pwr = row.get("total_power", 0)
    exp_pwr = row.get("expected_power", 0)

    if irr < 200:
        return "Low Sunlight", "Irradiance below 200 W/m2 - night or overcast"
    if exp_pwr < 100:
        return "Low Generation", "Expected generation too low for analysis"
    if act_pwr < 50:
        return "Investigate", "Actual output near zero - possible equipment issue"
    if pd.isna(pr):
        return "Insufficient Data", "Cannot compute performance ratio"
    if pr >= 0.95:
        return "Healthy", "Performance ratio within normal range"
    if pr >= 0.85:
        return "Monitor", f"Mild degradation detected (PR={pr:.2f})"
    if pr >= 0.70:
        return "Schedule Cleaning", f"Significant loss - PR={pr:.2f}, Loss={loss:.1f}%"
    return "Clean Immediately", f"Severe loss - PR={pr:.2f}, Loss={loss:.1f}%"


# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    print("Loading datasets...")
    ac_raw = pd.read_csv("9068_ac_power_data_20240101_20250430.csv")
    irr_raw = pd.read_csv("9068_irradiance_data_20240101_20250430.csv")
    env_raw = pd.read_csv("9068_environment_data_20240101_20250430.csv")

    print("Preprocessing...")
    ac = preprocess_ac(ac_raw)
    irr = preprocess_irr(irr_raw)
    env = preprocess_env(env_raw)

    print("Merging datasets...")
    merged = (
        ac.merge(irr, on="measured_on", how="inner")
        .merge(env, on="measured_on", how="inner")
    )
    merged = merged.dropna().sort_values("measured_on").reset_index(drop=True)

    print("Engineering features...")
    merged = engineer_features(merged)

    print("Training XGBoost model...")
    features = ["irradiance", "ambient_temp", "module_temp", "temp_diff"]
    target = "total_power"
    model_df = merged.dropna(subset=features + [target]).copy()

    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  = {mae:.2f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  R2   = {r2:.4f}")

    print("Generating predictions for full dataset...")
    merged["expected_power"] = model.predict(merged[features].fillna(0))
    merged["power_diff"] = merged["expected_power"] - merged["total_power"]
    merged["model_perf_ratio"] = np.where(
        merged["expected_power"] > 50,
        merged["total_power"] / merged["expected_power"],
        np.nan,
    )
    merged["model_perf_ratio"] = merged["model_perf_ratio"].clip(0, 1.2)
    merged["loss_pct"] = np.where(
        merged["expected_power"] > 50,
        (merged["power_diff"] / merged["expected_power"]) * 100,
        0,
    )
    merged["loss_pct"] = merged["loss_pct"].clip(0, 100)

    print("Running recommendation engine...")
    results = merged.apply(classify_condition, axis=1, result_type="expand")
    merged["recommendation"] = results[0]
    merged["reason"] = results[1]

    # Assign numeric priority
    priority_map = {
        "Clean Immediately": 1,
        "Schedule Cleaning": 2,
        "Investigate": 3,
        "Monitor": 4,
        "Low Generation": 5,
        "Insufficient Data": 6,
        "Low Sunlight": 7,
        "Healthy": 8,
    }
    merged["priority"] = merged["recommendation"].map(priority_map).fillna(9).astype(int)

    print("Saving processed_output.csv...")
    merged.to_csv("processed_output.csv", index=False)

    # Save model metrics
    importances = dict(zip(features, [float(x) for x in model.feature_importances_]))
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": len(merged),
        "feature_importances": importances,
        "features": features,
    }
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. {len(merged):,} records exported.")
    print("Recommendation summary:")
    print(merged["recommendation"].value_counts().to_string())


if __name__ == "__main__":
    main()
