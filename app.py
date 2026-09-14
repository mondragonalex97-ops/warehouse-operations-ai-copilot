
import os
import math
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Warehouse Operations AI Copilot", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("warehouse_operations_data.csv", parse_dates=["date"])
    return df.sort_values("date")

@st.cache_resource
def train_model(df):
    work = df.copy()
    work["dow"] = work["date"].dt.dayofweek
    work["month"] = work["date"].dt.month
    work["day_num"] = np.arange(len(work))
    work["lag_1_units"] = work["units"].shift(1)
    work["lag_7_units"] = work["units"].shift(7)
    work["rolling_7_units"] = work["units"].rolling(7).mean().shift(1)
    work = work.dropna().copy()

    features = ["dow", "month", "day_num", "promo_flag", "lag_1_units", "lag_7_units", "rolling_7_units"]
    X = work[features]
    y = work["units"]

    split = int(len(work) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, preds)
    return model, features, mape, work

def predict_next_day(df, model):
    next_date = df["date"].max() + pd.Timedelta(days=1)
    next_row = pd.DataFrame([{
        "dow": next_date.dayofweek,
        "month": next_date.month,
        "day_num": len(df),
        "promo_flag": 0,
        "lag_1_units": df["units"].iloc[-1],
        "lag_7_units": df["units"].iloc[-7],
        "rolling_7_units": df["units"].tail(7).mean()
    }])
    pred = float(model.predict(next_row)[0])
    return next_date, pred

def staffing_recommendation(pred_units, productivity=61, shift_hours=9.5, buffer=0.07):
    base_workers = pred_units / (productivity * shift_hours)
    recommended = math.ceil(base_workers * (1 + buffer))
    return recommended

def overtime_risk(pred_units, workers, productivity=61, shift_hours=9.5):
    capacity = workers * productivity * shift_hours
    ratio = pred_units / max(capacity, 1)
    if ratio >= 1.08:
        return "High", ratio
    elif ratio >= 0.97:
        return "Medium", ratio
    return "Low", ratio

def management_summary(df, pred_units, recommended_workers, risk):
    latest = df.iloc[-1]
    recent = df.tail(7)
    avg_units = recent["units"].mean()
    backlog_change = recent["backlog_units"].iloc[-1] - recent["backlog_units"].iloc[0]
    on_time_avg = recent["on_time_ship_pct"].mean()
    prod_avg = recent["units_per_labor_hour"].mean()

    direction = "above" if pred_units > avg_units else "below"
    pct = abs(pred_units - avg_units) / avg_units * 100

    backlog_text = (
        f"Backlog increased by {abs(backlog_change):,.0f} units over the last 7 days."
        if backlog_change > 0 else
        f"Backlog decreased by {abs(backlog_change):,.0f} units over the last 7 days."
    )

    return (
        f"Tomorrow's predicted volume is {pred_units:,.0f} units, about {pct:.1f}% {direction} "
        f"the recent 7-day average. Recommended staffing is {recommended_workers} associates "
        f"based on recent productivity and a 7% staffing buffer. Overtime risk is {risk}. "
        f"{backlog_text} Recent on-time shipping averaged {on_time_avg:.1f}% and productivity "
        f"averaged {prod_avg:.1f} units per labor hour."
    )

df = load_data()
model, features, mape, model_df = train_model(df)
next_date, pred_units = predict_next_day(df, model)
recommended_workers = staffing_recommendation(pred_units)
risk, capacity_ratio = overtime_risk(pred_units, recommended_workers)

st.title("Warehouse Operations AI Copilot")
st.caption("Demand forecasting, labor planning, risk detection, and operational insights")

latest = df.iloc[-1]
recent7 = df.tail(7)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Tomorrow Forecast", f"{pred_units:,.0f} units")
c2.metric("Recommended Staffing", f"{recommended_workers} associates")
c3.metric("Overtime Risk", risk)
c4.metric("Forecast MAPE", f"{mape*100:.1f}%")

st.subheader("AI Management Summary")
st.info(management_summary(df, pred_units, recommended_workers, risk))

left, right = st.columns(2)

with left:
    st.subheader("Daily Volume")
    st.line_chart(df.set_index("date")[["units"]].tail(60))

with right:
    st.subheader("Backlog Trend")
    st.line_chart(df.set_index("date")[["backlog_units"]].tail(60))

st.subheader("Operational KPIs")
k1, k2, k3, k4 = st.columns(4)
k1.metric("7-Day Avg Productivity", f"{recent7['units_per_labor_hour'].mean():.1f} UPH")
k2.metric("7-Day Avg On-Time Ship", f"{recent7['on_time_ship_pct'].mean():.1f}%")
k3.metric("Current Backlog", f"{latest['backlog_units']:,.0f} units")
k4.metric("7-Day Overtime", f"{recent7['overtime_hours'].sum():.1f} hrs")

st.subheader("Risk Flags")
risk_flags = []
if latest["backlog_units"] > recent7["backlog_units"].mean() * 1.2:
    risk_flags.append("Backlog is materially above the recent average.")
if latest["on_time_ship_pct"] < 94:
    risk_flags.append("On-time shipping is below 94%.")
if latest["error_rate_pct"] > 1.1:
    risk_flags.append("Error rate is elevated.")
if latest["absences"] > recent7["absences"].mean() * 1.25:
    risk_flags.append("Absenteeism is above the recent average.")
if not risk_flags:
    risk_flags.append("No major operational risk flags detected today.")

for flag in risk_flags:
    st.write("•", flag)

st.subheader("Recent Operations Data")
st.dataframe(
    df.tail(14).sort_values("date", ascending=False),
    use_container_width=True
)

st.subheader("Scenario Planner")
col1, col2, col3 = st.columns(3)
scenario_units = col1.number_input("Expected units", min_value=10000, value=int(pred_units), step=1000)
scenario_productivity = col2.number_input("Expected productivity (UPH)", min_value=30.0, value=61.0, step=1.0)
scenario_hours = col3.number_input("Shift hours", min_value=4.0, value=9.5, step=0.5)

scenario_workers = staffing_recommendation(
    scenario_units,
    productivity=scenario_productivity,
    shift_hours=scenario_hours
)
scenario_risk, ratio = overtime_risk(
    scenario_units,
    scenario_workers,
    productivity=scenario_productivity,
    shift_hours=scenario_hours
)

st.success(
    f"Recommended staffing: {scenario_workers} associates | "
    f"Estimated capacity utilization: {ratio*100:.1f}% | "
    f"Overtime risk: {scenario_risk}"
)

st.caption(
    "Portfolio project using synthetic warehouse data. Built for educational and demonstration purposes."
)
