
# Warehouse Operations AI Copilot

A portfolio project that demonstrates how machine learning and analytics can support warehouse labor planning and operational decision-making.

## What the project does

The application uses historical warehouse operating data to:

- Forecast next-day unit volume
- Recommend staffing levels
- Estimate overtime risk
- Track backlog, productivity, service level, and error rate
- Flag operational risks
- Generate a management-style operations summary
- Let managers test labor scenarios in an interactive planner

## Why this matters

Warehouse leaders constantly balance customer demand, staffing, productivity, backlog, service level, and overtime. This project turns those operational signals into clear recommendations.

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest regression
- Streamlit
- Machine learning forecasting
- Operations analytics

## Dataset

`warehouse_operations_data.csv` contains 240 days of synthetic warehouse operating data.

The data is intentionally synthetic so the project can be shared publicly without exposing proprietary employer information.

Fields include:

- Orders
- Units
- Planned workers
- Actual workers
- Absences
- Labor hours
- Units per labor hour
- Backlog
- Overtime
- On-time shipping percentage
- Error rate
- Promotion flag

## Run locally

1. Install Python 3.10+
2. Open a terminal in this folder
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
streamlit run app.py
```

## Machine learning approach

The forecasting model is a Random Forest Regressor using:

- Day of week
- Month
- Time trend
- Promotion flag
- Previous-day volume
- Previous-week volume
- Seven-day rolling average

The model is evaluated using Mean Absolute Percentage Error (MAPE).

## Business logic

Staffing is calculated from:

Expected Units / (Expected Units per Labor Hour × Shift Length)

A 7% staffing buffer is added to account for normal operational variation.

Overtime risk is classified as Low, Medium, or High based on forecast demand relative to modeled labor capacity.
