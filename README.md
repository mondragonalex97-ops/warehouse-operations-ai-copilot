
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

## Resume bullets

**Warehouse Operations AI Copilot | Python, Scikit-learn, Streamlit**

- Built a machine-learning warehouse forecasting tool using 240 days of synthetic operating data to predict daily fulfillment volume and support labor planning.
- Developed a staffing recommendation model using forecasted demand, productivity, labor hours, and capacity constraints to identify overtime risk.
- Created an interactive operations dashboard tracking backlog, on-time shipping, productivity, absenteeism, error rates, and labor scenarios.
- Automated management-style operational summaries that convert warehouse KPIs into actionable staffing and risk recommendations.

## Interview explanation

"I wanted to build something that connected supply chain operations with AI and analytics. I created a synthetic warehouse dataset and trained a machine-learning model to predict the next day's volume. I then connected that prediction to labor-capacity calculations so the tool recommends staffing and highlights overtime risk. The dashboard also monitors backlog, service levels, productivity, and absenteeism. I used synthetic data because I wanted the project to be realistic while avoiding any proprietary company information."

## Important note

This project is for portfolio and educational use. The dataset is synthetic and does not contain confidential employer data.
