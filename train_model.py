
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv("warehouse_operations_data.csv", parse_dates=["date"])
df = df.sort_values("date").copy()

df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_num"] = np.arange(len(df))
df["lag_1_units"] = df["units"].shift(1)
df["lag_7_units"] = df["units"].shift(7)
df["rolling_7_units"] = df["units"].rolling(7).mean().shift(1)
df = df.dropna()

features = [
    "dow",
    "month",
    "day_num",
    "promo_flag",
    "lag_1_units",
    "lag_7_units",
    "rolling_7_units",
]

X = df[features]
y = df["units"]

split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    min_samples_leaf=2
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predictions)

print(f"Test MAPE: {mape*100:.2f}%")
print("\nFeature importance:")
for feature, importance in sorted(
    zip(features, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature}: {importance:.3f}")
