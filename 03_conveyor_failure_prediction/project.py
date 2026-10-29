# Run from this project folder. Synthetic data only; AI-assisted educational work.
# %% 1. Read the inspection examples
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score

df = pd.read_csv("data/synthetic_data.csv")
print(df.head().to_string(index=False))

# %% 2. Inspect the sensor data with Pandas
print(df[["temperature_c", "vibration_mm_s", "hours_since_service"]].describe().round(2).to_string())
summary = df.groupby("stop_next_24h")[["temperature_c", "vibration_mm_s"]].mean()
print(summary.round(2).to_string())

# %% 3. Hold out independent conveyors
X = df[["temperature_c", "vibration_mm_s", "hours_since_service"]]
y = df["stop_next_24h"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# %% 4. Put different sensor units on a comparable scale
# Learn the scaling from training data only; reuse it on test data.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% 5. Train a model that returns a probability
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
probability = model.predict_proba(X_test_scaled)[:, 1]
predicted = (probability >= 0.50).astype(int)

# %% 6. Compare probability errors and inspect alerts
baseline = [y_train.mean()] * len(y_test)
print(f"Model Brier score: {brier_score_loss(y_test, probability):.3f}")
print(f"Baseline Brier score: {brier_score_loss(y_test, baseline):.3f}")
print(f"Alert precision: {precision_score(y_test, predicted, zero_division=0):.3f}")
print(f"Stoppage recall: {recall_score(y_test, predicted, zero_division=0):.3f}")
print("Brier score measures probability error; lower is better.")
results = pd.DataFrame({"conveyor_id": df.loc[X_test.index, "conveyor_id"], "observed_stop": y_test, "risk": probability})
print(results.sort_values("risk", ascending=False).head().round(3).to_string(index=False))

# %% 7. Estimate risk for one new inspection
new_inspection = pd.DataFrame({"temperature_c": [85], "vibration_mm_s": [8], "hours_since_service": [900]})
new_scaled = scaler.transform(new_inspection)
risk = model.predict_proba(new_scaled)[0, 1]
print(f"Illustrative next-24-hour stoppage risk: {risk:.1%}")
results["risk"].plot.hist(bins=10, title="Synthetic held-out conveyor risks")
plt.xlabel("Predicted probability")
plt.tight_layout()
plt.savefig("results/evaluation.png")
