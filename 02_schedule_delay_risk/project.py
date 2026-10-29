# Run from this project folder. Synthetic data only; AI-assisted educational work.
# %% 1. Read the task records
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/synthetic_data.csv")
print(df.head())
df.info()
print(df.describe().round(2))
print("Missing values:")
print(df.isna().sum())

# %% 2. Find delays and count them with Pandas
df["delay_days"] = df["actual_days"] - df["planned_days"]
df["delayed"] = (df["delay_days"] > 3).astype(int)
summary = df.groupby("material_ready")["delayed"].agg(["count", "mean"])
print(summary.round(3).to_string())
print("Delayed by more than 3 days:", df["delayed"].sum())
sns.countplot(data=df, x="material_ready", hue="delayed")
plt.title("Historical synthetic delays and material readiness")
plt.tight_layout()
plt.savefig("results/pandas_exploration.png")
plt.close()

# %% 3. Split independent tasks into training and testing
# These inputs are known or forecast at kickoff, before the actual finish.
from sklearn.model_selection import train_test_split
X = df[["material_ready", "rain_forecast_days", "crew_size"]]
y = df["delayed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# %% 4. Train a tree with only two levels
from sklearn.tree import DecisionTreeClassifier, plot_tree
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# %% 5. Check alerts and compare with a simple baseline
# The baseline always predicts the most common training outcome.
from sklearn.metrics import accuracy_score, precision_score, recall_score
baseline = [y_train.mode()[0]] * len(y_test)
print(f"Tree accuracy: {accuracy_score(y_test, predicted):.3f}")
print(f"Baseline accuracy: {accuracy_score(y_test, baseline):.3f}")
print(f"Alert precision: {precision_score(y_test, predicted, zero_division=0):.3f}")
print(f"Delay recall: {recall_score(y_test, predicted, zero_division=0):.3f}")

# %% 6. Draw the actual decision rules
plt.figure(figsize=(10, 5))
plot_tree(model, feature_names=list(X.columns), class_names=["On time / small delay", "Delay > 3 days"], filled=True, rounded=True)
plt.title("Two-level decision tree | synthetic tasks")
plt.tight_layout()
plt.savefig("results/evaluation.png")
new_task = pd.DataFrame({"material_ready": [0], "rain_forecast_days": [5], "crew_size": [12]})
print("New task delay alert:", int(model.predict(new_task)[0]))
