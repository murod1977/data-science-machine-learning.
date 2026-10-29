# Run from this project folder. Synthetic data only; AI-assisted educational work.
# %% 1. Read the historical examples
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/synthetic_data.csv")
print(df.head())
df.info()
print(df.describe().round(2))
print("Missing values:")
print(df.isna().sum())

# %% 2. Calculate costs with Pandas
df["planned_cost"] = df["quantity_m3"] * df["unit_price_usd"]
df["overrun"] = df["actual_cost_usd"] - df["planned_cost"]
df["overrun_percent"] = df["overrun"] / df["planned_cost"] * 100
summary = df.groupby("work_type")[["planned_cost", "actual_cost_usd"]].sum()
print(summary.round(2).to_string())
print("Packages above budget:", len(df[df["overrun"] > 0]))
sns.scatterplot(data=df, x="quantity_m3", y="actual_cost_usd", hue="work_type")
plt.title("Historical synthetic costs by work type")
plt.tight_layout()
plt.savefig("results/pandas_exploration.png")
plt.close()

# %% 3. Keep some projects for the final check
# Only planned information is used as input, never actual cost or overrun.
from sklearn.model_selection import train_test_split
X = df[["planned_cost", "complexity"]]
y = df["actual_cost_usd"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("Training:", len(X_train), "Testing:", len(X_test))

# %% 4. Fit a straight-line cost model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)

# %% 5. Compare the model with the original budget
from sklearn.metrics import mean_absolute_error
model_error = mean_absolute_error(y_test, predicted)
budget_error = mean_absolute_error(y_test, X_test["planned_cost"])
print(f"Model MAE: {model_error:.2f} USD")
print(f"Budget MAE: {budget_error:.2f} USD")
print("MAE means average absolute prediction error; lower is better.")

# %% 6. Display the comparison and estimate one new package
comparison = pd.DataFrame({"actual": y_test, "predicted": predicted})
print(comparison.head().round(2).to_string(index=False))
comparison.head(8).plot.bar(title="Synthetic test projects: cost in USD", rot=0)
plt.ylabel("USD")
plt.tight_layout()
plt.savefig("results/evaluation.png")
new_package = pd.DataFrame({"planned_cost": [50000], "complexity": [3]})
estimate = model.predict(new_package)[0]
print(f"Example final cost estimate: {estimate:.2f} USD")
