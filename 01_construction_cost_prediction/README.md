# Construction Cost Analysis and Prediction

**Beginner/intermediate educational project | Synthetic data | AI-assisted implementation**

## Question
Estimate final package cost from the planned budget and complexity. Use this as an estimator-review demonstration.

## Start here
Open [walkthrough.ipynb](walkthrough.ipynb): short Python cells with explanations in English and Russian, saved outputs and a chart. The entire analysis is also visible in [project.py](project.py), approximately 34 non-comment lines. No custom classes, pipelines or random forests.

## Run
Python 3.11 or newer; preferably a virtual environment. From this project folder:
```bash
python -m pip install -r requirements.txt
python project.py
```
Jupyter is optional and installed separately. The checked-in CSV is ready to read. To regenerate it, run `python generate_data.py` from this folder.

## Data dictionary and assumptions
planned_cost = quantity_m3 × unit_price_usd. Target: actual_cost_usd. Complexity is a fictional pre-start score from 1 to 5. Work type is used for descriptive grouping only.

There are 120 independent examples, one per fictional project/task/conveyor. No repeated entities or overlapping outcome windows. Values are complete, so imputation is unnecessary. The generator's fixed formulas and random noise are visible in `generate_data.py`; these are teaching assumptions, not calibrated engineering laws. No real company data are used. Descriptive summaries use historical outcomes; predictors exclude those outcomes. Model settings are fixed in advance and not chosen by trying alternatives on the test set. The 25% holdout is small; no confidence intervals, cross-validation or probability calibration are claimed.

## Saved results
```json
{
  "model_mae_usd": 2619.6750100036616,
  "planned_budget_mae_usd": 6769.9400000000005,
  "synthetic_only": true,
  "train_rows": 90,
  "test_rows": 30
}
```
![Synthetic evaluation](results/evaluation.png)

`results/summary.txt` contains the printed analysis; `results/test_predictions.csv` contains held-out predictions. These metrics describe only synthetic examples. The simple baseline helps assess whether the model adds value within this exercise; it does not demonstrate commercial savings.

## Interview and next step
Explain the input columns, target, train/test split, model and baseline before presenting the work. AI assisted code preparation; do not claim independent implementation or professional deployment. Practise changing one small input and explaining the output. Real adoption would require approved historical data, forward-time evaluation, engineering/planning review and monitoring.
