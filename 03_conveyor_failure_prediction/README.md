# Conveyor Stoppage Risk with Logistic Regression

**Beginner/intermediate educational project | Synthetic data | AI-assisted implementation**

## Question
Estimate a fictional conveyor's unplanned stoppage risk in the following 24 hours from three prior-inspection measurements.

## Start here
Open [walkthrough.ipynb](walkthrough.ipynb): short Python cells with explanations in English and Russian, saved outputs and a chart. The entire analysis is also visible in [project.py](project.py), approximately 37 non-comment lines. No custom classes, pipelines or random forests.

## Run
Python 3.11 or newer; preferably a virtual environment. From this project folder:
```bash
python -m pip install -r requirements.txt
python project.py
```
Jupyter is optional and installed separately. The checked-in CSV is ready to read. To regenerate it, run `python generate_data.py` from this folder.

## Data dictionary and assumptions
temperature_c: bearing temperature in Celsius. vibration_mm_s: drive vibration in mm/s. hours_since_service: operating hours since maintenance. Target stop_next_24h: 0/1. One row per conveyor; no repeated time windows.

There are 240 independent examples, one per fictional project/task/conveyor. No repeated entities or overlapping outcome windows. Values are complete, so imputation is unnecessary. The generator's fixed formulas and random noise are visible in `generate_data.py`; these are teaching assumptions, not calibrated engineering laws. No real company data are used. Descriptive summaries use historical outcomes; predictors exclude those outcomes. Model settings are fixed in advance and not chosen by trying alternatives on the test set. The 25% holdout is small; no confidence intervals, cross-validation or probability calibration are claimed.

## Saved results
```json
{
  "brier_score": 0.18765560632096343,
  "baseline_brier_score": 0.23234567901234565,
  "precision": 0.625,
  "recall": 0.45454545454545453,
  "synthetic_only": true,
  "train_rows": 180,
  "test_rows": 60
}
```
![Synthetic evaluation](results/evaluation.png)

`results/summary.txt` contains the printed analysis; `results/test_predictions.csv` contains held-out predictions. These metrics describe only synthetic examples. The simple baseline helps assess whether the model adds value within this exercise; it does not demonstrate commercial savings.

## Interview and next step
Explain the input columns, target, train/test split, model and baseline before presenting the work. AI assisted code preparation; do not claim independent implementation or professional deployment. Practise changing one small input and explaining the output. Real adoption would require approved historical data, forward-time evaluation, engineering/planning review and monitoring.
