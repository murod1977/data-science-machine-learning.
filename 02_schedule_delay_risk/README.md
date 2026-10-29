# Construction Schedule Delay Risk

**Educational portfolio project | Synthetic data only | AI-assisted implementation**

Prepared for Murod Rustamov's construction analytics learning portfolio. This is an independent demonstration, not a Webuild project or deployment. No company records, confidential data, or measured business savings are used.

## Business question
At work-package kickoff, which activities are at risk of finishing more than seven days late?

## Intended decision
Flag activities for planner review, alternative sourcing and weather contingency discussion.

## Reproduce
Use Python 3.11 or newer in a virtual environment. From this project folder:
```bash
python -m pip install -r requirements.txt
python project.py
```
This regenerates the 1,920-row CSV, held-out predictions, metrics, plot and permutation importance. Fixed seeds make the run reproducible in the recorded environment. The accompanying notebook runs the same source of truth. No network, API key or paid service is required after installing dependencies.

## Data and prediction time
Predictors are kickoff estimates and forecasts. Actual duration and realised weather are unavailable to the model.

| Input | Meaning |
|---|---|
| `planned_days` | Planned package duration, days |
| `crew_size` | Assigned workers at kickoff |
| `supply_risk` | Pre-start supply risk score, 0-1 |
| `forecast_rain_days` | Rain days forecast within the planning window |
| `complexity` | Engineering complexity score, 1-5 |

Target: `delayed_over_7_days`. Split identifier: `project_id`; it is excluded from model inputs. The complete generator and its assumed outcome relationships are visible in `generate_data()` in `project.py`. Each generator injects approximately 1.5% missing predictor values, handled inside a training-only imputation pipeline. Labels are generated, not collected. USD is a demonstration currency.

## Evaluation design
75% of groups train the model; 25% are held out with a fixed seed. Projects, assets or suppliers never overlap across the split. This measures transfer to unseen synthetic groups, not future-time performance. Classification labels are sampled from probabilities, so perfect predictions are neither expected nor desirable. The classifier threshold is fixed at 0.50 in advance; it is not selected on the test set. No hyperparameter search was conducted. For real adoption, add temporal validation, uncertainty estimates, calibration review, drift monitoring and an operational pilot.

## Recorded held-out results
These numbers describe the synthetic generator only; they do not establish real-site accuracy.

| Metric | Value |
|---|---:|
| test_prevalence | 0.3417 |
| average_precision | 0.5367 |
| baseline_average_precision | 0.3417 |
| roc_auc | 0.6912 |
| brier_score | 0.2039 |
| baseline_brier_score | 0.2256 |
| precision_at_05 | 0.5748 |
| recall_at_05 | 0.4451 |

Regression compares random forest with a training-mean baseline and the original planned budget. Classification compares random forest with the training prevalence baseline; average precision and Brier score are more informative here than accuracy alone. Lower MAE/Brier is better; higher average precision/ROC AUC is better. Permutation importance describes model sensitivity, not causality.

![Synthetic evaluation](results/evaluation.png)

## Limitations and next step
The binary outcome is sampled from assumed probabilities. Forecast quality, critical-path dependencies and intervention effects are not modelled.

Synthetic assumptions simplify reality and can favour the fitted model. Before use, obtain approved real historical data, audit data availability at prediction time, and agree an evaluation protocol with planners, engineers or procurement staff. No production integration or validated cost reduction is claimed.

## Interview explanation
1. Explain the operational question and when the prediction is made.
2. Show how the synthetic data and labels are produced.
3. Explain why group overlap and post-outcome inputs would cause leakage.
4. Compare the held-out model with the simple baseline.
5. Describe one decision the result could support and what it cannot prove.

## Files
`project.py`: generator, training and evaluation. `data/`: generated CSV. `results/`: saved evidence. `walkthrough.ipynb`: interactive entry point.

## Full analysis notebook
Open [walkthrough.ipynb](walkthrough.ipynb) for the complete executed workflow: synthetic generation, Pandas checks and exploration, group split, scikit-learn Pipeline, baseline comparison, metrics and interpretation. Code, tables and charts are visible directly on GitHub.
