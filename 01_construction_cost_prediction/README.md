# Construction Cost Prediction

**Educational portfolio project | Synthetic data only | AI-assisted implementation**

Prepared for Murod Rustamov's construction analytics learning portfolio. This is an independent demonstration, not a Webuild project or deployment. No company records, confidential data, or measured business savings are used.

## Business question
At tender review, which work packages may exceed their initial budget?

## Intended decision
Prioritise estimator review and contingency discussion. Predictions are advisory; they do not authorise budgets.

## Reproduce
Use Python 3.11 or newer in a virtual environment. From this project folder:
```bash
python -m pip install -r requirements.txt
python project.py
```
This regenerates the 1,920-row CSV, held-out predictions, metrics, plot and permutation importance. Fixed seeds make the run reproducible in the recorded environment. The accompanying notebook runs the same source of truth. No network, API key or paid service is required after installing dependencies.

## Data and prediction time
All predictors describe information available at tender review. Final cost is an outcome, never an input.

| Input | Meaning |
|---|---|
| `planned_cost_usd` | Initial package budget, USD |
| `quantity_m3` | Planned work volume, cubic metres |
| `complexity` | Engineering complexity score, 1-5 |
| `market_index` | Assumed material price index at estimation |
| `remote_site` | Remote site indicator, 0/1 |

Target: `final_cost_usd`. Split identifier: `project_id`; it is excluded from model inputs. The complete generator and its assumed outcome relationships are visible in `generate_data()` in `project.py`. Each generator injects approximately 1.5% missing predictor values, handled inside a training-only imputation pipeline. Labels are generated, not collected. USD is a demonstration currency.

## Evaluation design
75% of groups train the model; 25% are held out with a fixed seed. Projects, assets or suppliers never overlap across the split. This measures transfer to unseen synthetic groups, not future-time performance. Classification labels are sampled from probabilities, so perfect predictions are neither expected nor desirable. The classifier threshold is fixed at 0.50 in advance; it is not selected on the test set. No hyperparameter search was conducted. For real adoption, add temporal validation, uncertainty estimates, calibration review, drift monitoring and an operational pilot.

## Recorded held-out results
These numbers describe the synthetic generator only; they do not establish real-site accuracy.

| Metric | Value |
|---|---:|
| model_mae_usd | 23077.7181 |
| baseline_mae_usd | 158465.4337 |
| budget_mae_usd | 32376.4153 |
| r2 | 0.9610 |

Regression compares random forest with a training-mean baseline and the original planned budget. Classification compares random forest with the training prevalence baseline; average precision and Brier score are more informative here than accuracy alone. Lower MAE/Brier is better; higher average precision/ROC AUC is better. Permutation importance describes model sensitivity, not causality.

![Synthetic evaluation](results/evaluation.png)

## Limitations and next step
Final costs follow a deliberately simplified synthetic formula. The model does not estimate commercial savings or replace quantity surveying.

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
