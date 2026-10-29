# Conveyor Belt Stoppage Risk

**Educational prototype | Synthetic data only | AI-assisted implementation**

## Business question
Which conveyors should an engineer inspect first because of possible unplanned stoppage in the next 24 hours? This independent construction-material handling example uses no company data. It is not deployed, safety-validated or endorsed by Webuild.

## Decision and potential value
Rank inspection candidates using predicted risk. A qualified engineer decides what action is appropriate. Potential value is better maintenance prioritisation and reduced disruption; no real downtime reduction or financial savings have been measured.

## Reproduce
Use Python 3.11 or newer, preferably in a virtual environment. From this folder:
```bash
python -m pip install -r requirements.txt
python project.py
```
Jupyter is optional and installed separately. `walkthrough.ipynb` calls the same training code. No API key, external dataset or paid service is needed after dependency installation.

## Data and assumptions
The generator creates 1,920 independent fictional conveyor snapshots, one per conveyor. Each input summarises the previous hour; the target is an unplanned stoppage within the following 24 hours. This is a cross-sectional demonstration, not a longitudinal simulation or a remaining-useful-life model. Equipment specifications and sensor thresholds are illustrative. There is no inference about the user's actual conveyors or transported material.

| Predictor | Definition |
|---|---|
| `vibration_mm_s` | Prior-hour drive vibration, mm/s |
| `bearing_temperature_c` | Prior-hour bearing temperature, Celsius |
| `motor_current_a` | Prior-hour motor current, amperes |
| `belt_speed_m_s` | Prior-hour belt speed, m/s |
| `throughput_tph` | Prior-hour transported mass, tonnes/hour |
| `hours_since_maintenance` | Operating hours since maintenance |

`conveyor_id` identifies the equipment and is never a predictor. The binary target is `unplanned_stop_next_24h`. Labels are sampled from the explicit assumed logistic probability in `generate_data()`. Coefficients are educational assumptions, not estimated engineering failure laws. Approximately 1.5% of predictor values are missing.

## Validation and leakage controls
Train on 75% of conveyor IDs, test on 25% unseen IDs. Each ID appears once. Median imputation is fitted only on training data. Target and ID are excluded from features. A random forest is compared with a training-prevalence baseline. Seeds are fixed. There is no test-set tuning; the decision threshold is fixed at 0.50 before evaluation. Average precision summarises ranking quality, Brier score assesses probability error, precision measures alert reliability and recall measures the share of stoppages found. These are single-split estimates without confidence intervals.

## Recorded synthetic results
| Metric | Value |
|---|---:|
| test_prevalence | 0.2500 |
| average_precision | 0.6491 |
| baseline_average_precision | 0.2500 |
| roc_auc | 0.8409 |
| brier_score | 0.1338 |
| baseline_brier_score | 0.1875 |
| precision_at_05 | 0.7077 |
| recall_at_05 | 0.3833 |

![Held-out synthetic evaluation](results/evaluation.png)

Full predictions, split assignments and permutation importance are in `results/`. Importance is not causality. The model learns the imposed synthetic relationships; these results do not establish accuracy on a real conveyor system. Missed events and false alarms remain.

## What a real pilot needs
Approved time-stamped sensor and event data, confirmed failure definitions, alignment of features before the prediction time, sufficient exposure periods and engineering input. For repeated snapshots, keep equipment and overlapping outcome windows separated and use forward-time evaluation with a purge gap. Compare with existing alarms, review probability calibration, agree inspection costs and operational thresholds, and monitor performance. This prototype must not operate or shut down equipment.

## Interview walkthrough
Explain the 24-hour target; distinguish inputs from future outcomes; show the synthetic generator; explain held-out conveyors and training-only preprocessing; compare average precision and Brier score with the baseline; explain why the model requires validation before operational use.
