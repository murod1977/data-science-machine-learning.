# Construction Data Science Learning Portfolio

**Murod Rustamov | Beginner/intermediate projects | Synthetic data | AI-assisted implementation**

Three small learning projects with different business questions and model types. The priority is understandable code that the learner can practise and explain. Educational projects using synthetic data; results require validation on real operational data.

| Project | Pandas practice | Model | Output |
|---|---|---|---|
| [Cost analysis](01_construction_cost_prediction/walkthrough.ipynb) | Calculate costs and overruns; group by work type | Linear regression | Estimated cost in USD |
| [Schedule delays](02_schedule_delay_risk/walkthrough.ipynb) | Calculate delay days; count late tasks | Two-level decision tree | Delay alert and visible rules |
| [Conveyor inspections](03_conveyor_failure_prediction/walkthrough.ipynb) | Describe sensors; compare historical averages | Logistic regression | Next-24-hour stoppage probability |

Each folder contains a short `project.py`, an executed notebook with English/Russian explanations, a synthetic CSV, a separate reproducible generator, one plot and saved test results. Open the notebook links above to see the code directly. These are deliberately simple teaching examples, not production systems.

## Workflow
1. Explore the CSV with Pandas: `head()`, `info()`, `describe()` and `isna().sum()`.
2. Calculate and summarise with Pandas; draw a Matplotlib/Seaborn chart.
3. Train and check one simple scikit-learn model.

## Run all projects
Python 3.11 or newer. From the repository root:
```bash
python -m pip install -r requirements.txt
python run_all.py
```
Alternatively, enter one project folder and run `python project.py`. Jupyter is optional and installed separately.

## Limits
Small, independent synthetic samples make the exercises manageable. Metrics measure assumed generator relationships, not real construction accuracy. A real pilot requires approved historical data, temporal evaluation and operational ownership. No confidential records, resumes or certificates are included.
