# Construction Data Science and Machine Learning Portfolio

**Murod Rustamov | Educational projects | Synthetic data only | AI-assisted implementation**

Three reproducible demonstrations of how data analysis and machine learning could support construction decisions. These are learning prototypes, not company deployments. No Webuild data, branding or confidential information is included. Results measure synthetic assumptions rather than real-world performance.

| Project | Decision supported | Method |
|---|---|---|
| [Construction cost prediction](01_construction_cost_prediction/) | Review budgets and contingency assumptions | Regression; mean and planned-budget baselines |
| [Schedule delay risk](02_schedule_delay_risk/) | Prioritise planner review | Binary classification; held-out projects |
| [Conveyor stoppage risk](03_conveyor_failure_prediction/) | Prioritise conveyor inspections | Binary classification; held-out conveyors |

Each directory is self-contained: read its README, install its requirements, and run `python project.py`. Data, predictions, metrics and evaluation plots are committed for inspection without running code. Notebooks show the full executed Pandas and scikit-learn analysis, including tables and charts; Jupyter is optional and must be installed separately.

## Responsible interpretation
AI assisted implementation and documentation. The candidate should review and explain each workflow before presenting it as portfolio work. No production ML engineering experience, realised savings, safety validation or independent development is implied. Predictions require human review. A real pilot would need approved historical data, temporal validation, monitoring and business ownership.
