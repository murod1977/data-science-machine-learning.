from pathlib import Path
import numpy as np
import pandas as pd
Path('data').mkdir(exist_ok=True)
idx = 0
rng = np.random.default_rng(100)
n = 120
kind = rng.choice(['Concrete', 'Excavation', 'Rockfill'], n)
qty = rng.integers(100, 1500, n)
rate = pd.Series(kind).map({'Concrete': 110, 'Excavation': 18, 'Rockfill': 32}).to_numpy()
complexity = rng.integers(1, 6, n)
actual = np.round(qty * rate * 1.04 + complexity * 1800 + rng.normal(0, 3000, n), 2)
df = pd.DataFrame(dict(project_id=np.arange(1, n + 1), work_type=kind, quantity_m3=qty, unit_price_usd=rate, complexity=complexity, actual_cost_usd=actual))
df.to_csv('data/synthetic_data.csv', index=False)
