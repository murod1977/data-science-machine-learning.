from pathlib import Path
import numpy as np
import pandas as pd
Path('data').mkdir(exist_ok=True)
idx = 1
rng = np.random.default_rng(101)
n = 160
planned = rng.integers(15, 61, n)
ready = rng.integers(0, 2, n)
rain = rng.integers(0, 9, n)
crew = rng.integers(8, 31, n)
delay = np.rint((1 - ready) * 5 + rain * 0.7 - crew * 0.1 + rng.normal(0, 2, n)).astype(int)
df = pd.DataFrame(dict(task_id=np.arange(1, n + 1), planned_days=planned, actual_days=planned + delay, material_ready=ready, rain_forecast_days=rain, crew_size=crew))
df.to_csv('data/synthetic_data.csv', index=False)
