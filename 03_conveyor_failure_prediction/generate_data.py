from pathlib import Path
import numpy as np
import pandas as pd
Path('data').mkdir(exist_ok=True)
idx = 2
rng = np.random.default_rng(102)
n = 240
temp = rng.integers(45, 101, n)
vib = np.round(rng.uniform(1, 12, n), 2)
hours = rng.integers(10, 1400, n)
probability = 1 / (1 + np.exp(-(-6 + 0.05 * (temp - 40) + 0.4 * vib + 0.0015 * hours)))
df = pd.DataFrame(dict(conveyor_id=np.arange(1, n + 1), temperature_c=temp, vibration_mm_s=vib, hours_since_service=hours, stop_next_24h=rng.binomial(1, probability)))
df.to_csv('data/synthetic_data.csv', index=False)
