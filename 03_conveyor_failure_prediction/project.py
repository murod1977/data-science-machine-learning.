"""AI-assisted educational demonstration. Synthetic data only; no Webuild data."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_absolute_error, r2_score, average_precision_score, roc_auc_score, precision_score, recall_score, brier_score_loss, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
CONFIG = {'slug': '03_conveyor_failure_prediction', 'title': 'Conveyor Belt Stoppage Risk', 'kind': 'classification', 'target': 'unplanned_stop_next_24h', 'group': 'conveyor_id', 'seed': 73, 'features': {'vibration_mm_s': 'Prior-hour drive vibration, mm/s', 'bearing_temperature_c': 'Prior-hour bearing temperature, Celsius', 'motor_current_a': 'Prior-hour motor current, amperes', 'belt_speed_m_s': 'Prior-hour belt speed, m/s', 'throughput_tph': 'Prior-hour transported mass, tonnes/hour', 'hours_since_maintenance': 'Operating hours since maintenance'}}
BASE = Path(__file__).resolve().parent
FEATURES = list(CONFIG['features'])

def generate_data(seed=CONFIG['seed']):
    """One independent snapshot per fictional conveyor, not a real sensor history.

    The logistic coefficients are illustrative assumptions, not engineering limits.
    All measurements precede the next-24-hour label window.
    """
    rng = np.random.default_rng(seed)
    n = 1920
    vibration = rng.uniform(1, 12, n)
    temp = rng.uniform(40, 105, n)
    throughput = rng.uniform(150, 1600, n)
    speed = rng.uniform(1.0, 4.0, n)
    current = np.clip(40 + 0.065 * throughput + rng.normal(0, 15, n), 20, 190)
    hours = rng.uniform(10, 1400, n)
    logit = -5.6 + 0.3 * vibration + 0.04 * (temp - 55) + 0.0015 * hours + 0.012 * (current - 80) + 0.0005 * (throughput - 700) + 0.35 * (speed < 1.5)
    probability = 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(dict(conveyor_id=np.arange(n), vibration_mm_s=vibration, bearing_temperature_c=temp, motor_current_a=current, belt_speed_m_s=speed, throughput_tph=throughput, hours_since_maintenance=hours, unplanned_stop_next_24h=rng.binomial(1, probability)))
    for column in FEATURES:
        df.loc[rng.random(n) < 0.015, column] = np.nan
    return df

def run():
    out = BASE / 'results'
    out.mkdir(exist_ok=True)
    (BASE / 'data').mkdir(exist_ok=True)
    df = generate_data()
    df.to_csv(BASE / 'data/synthetic_data.csv', index=False)
    group = CONFIG['group']
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=19).split(df, groups=df[group]))
    assert set(df.iloc[train][group]).isdisjoint(set(df.iloc[test][group]))
    Xtr, Xte = (df.iloc[train][FEATURES], df.iloc[test][FEATURES])
    ytr, yte = (df.iloc[train][CONFIG['target']], df.iloc[test][CONFIG['target']])
    assert CONFIG['target'] not in FEATURES and group not in FEATURES
    regression = CONFIG['kind'] == 'regression'
    estimator = RandomForestRegressor(n_estimators=160, min_samples_leaf=8, random_state=23, n_jobs=1) if regression else RandomForestClassifier(n_estimators=160, min_samples_leaf=14, random_state=23, n_jobs=1)
    model = make_pipeline(SimpleImputer(strategy='median'), estimator)
    model.fit(Xtr, ytr)
    metrics = {'synthetic_only': True, 'train_rows': len(train), 'test_rows': len(test), 'split': '25% held-out groups; seed 19', 'seed': CONFIG['seed']}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if regression:
        pred = model.predict(Xte)
        baseline = DummyRegressor(strategy='mean').fit(Xtr, ytr).predict(Xte)
        metrics.update(model_mae_usd=float(mean_absolute_error(yte, pred)), baseline_mae_usd=float(mean_absolute_error(yte, baseline)), budget_mae_usd=float(mean_absolute_error(yte, Xte.planned_cost_usd.fillna(Xtr.planned_cost_usd.median()))), r2=float(r2_score(yte, pred)))
        ax.scatter(yte, pred, s=12, alpha=0.5, color='#147d92')
        lo, hi = (float(yte.min()), float(yte.max()))
        ax.plot([lo, hi], [lo, hi], color='#c65e34', ls='--')
        ax.set(xlabel='Observed synthetic final cost (USD)', ylabel='Predicted final cost (USD)')
        scoring = 'neg_mean_absolute_error'
    else:
        pred = model.predict_proba(Xte)[:, 1]
        baseline = DummyClassifier(strategy='prior').fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        label = pred >= 0.5
        assert ytr.nunique() == 2 and yte.nunique() == 2
        metrics.update(test_prevalence=float(yte.mean()), average_precision=float(average_precision_score(yte, pred)), baseline_average_precision=float(average_precision_score(yte, baseline)), roc_auc=float(roc_auc_score(yte, pred)), brier_score=float(brier_score_loss(yte, pred)), baseline_brier_score=float(brier_score_loss(yte, baseline)), precision_at_05=float(precision_score(yte, label, zero_division=0)), recall_at_05=float(recall_score(yte, label, zero_division=0)))
        ConfusionMatrixDisplay.from_predictions(yte, label, ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Fixed threshold: 0.50 (not optimised on test data)')
        scoring = 'average_precision'
    assert np.isfinite(pred).all()
    fig.suptitle(CONFIG['title'] + ' | SYNTHETIC DATA', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'evaluation.png', dpi=160)
    plt.close(fig)
    perm = permutation_importance(model, Xte, yte, scoring=scoring, n_repeats=5, random_state=29)
    importance = pd.DataFrame({'feature': FEATURES, 'importance': perm.importances_mean, 'std': perm.importances_std}).sort_values('importance', ascending=False)
    importance.to_csv(out / 'permutation_importance.csv', index=False)
    predictions = df.iloc[test][[group, CONFIG['target']]].copy()
    predictions['prediction'] = pred
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame({'row_id': np.r_[train, test], 'split': ['train'] * len(train) + ['test'] * len(test)}).to_csv(out / 'split.csv', index=False)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))
    return metrics
if __name__ == '__main__':
    run()
