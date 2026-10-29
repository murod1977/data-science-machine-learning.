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
CONFIG = {'slug': '02_schedule_delay_risk', 'title': 'Construction Schedule Delay Risk', 'kind': 'classification', 'target': 'delayed_over_7_days', 'unit': 'probability', 'group': 'project_id', 'seed': 42, 'question': 'At work-package kickoff, which activities are at risk of finishing more than seven days late?', 'decision': 'Flag activities for planner review, alternative sourcing and weather contingency discussion.', 'availability': 'Predictors are kickoff estimates and forecasts. Actual duration and realised weather are unavailable to the model.', 'features': {'planned_days': 'Planned package duration, days', 'crew_size': 'Assigned workers at kickoff', 'supply_risk': 'Pre-start supply risk score, 0-1', 'forecast_rain_days': 'Rain days forecast within the planning window', 'complexity': 'Engineering complexity score, 1-5'}, 'limitation': 'The binary outcome is sampled from assumed probabilities. Forecast quality, critical-path dependencies and intervention effects are not modelled.'}
BASE = Path(__file__).resolve().parent
FEATURES = list(CONFIG['features'])

def generate_data(seed=CONFIG['seed']):
    """Expose the assumptions instead of presenting synthetic relationships as facts."""
    rng = np.random.default_rng(seed)
    n = 1920
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    slug = CONFIG['slug']
    groups = np.repeat(np.arange(240), 8)
    days = rng.integers(10, 91, n)
    crew = rng.integers(8, 65, n)
    supply = rng.uniform(0, 1, n)
    rain = rng.integers(0, 16, n)
    complexity = rng.integers(1, 6, n)
    p = sigmoid(-3.5 + 2.6 * supply + 0.12 * rain + 0.4 * complexity - 0.025 * crew + 0.006 * days + rng.normal(0, 0.35, 240)[groups])
    data = pd.DataFrame(dict(project_id=groups, planned_days=days, crew_size=crew, supply_risk=supply, forecast_rain_days=rain, complexity=complexity, delayed_over_7_days=rng.binomial(1, p)))
    for col in FEATURES:
        data.loc[rng.random(n) < 0.015, col] = np.nan
    return data

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
