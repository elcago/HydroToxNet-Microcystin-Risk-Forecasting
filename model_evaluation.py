from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    brier_score_loss,
)

from model_training import (
    ROOT,
    OUTDIR,
    CKPT_DIR,
    BATCH_SIZE,
    DEVICE,
    HORIZONS,
    TARGETS,
    SEEDS,
    ForecastDataset,
    build_model,
    load_data,
    move_batch,
    predict_batch,
)

PRED_DIR = OUTDIR / 'predictions'
METRIC_DIR = OUTDIR / 'metrics'
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 2026


def safe_auprc(y, p):
    return average_precision_score(y, p) if np.sum(y) > 0 else np.nan


def safe_auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def best_f1_threshold(y, p):
    candidates = np.unique(np.r_[0.01, np.linspace(0.05, 0.95, 91), 0.99])
    scores = [f1_score(y, p >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def calculate_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        'AUPRC': safe_auprc(y, p),
        'AUROC': safe_auroc(y, p),
        'F1': f1_score(y, pred, zero_division=0),
        'MCC': matthews_corrcoef(y, pred) if len(np.unique(y)) > 1 else 0.0,
        'Precision': precision_score(y, pred, zero_division=0),
        'Recall': recall_score(y, pred, zero_division=0),
        'Brier': brier_score_loss(y, p),
    }


def load_checkpoint(path, params):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = build_model(params, checkpoint['config'], checkpoint.get('flags', {})).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def predict_split(model, dataset, seed, split):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    rows = []

    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch)
            probs = torch.sigmoid(predict_batch(model, batch)).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            masks = batch['masks'].cpu().numpy()

            for b in range(len(raw_batch['station'])):
                station = raw_batch['station'][b]
                year = int(raw_batch['year'][b])
                week_start = raw_batch['week_start'][b]

                for hi, horizon in enumerate(HORIZONS):
                    for ci, target in enumerate(TARGETS):
                        if masks[b, hi, ci] < 0.5:
                            continue
                        rows.append({
                            'split': split,
                            'seed': seed,
                            'station': station,
                            'year': year,
                            'week_start': week_start,
                            'target': target,
                            'horizon': horizon,
                            'y_true': int(labels[b, hi, ci]),
                            'y_prob': float(probs[b, hi, ci]),
                        })

    return pd.DataFrame(rows)


def average_seed_predictions(predictions):
    keys = [
        'split', 'station', 'year', 'week_start',
        'target', 'horizon', 'y_true',
    ]
    return predictions.groupby(keys, as_index=False).agg(
        y_prob=('y_prob', 'mean'),
        n_seeds=('seed', 'nunique'),
    )


def select_thresholds(validation_predictions):
    thresholds = {}
    for target in TARGETS:
        for horizon in HORIZONS:
            part = validation_predictions[
                (validation_predictions['target'] == target)
                & (validation_predictions['horizon'] == horizon)
            ]
            thresholds[(target, horizon)] = best_f1_threshold(
                part['y_true'].to_numpy(dtype=int),
                part['y_prob'].to_numpy(dtype=float),
            )
    return thresholds


def evaluate_predictions(predictions, thresholds):
    rows = []
    for target in TARGETS:
        for horizon in HORIZONS:
            part = predictions[
                (predictions['target'] == target)
                & (predictions['horizon'] == horizon)
            ]
            y = part['y_true'].to_numpy(dtype=int)
            p = part['y_prob'].to_numpy(dtype=float)
            threshold = thresholds[(target, horizon)]
            rows.append({
                'target': target,
                'horizon': horizon,
                'n': len(part),
                'prevalence': float(y.mean()) if len(y) else np.nan,
                'threshold': threshold,
                **calculate_metrics(y, p, threshold),
            })
    return pd.DataFrame(rows)


def station_year_bootstrap_ci(predictions, thresholds, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    rows = []

    for target in TARGETS:
        for horizon in HORIZONS:
            part = predictions[
                (predictions['target'] == target)
                & (predictions['horizon'] == horizon)
            ].copy()
            part['cluster'] = part['station'].astype(str) + '__' + part['year'].astype(str)
            clusters = part['cluster'].unique()
            samples = {c: part[part['cluster'] == c] for c in clusters}
            threshold = thresholds[(target, horizon)]
            metric_draws = {name: [] for name in ['AUPRC', 'AUROC', 'F1', 'MCC', 'Brier']}

            for _ in range(reps):
                chosen = rng.choice(clusters, size=len(clusters), replace=True)
                boot = pd.concat([samples[c] for c in chosen], ignore_index=True)
                y = boot['y_true'].to_numpy(dtype=int)
                p = boot['y_prob'].to_numpy(dtype=float)
                metrics = calculate_metrics(y, p, threshold)
                for name in metric_draws:
                    if pd.notna(metrics[name]):
                        metric_draws[name].append(metrics[name])

            for name, values in metric_draws.items():
                low, high = np.quantile(values, [0.025, 0.975]) if values else (np.nan, np.nan)
                rows.append({
                    'target': target,
                    'horizon': horizon,
                    'metric': name,
                    'ci_low': low,
                    'ci_high': high,
                })

    return pd.DataFrame(rows)


def paired_bootstrap_auprc(reference, comparison, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    keys = ['station', 'year', 'week_start', 'target', 'horizon', 'y_true']
    merged = reference.merge(
        comparison,
        on=keys,
        suffixes=('_reference', '_comparison'),
        validate='one_to_one',
    )
    rng = np.random.default_rng(seed)
    rows = []

    for target in TARGETS:
        for horizon in HORIZONS:
            part = merged[
                (merged['target'] == target)
                & (merged['horizon'] == horizon)
            ].copy()
            part['cluster'] = part['station'].astype(str) + '__' + part['year'].astype(str)
            clusters = part['cluster'].unique()
            samples = {c: part[part['cluster'] == c] for c in clusters}
            diffs = []

            for _ in range(reps):
                chosen = rng.choice(clusters, size=len(clusters), replace=True)
                boot = pd.concat([samples[c] for c in chosen], ignore_index=True)
                y = boot['y_true'].to_numpy(dtype=int)
                if y.sum() == 0:
                    continue
                ref = average_precision_score(y, boot['y_prob_reference'])
                cmp = average_precision_score(y, boot['y_prob_comparison'])
                diffs.append(ref - cmp)

            diffs = np.asarray(diffs, dtype=float)
            observed = (
                safe_auprc(part['y_true'], part['y_prob_reference'])
                - safe_auprc(part['y_true'], part['y_prob_comparison'])
            )
            p_value = 2.0 * min(
                np.mean(diffs <= 0.0),
                np.mean(diffs >= 0.0),
            ) if len(diffs) else np.nan

            rows.append({
                'target': target,
                'horizon': horizon,
                'delta_auprc': observed,
                'ci_low': np.quantile(diffs, 0.025) if len(diffs) else np.nan,
                'ci_high': np.quantile(diffs, 0.975) if len(diffs) else np.nan,
                'p_value': p_value,
            })

    return pd.DataFrame(rows)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.full(len(p), np.nan)
    running = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (len(p) - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running

    return adjusted


def add_holm_adjustment(comparisons):
    out = comparisons.copy()
    out['p_holm'] = np.nan
    valid = out['p_value'].notna()
    out.loc[valid, 'p_holm'] = holm_adjust(out.loc[valid, 'p_value'].to_numpy())
    return out


def main():
    df, params = load_data()
    all_predictions = []

    for seed in SEEDS:
        path = CKPT_DIR / f'hydrotoxnet_seed{seed}.pt'
        model, _ = load_checkpoint(path, params)

        for split in ['validation', 'test']:
            dataset = ForecastDataset(df, params, split)
            pred = predict_split(model, dataset, seed, split)
            all_predictions.append(pred)

    seed_predictions = pd.concat(all_predictions, ignore_index=True)
    seed_predictions.to_csv(PRED_DIR / 'seed_predictions.csv', index=False)

    averaged = average_seed_predictions(seed_predictions)
    averaged.to_csv(PRED_DIR / 'seed_averaged_predictions.csv', index=False)

    validation = averaged[averaged['split'] == 'validation'].copy()
    test = averaged[averaged['split'] == 'test'].copy()
    thresholds = select_thresholds(validation)

    threshold_json = {
        f'{target}_{horizon}': value
        for (target, horizon), value in thresholds.items()
    }
    with open(METRIC_DIR / 'validation_thresholds.json', 'w') as f:
        json.dump(threshold_json, f, indent=2)

    metrics = evaluate_predictions(test, thresholds)
    metrics.to_csv(METRIC_DIR / 'test_metrics.csv', index=False)

    ci = station_year_bootstrap_ci(test, thresholds)
    ci.to_csv(METRIC_DIR / 'test_metric_bootstrap_ci.csv', index=False)

    print(metrics.round(4).to_string(index=False))
    print('Saved:', OUTDIR)


if __name__ == '__main__':
    main()
