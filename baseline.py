from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from model_training import ROOT, OUTDIR, HORIZONS, TARGETS, SEEDS, seed_all
from model_evaluation import (
    average_seed_predictions,
    select_thresholds,
    evaluate_predictions,
    paired_bootstrap_auprc,
    add_holm_adjustment,
)

PREP_DIR = ROOT / 'results' / 'preprocessing'
DATA_PATH = PREP_DIR / 'hydrotoxnet_weekly_processed.csv'
PARAM_PATH = PREP_DIR / 'preprocessing_parameters.json'
BASELINE_DIR = OUTDIR / 'baselines'
BASELINE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.dropout(h[-1])).squeeze(-1)


class StaticGATClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.node = nn.Linear(input_dim, hidden_dim)
        self.attn_source = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_target = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, adjacency, target_idx):
        h = torch.relu(self.node(x))
        source_score = self.attn_source(h).squeeze(-1)
        target_h = h[torch.arange(len(h), device=h.device), target_idx]
        target_score = self.attn_target(target_h).squeeze(-1)
        score = source_score + target_score[:, None]
        mask = adjacency[target_idx] > 0
        score = score.masked_fill(~mask, -1e9)
        weight = torch.softmax(score, dim=1)
        pooled = torch.sum(weight.unsqueeze(-1) * h, dim=1)
        return self.head(self.dropout(pooled)).squeeze(-1)


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['week_start'] = pd.to_datetime(df['week_start'])
    with open(PARAM_PATH) as f:
        params = json.load(f)
    return df, params


def tabular_features(params):
    return list(dict.fromkeys(
        params['ecological_features']
        + params['history_features']
        + params['river_features']
        + params['meteorological_features']
    ))


def make_xy(df, params, split, target, horizon):
    mask = (df['split'] == split) & (df[f'mask_{target}_{horizon}'] == 1)
    part = df.loc[mask].copy()
    features = tabular_features(params)
    X = part[features].to_numpy(dtype=np.float32)
    y = part[f'y_{target}_{horizon}'].to_numpy(dtype=int)
    return part, X, y


def predict_sklearn(model, part, X, seed, split, target, horizon):
    prob = model.predict_proba(X)[:, 1]
    return pd.DataFrame({
        'split': split,
        'seed': seed,
        'station': part['station'].to_numpy(),
        'year': part['year'].to_numpy(dtype=int),
        'week_start': part['week_start'].astype(str).to_numpy(),
        'target': target,
        'horizon': horizon,
        'y_true': part[f'y_{target}_{horizon}'].to_numpy(dtype=int),
        'y_prob': prob,
    })


def tune_sklearn(name, df, params, target, horizon, seed):
    train_part, X_train, y_train = make_xy(df, params, 'train', target, horizon)
    val_part, X_val, y_val = make_xy(df, params, 'validation', target, horizon)

    if name == 'logistic_regression':
        configs = [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]]
    elif name == 'random_forest':
        configs = [
            {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': leaf}
            for n in [200, 500]
            for d in [None, 8, 16]
            for leaf in [1, 3, 5]
        ]
    else:
        configs = [
            {'n_estimators': n, 'max_depth': d, 'learning_rate': lr}
            for n in [200, 500]
            for d in [2, 4, 6]
            for lr in [0.03, 0.1]
        ]

    best_config = None
    best_score = -np.inf

    for config in configs:
        if name == 'logistic_regression':
            model = LogisticRegression(
                C=config['C'],
                class_weight='balanced',
                max_iter=3000,
                random_state=seed,
            )
        elif name == 'random_forest':
            model = RandomForestClassifier(
                **config,
                class_weight='balanced',
                random_state=seed,
                n_jobs=-1,
            )
        else:
            if XGBClassifier is None:
                raise ImportError('xgboost is required for the XGBoost baseline.')
            model = XGBClassifier(
                **config,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric='logloss',
                random_state=seed,
                n_jobs=-1,
            )

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, prob)

        if score > best_score:
            best_score = score
            best_config = config

    return best_config


def fit_sklearn(name, config, X, y, seed):
    if name == 'logistic_regression':
        model = LogisticRegression(
            C=config['C'], class_weight='balanced', max_iter=3000, random_state=seed
        )
    elif name == 'random_forest':
        model = RandomForestClassifier(
            **config, class_weight='balanced', random_state=seed, n_jobs=-1
        )
    else:
        model = XGBClassifier(
            **config,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric='logloss',
            random_state=seed,
            n_jobs=-1,
        )
    model.fit(X, y)
    return model


def sequence_dataset(df, params, split, target, horizon):
    features = list(dict.fromkeys(
        params['ecological_features'] + params['river_features'] + params['meteorological_features']
        + ['particulate_microcystin']
    ))
    lookup = {
        (row.station, row.week_start): row
        for _, row in df.iterrows()
    }
    rows = []
    sequences = []
    labels = []

    candidates = df[(df['split'] == split) & (df[f'mask_{target}_{horizon}'] == 1)]
    for _, row in candidates.iterrows():
        seq = []
        valid = True
        for lag in [3, 2, 1, 0]:
            key = (row['station'], row['week_start'] - pd.Timedelta(weeks=lag))
            if key not in lookup:
                valid = False
                break
            seq.append(lookup[key][features].to_numpy(dtype=np.float32))
        if not valid:
            continue
        rows.append(row)
        sequences.append(seq)
        labels.append(int(row[f'y_{target}_{horizon}']))

    return pd.DataFrame(rows), np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train_torch_binary(model, X_train, y_train, X_val, y_val, lr, weight_decay, seed, extra=None):
    seed_all(seed)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    positives = max(float(y_train.sum()), 1.0)
    negatives = float(len(y_train) - y_train.sum())
    pos_weight = torch.tensor(negatives / positives, device=DEVICE)
    best_state = None
    best_score = -np.inf
    wait = 0

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)

    for _ in range(200):
        model.train()
        order = torch.randperm(len(X_train_t), device=DEVICE)
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            optimizer.zero_grad()
            if extra is None:
                logits = model(X_train_t[idx])
            else:
                logits = model(X_train_t[idx], extra['adjacency'], extra['target_train'][idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, y_train_t[idx], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if extra is None:
                prob = torch.sigmoid(model(X_val_t)).cpu().numpy()
            else:
                target_val = extra['target_val']
                prob = torch.sigmoid(model(X_val_t, extra['adjacency'], target_val)).cpu().numpy()
        score = average_precision_score(y_val, prob)

        if score > best_score + 1e-8:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= 20:
            break

    model.load_state_dict(best_state)
    return model


def run_tabular_baseline(name, df, params):
    predictions = []
    config_rows = []

    for target in TARGETS:
        for horizon in HORIZONS:
            config = tune_sklearn(name, df, params, target, horizon, SEEDS[0])
            config_rows.append({'target': target, 'horizon': horizon, **config})

            for seed in SEEDS:
                _, X_train, y_train = make_xy(df, params, 'train', target, horizon)
                model = fit_sklearn(name, config, X_train, y_train, seed)

                for split in ['validation', 'test']:
                    part, X, _ = make_xy(df, params, split, target, horizon)
                    predictions.append(
                        predict_sklearn(model, part, X, seed, split, target, horizon)
                    )

    return pd.concat(predictions, ignore_index=True), pd.DataFrame(config_rows)


def run_lstm(df, params):
    predictions = []

    for target in TARGETS:
        for horizon in HORIZONS:
            train_part, X_train, y_train = sequence_dataset(df, params, 'train', target, horizon)
            val_part, X_val, y_val = sequence_dataset(df, params, 'validation', target, horizon)
            test_part, X_test, y_test = sequence_dataset(df, params, 'test', target, horizon)

            for seed in SEEDS:
                model = LSTMClassifier(X_train.shape[-1], hidden_dim=32, dropout=0.2)
                model = train_torch_binary(
                    model, X_train, y_train, X_val, y_val, 1e-3, 1e-4, seed
                )
                model.eval()
                with torch.no_grad():
                    val_prob = torch.sigmoid(model(torch.tensor(X_val, dtype=torch.float32, device=DEVICE))).cpu().numpy()
                    test_prob = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE))).cpu().numpy()

                for split, part, prob in [
                    ('validation', val_part, val_prob),
                    ('test', test_part, test_prob),
                ]:
                    predictions.append(pd.DataFrame({
                        'split': split,
                        'seed': seed,
                        'station': part['station'].to_numpy(),
                        'year': part['year'].to_numpy(dtype=int),
                        'week_start': part['week_start'].astype(str).to_numpy(),
                        'target': target,
                        'horizon': horizon,
                        'y_true': part[f'y_{target}_{horizon}'].to_numpy(dtype=int),
                        'y_prob': prob,
                    }))

    return pd.concat(predictions, ignore_index=True)


def station_geometry(df, stations):
    coords = df.groupby('station')[['latitude', 'longitude']].median().reindex(stations)
    n = len(stations)
    distance = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            lat1, lon1 = np.deg2rad(coords.iloc[i].to_numpy())
            lat2, lon2 = np.deg2rad(coords.iloc[j].to_numpy())
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            distance[i, j] = 6371.0088 * 2 * np.arcsin(np.sqrt(a))
    adjacency = np.exp(-distance / 10.0)
    np.fill_diagonal(adjacency, 0.0)
    return torch.tensor(adjacency, dtype=torch.float32, device=DEVICE)


def graph_dataset(df, params, split, target, horizon):
    stations = params['stations']
    station_to_idx = {s: i for i, s in enumerate(stations)}
    features = list(dict.fromkeys(params['ecological_features'] + params['river_features'] + params['meteorological_features'] + ['particulate_microcystin']))
    by_week = {w: g.set_index('station') for w, g in df.groupby('week_start')}
    candidates = df[(df['split'] == split) & (df[f'mask_{target}_{horizon}'] == 1)]
    X, y, target_idx, rows = [], [], [], []

    for _, row in candidates.iterrows():
        snap = by_week[row['week_start']]
        nodes = np.zeros((len(stations), len(features)), dtype=np.float32)
        for i, station in enumerate(stations):
            if station in snap.index:
                node = snap.loc[station]
                if isinstance(node, pd.DataFrame):
                    node = node.iloc[0]
                nodes[i] = node[features].to_numpy(dtype=np.float32)
        X.append(nodes)
        y.append(int(row[f'y_{target}_{horizon}']))
        target_idx.append(station_to_idx[row['station']])
        rows.append(row)

    return pd.DataFrame(rows), np.asarray(X), np.asarray(y, dtype=np.float32), np.asarray(target_idx, dtype=np.int64)


def run_gat(df, params):
    predictions = []
    adjacency = station_geometry(df, params['stations'])

    for target in TARGETS:
        for horizon in HORIZONS:
            train_part, X_train, y_train, idx_train = graph_dataset(df, params, 'train', target, horizon)
            val_part, X_val, y_val, idx_val = graph_dataset(df, params, 'validation', target, horizon)
            test_part, X_test, y_test, idx_test = graph_dataset(df, params, 'test', target, horizon)

            for seed in SEEDS:
                model = StaticGATClassifier(X_train.shape[-1], hidden_dim=32, dropout=0.2)
                extra = {
                    'adjacency': adjacency,
                    'target_train': torch.tensor(idx_train, dtype=torch.long, device=DEVICE),
                    'target_val': torch.tensor(idx_val, dtype=torch.long, device=DEVICE),
                }
                model = train_torch_binary(
                    model, X_train, y_train, X_val, y_val, 1e-3, 1e-4, seed, extra
                )
                model.eval()
                with torch.no_grad():
                    val_prob = torch.sigmoid(model(
                        torch.tensor(X_val, dtype=torch.float32, device=DEVICE),
                        adjacency,
                        torch.tensor(idx_val, dtype=torch.long, device=DEVICE),
                    )).cpu().numpy()
                    test_prob = torch.sigmoid(model(
                        torch.tensor(X_test, dtype=torch.float32, device=DEVICE),
                        adjacency,
                        torch.tensor(idx_test, dtype=torch.long, device=DEVICE),
                    )).cpu().numpy()

                for split, part, prob in [
                    ('validation', val_part, val_prob),
                    ('test', test_part, test_prob),
                ]:
                    predictions.append(pd.DataFrame({
                        'split': split,
                        'seed': seed,
                        'station': part['station'].to_numpy(),
                        'year': part['year'].to_numpy(dtype=int),
                        'week_start': part['week_start'].astype(str).to_numpy(),
                        'target': target,
                        'horizon': horizon,
                        'y_true': part[f'y_{target}_{horizon}'].to_numpy(dtype=int),
                        'y_prob': prob,
                    }))

    return pd.concat(predictions, ignore_index=True)


def finalize_baseline(name, seed_predictions, config_table=None):
    out = BASELINE_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    seed_predictions.to_csv(out / 'seed_predictions.csv', index=False)
    if config_table is not None:
        config_table.to_csv(out / 'selected_hyperparameters.csv', index=False)

    averaged = average_seed_predictions(seed_predictions)
    averaged.to_csv(out / 'seed_averaged_predictions.csv', index=False)
    val = averaged[averaged['split'] == 'validation']
    test = averaged[averaged['split'] == 'test']
    thresholds = select_thresholds(val)
    metrics = evaluate_predictions(test, thresholds)
    metrics.to_csv(out / 'test_metrics.csv', index=False)

    reference = pd.read_csv(OUTDIR / 'predictions' / 'seed_averaged_predictions.csv')
    reference = reference[reference['split'] == 'test']
    comparison = paired_bootstrap_auprc(reference, test)
    comparison = add_holm_adjustment(comparison)
    comparison.to_csv(out / 'paired_bootstrap.csv', index=False)
    return metrics, comparison


def main():
    df, params = load_data()
    metric_rows = []
    test_rows = []

    for name in ['logistic_regression', 'random_forest', 'xgboost']:
        print('Running:', name)
        pred, configs = run_tabular_baseline(name, df, params)
        metrics, tests = finalize_baseline(name, pred, configs)
        metrics.insert(0, 'model', name)
        tests.insert(0, 'model', name)
        metric_rows.append(metrics)
        test_rows.append(tests)

    print('Running: lstm')
    pred = run_lstm(df, params)
    metrics, tests = finalize_baseline('lstm', pred)
    metrics.insert(0, 'model', 'lstm')
    tests.insert(0, 'model', 'lstm')
    metric_rows.append(metrics)
    test_rows.append(tests)

    print('Running: gat')
    pred = run_gat(df, params)
    metrics, tests = finalize_baseline('gat', pred)
    metrics.insert(0, 'model', 'gat')
    tests.insert(0, 'model', 'gat')
    metric_rows.append(metrics)
    test_rows.append(tests)

    pd.concat(metric_rows, ignore_index=True).to_csv(
        BASELINE_DIR / 'baseline_metrics.csv', index=False
    )
    pd.concat(test_rows, ignore_index=True).to_csv(
        BASELINE_DIR / 'baseline_paired_bootstrap.csv', index=False
    )

    print('Saved:', BASELINE_DIR)


if __name__ == '__main__':
    main()
