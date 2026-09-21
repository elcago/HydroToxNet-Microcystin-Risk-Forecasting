from pathlib import Path
from itertools import product
import copy
import hashlib
import json
import math
import random
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score

from model import HydroToxNet

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ROOT = Path('/content/drive/MyDrive/hab_stgnn')
PREP_DIR = ROOT / 'results' / 'preprocessing'
DATA_PATH = PREP_DIR / 'hydrotoxnet_weekly_processed.csv'
PARAM_PATH = PREP_DIR / 'preprocessing_parameters.json'
OUTDIR = ROOT / 'results' / 'hydrotoxnet'
CKPT_DIR = OUTDIR / 'checkpoints'
SEARCH_DIR = OUTDIR / 'hyperparameter_search'
OUTDIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [7, 14, 21]
TARGETS = ['Elevated', 'HigherRisk']
SEEDS = [11, 29, 47, 71, 97]
SEARCH_SEED = 2026
N_SEARCH = 100
RUN_RANDOM_SEARCH = True
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEARCH_SPACE = {
    'representation_dim': [8, 16, 32, 64, 128, 256],
    'learning_rate': [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'weight_decay': [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
}

PAPER_CONFIG = {
    'representation_dim': 32,
    'learning_rate': 1e-4,
    'dropout': 0.2,
    'weight_decay': 1e-4,
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def build_station_geometry(df, stations):
    coords = (
        df.groupby('station')[['latitude', 'longitude']]
        .median()
        .reindex(stations)
    )
    if coords.isna().any().any():
        raise ValueError('Latitude/longitude are missing for one or more stations.')

    n = len(stations)
    distance = np.zeros((n, n), dtype=np.float32)
    bearing = np.zeros((n, n), dtype=np.float32)

    for source in range(n):
        for target in range(n):
            if source == target:
                continue
            a = coords.iloc[source]
            b = coords.iloc[target]
            distance[source, target] = haversine_km(
                a.latitude, a.longitude, b.latitude, b.longitude
            )
            bearing[source, target] = bearing_deg(
                a.latitude, a.longitude, b.latitude, b.longitude
            )

    max_distance = distance.max()
    normalized = distance / max_distance if max_distance > 0 else distance
    return distance, normalized, bearing


class ForecastDataset(Dataset):
    def __init__(self, df, params, split, remove_nitrogen=False, remove_prior_history=False):
        self.df = df.copy()
        self.params = params
        self.stations = params['stations']
        self.station_to_idx = {s: i for i, s in enumerate(self.stations)}
        self.ecological = params['ecological_features']
        self.river = params['river_features']
        self.met = params['meteorological_features']
        self.history = params['history_features']
        self.remove_nitrogen = remove_nitrogen
        self.remove_prior_history = remove_prior_history

        self.distance, self.normalized_distance, self.bearing = build_station_geometry(
            self.df, self.stations
        )

        self.by_week = {
            week: g.set_index('station')
            for week, g in self.df.groupby('week_start')
        }

        candidates = self.df[self.df['split'] == split].copy()
        target_mask_cols = [
            f'mask_{target}_{h}'
            for h in HORIZONS
            for target in TARGETS
        ]
        candidates = candidates[candidates[target_mask_cols].sum(axis=1) > 0]
        self.rows = candidates.reset_index(drop=True)

        self.nitrogen_features = {
            'ammonia',
            'nitrate_nitrite',
            'ammonia_fraction_inorganic_n',
            'inorganic_n_tp_ratio',
        }

    def __len__(self):
        return len(self.rows)

    def _feature_vector(self, row, columns):
        values = row[columns].to_numpy(dtype=np.float32, copy=True)
        if self.remove_nitrogen:
            for i, col in enumerate(columns):
                if col in self.nitrogen_features:
                    values[i] = 0.0
        return values

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        week = row['week_start']
        target_station = row['station']
        target_idx = self.station_to_idx[target_station]
        snapshot = self.by_week[week]

        target_ecological = self._feature_vector(row, self.ecological)
        toxin_history = row[self.history].to_numpy(dtype=np.float32, copy=True)

        if self.remove_prior_history:
            toxin_history[1:] = 0.0

        source_ecological = np.zeros(
            (len(self.stations), len(self.ecological)), dtype=np.float32
        )
        source_mc_raw = np.zeros(len(self.stations), dtype=np.float32)
        source_mask = np.zeros(len(self.stations), dtype=bool)

        for station_idx, station in enumerate(self.stations):
            if station not in snapshot.index:
                continue
            source_row = snapshot.loc[station]
            if isinstance(source_row, pd.DataFrame):
                source_row = source_row.iloc[0]

            source_ecological[station_idx] = self._feature_vector(
                source_row, self.ecological
            )
            raw_col = 'particulate_microcystin__raw'
            raw_mc = source_row.get(raw_col, np.nan)
            if pd.notna(raw_mc):
                source_mc_raw[station_idx] = float(raw_mc)
                source_mask[station_idx] = True

        source_mask[target_idx] = False

        wind_dir_raw = row.get('wind_direction_deg__raw', row.get('wind_direction_deg', np.nan))
        if pd.isna(wind_dir_raw):
            wind_dir_raw = 0.0
        travel_direction = (float(wind_dir_raw) + 180.0) % 360.0
        delta = np.deg2rad(self.bearing[:, target_idx] - travel_direction)
        wind_alignment = np.cos(delta).astype(np.float32)

        labels = np.zeros((3, 2), dtype=np.float32)
        masks = np.zeros((3, 2), dtype=np.float32)

        for hi, horizon in enumerate(HORIZONS):
            for ci, target in enumerate(TARGETS):
                y = row[f'y_{target}_{horizon}']
                m = row[f'mask_{target}_{horizon}']
                labels[hi, ci] = 0.0 if pd.isna(y) else float(y)
                masks[hi, ci] = float(m)

        return {
            'target_ecological': torch.tensor(target_ecological),
            'toxin_history': torch.tensor(toxin_history),
            'source_ecological': torch.tensor(source_ecological),
            'source_mc_raw': torch.tensor(source_mc_raw),
            'source_mask': torch.tensor(source_mask),
            'distance_km': torch.tensor(self.distance[:, target_idx]),
            'normalized_distance': torch.tensor(self.normalized_distance[:, target_idx]),
            'wind_alignment': torch.tensor(wind_alignment),
            'river': torch.tensor(row[self.river].to_numpy(dtype=np.float32)),
            'meteorology': torch.tensor(row[self.met].to_numpy(dtype=np.float32)),
            'labels': torch.tensor(labels),
            'masks': torch.tensor(masks),
            'station': target_station,
            'year': int(row['year']),
            'week_start': str(row['week_start']),
        }


def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df['week_start'] = pd.to_datetime(df['week_start'])
    with open(PARAM_PATH) as f:
        params = json.load(f)
    return df, params


def build_model(params, config, flags=None):
    flags = flags or {}
    return HydroToxNet(
        ecological_dim=len(params['ecological_features']),
        river_dim=len(params['river_features']),
        meteorological_dim=len(params['meteorological_features']),
        representation_dim=config['representation_dim'],
        horizon_dim=8,
        dropout=config['dropout'],
        **flags,
    )


def move_batch(batch):
    tensor_keys = [
        'target_ecological', 'toxin_history', 'source_ecological',
        'source_mc_raw', 'source_mask', 'distance_km',
        'normalized_distance', 'wind_alignment', 'river',
        'meteorology', 'labels', 'masks',
    ]
    return {k: batch[k].to(DEVICE) for k in tensor_keys}


def predict_batch(model, batch):
    return model(
        batch['target_ecological'],
        batch['toxin_history'],
        batch['source_ecological'],
        batch['source_mc_raw'],
        batch['source_mask'],
        batch['distance_km'],
        batch['normalized_distance'],
        batch['wind_alignment'],
        batch['river'],
        batch['meteorology'],
    )


def class_weights(dataset):
    y = np.zeros((3, 2), dtype=float)
    n = np.zeros((3, 2), dtype=float)

    for _, row in dataset.rows.iterrows():
        for hi, horizon in enumerate(HORIZONS):
            for ci, target in enumerate(TARGETS):
                if row[f'mask_{target}_{horizon}'] == 1:
                    n[hi, ci] += 1
                    y[hi, ci] += row[f'y_{target}_{horizon}']

    negatives = n - y
    weights = negatives / np.maximum(y, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def weighted_masked_bce(logits, labels, masks, pos_weights):
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    weights = torch.where(labels > 0.5, pos_weights.unsqueeze(0), torch.ones_like(losses))
    weights = weights * masks
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def validation_elevated_7_auprc(model, loader):
    model.eval()
    truth = []
    prob = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch)
            logits = predict_batch(model, batch)
            p = torch.sigmoid(logits[:, 0, 0])
            mask = batch['masks'][:, 0, 0] > 0.5
            truth.extend(batch['labels'][mask, 0, 0].cpu().numpy().tolist())
            prob.extend(p[mask].cpu().numpy().tolist())

    if len(set(truth)) < 2:
        return float('nan')
    return average_precision_score(truth, prob)


def train_one_seed(
    df,
    params,
    config,
    seed,
    flags=None,
    remove_nitrogen=False,
    remove_prior_history=False,
    save_path=None,
):
    seed_all(seed)

    train_ds = ForecastDataset(
        df, params, 'train', remove_nitrogen, remove_prior_history
    )
    val_ds = ForecastDataset(
        df, params, 'validation', remove_nitrogen, remove_prior_history
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(params, config, flags).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    pos_weights = class_weights(train_ds)

    best_state = None
    best_score = -np.inf
    best_epoch = 0
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for raw_batch in train_loader:
            batch = move_batch(raw_batch)
            optimizer.zero_grad()
            logits = predict_batch(model, batch)
            loss = weighted_masked_bce(
                logits,
                batch['labels'],
                batch['masks'],
                pos_weights,
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        val_score = validation_elevated_7_auprc(model, val_loader)
        train_loss = total_loss / max(total_batches, 1)
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'validation_elevated_7_auprc': val_score,
        })

        score = -np.inf if np.isnan(val_score) else val_score
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError('Training did not produce a valid checkpoint.')

    model.load_state_dict(best_state)

    checkpoint = {
        'model_state_dict': best_state,
        'config': config,
        'flags': flags or {},
        'seed': seed,
        'best_epoch': best_epoch,
        'best_validation_elevated_7_auprc': best_score,
        'positive_class_weights': pos_weights.detach().cpu().numpy().tolist(),
        'remove_nitrogen': remove_nitrogen,
        'remove_prior_history': remove_prior_history,
        'params': params,
    }

    if save_path is not None:
        torch.save(checkpoint, save_path)
        pd.DataFrame(history).to_csv(
            save_path.with_suffix('.history.csv'), index=False
        )

    return model, checkpoint, pd.DataFrame(history)


def all_search_configs():
    keys = list(SEARCH_SPACE)
    values = [SEARCH_SPACE[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def run_random_search(df, params):
    configs = all_search_configs()
    rng = random.Random(SEARCH_SEED)
    chosen = rng.sample(configs, min(N_SEARCH, len(configs)))
    rows = []

    for index, config in enumerate(chosen, start=1):
        print(f'Search {index}/{len(chosen)}:', config)
        _, checkpoint, _ = train_one_seed(
            df,
            params,
            config,
            SEARCH_SEED,
        )
        rows.append({
            **config,
            'validation_elevated_7_auprc': checkpoint['best_validation_elevated_7_auprc'],
            'best_epoch': checkpoint['best_epoch'],
        })

    results = pd.DataFrame(rows).sort_values(
        'validation_elevated_7_auprc', ascending=False
    )
    results.to_csv(SEARCH_DIR / 'random_search_results.csv', index=False)
    return results.iloc[0][list(SEARCH_SPACE)].to_dict()


def main():
    df, params = load_data()

    if RUN_RANDOM_SEARCH:
        selected = run_random_search(df, params)
        selected['representation_dim'] = int(selected['representation_dim'])
    else:
        selected = PAPER_CONFIG.copy()

    with open(OUTDIR / 'selected_config.json', 'w') as f:
        json.dump(selected, f, indent=2)

    run_rows = []
    for seed in SEEDS:
        checkpoint_path = CKPT_DIR / f'hydrotoxnet_seed{seed}.pt'
        print('Training seed:', seed)
        _, checkpoint, _ = train_one_seed(
            df,
            params,
            selected,
            seed,
            save_path=checkpoint_path,
        )
        run_rows.append({
            'seed': seed,
            'best_epoch': checkpoint['best_epoch'],
            'validation_elevated_7_auprc': checkpoint['best_validation_elevated_7_auprc'],
            'checkpoint': str(checkpoint_path),
        })

    pd.DataFrame(run_rows).to_csv(OUTDIR / 'training_summary.csv', index=False)

    provenance = {
        'input_sha256': sha256_file(DATA_PATH),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'device': str(DEVICE),
        'seeds': SEEDS,
        'batch_size': BATCH_SIZE,
        'max_epochs': MAX_EPOCHS,
        'patience': PATIENCE,
    }
    with open(OUTDIR / 'provenance.json', 'w') as f:
        json.dump(provenance, f, indent=2)

    print('Saved:', OUTDIR)


if __name__ == '__main__':
    main()
