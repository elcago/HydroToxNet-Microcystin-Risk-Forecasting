from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

ROOT = Path('/content/drive/MyDrive/hab_stgnn')
MASTER_PATH = ROOT / 'lake_erie_multimodal_master_2012_2022.csv'
OUTDIR = ROOT / 'results' / 'preprocessing'
OUTDIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PATH = OUTDIR / 'hydrotoxnet_weekly_processed.csv'
PARAM_PATH = OUTDIR / 'preprocessing_parameters.json'
MISSING_PATH = OUTDIR / 'training_missingness.csv'
CORR_PATH = OUTDIR / 'training_spearman_correlation.csv'
HIGH_CORR_PATH = OUTDIR / 'training_spearman_high_pairs.csv'

STATIONS = ['WE02', 'WE04', 'WE06', 'WE08', 'WE09', 'WE12', 'WE13', 'WE15']
TRAIN_YEARS = range(2012, 2018)
VAL_YEARS = [2018]
TEST_YEARS = range(2019, 2023)
HORIZONS = [7, 14, 21]
THRESHOLDS = {'Elevated': 1.0, 'HigherRisk': 1.6}
MISSING_LIMIT = 0.40

BASE_PREDICTORS = [
    'water_temperature',
    'turbidity',
    'secchi_depth',
    'ctd_dissolved_oxygen',
    'ctd_specific_conductivity',
    'chlorophyll_a',
    'phycocyanin',
    'particulate_microcystin',
    'ammonia',
    'nitrate_nitrite',
    'total_phosphorus',
    'soluble_reactive_phosphorus',
    'river_discharge_cfs',
    'river_nitrate',
    'river_srp',
    'wind_speed_ms',
    'wind_direction_deg',
]

ECOLOGICAL_BASE = [
    'water_temperature',
    'turbidity',
    'secchi_depth',
    'ctd_dissolved_oxygen',
    'ctd_specific_conductivity',
    'chlorophyll_a',
    'phycocyanin',
    'ammonia',
    'nitrate_nitrite',
    'total_phosphorus',
    'soluble_reactive_phosphorus',
]

RIVER_FEATURES = ['river_discharge_cfs', 'river_nitrate', 'river_srp']
MET_FEATURES = ['wind_speed_ms', 'wind_direction_deg']
DERIVED_FEATURES = [
    'ammonia_fraction_inorganic_n',
    'inorganic_n_tp_ratio',
    'sin_week',
    'cos_week',
]



FIELD_ALIASES = {
    'water_temperature': ['Sample Temperature (°C)'],
    'ctd_specific_conductivity': ['CTD Specific Conductivity (µS/cm)'],
    'ctd_dissolved_oxygen': ['CTD Dissolved Oxygen (mg/L)'],
}


def combine_equivalent_fields(df):
    out = df.copy()
    for canonical, alternatives in FIELD_ALIASES.items():
        if canonical not in out.columns:
            out[canonical] = np.nan
        for alt in alternatives:
            if alt in out.columns:
                out[canonical] = out[canonical].combine_first(out[alt])
    return out

def normalize_station(value):
    text = str(value).strip().upper()
    digits = ''.join(ch for ch in text if ch.isdigit())
    return f'WE{int(digits):02d}' if digits else text


def safe_numeric(series):
    text = series.astype(str)
    text = text.str.replace('<', '', regex=False)
    text = text.str.replace('>', '', regex=False)
    text = text.str.split('±').str[0]
    return pd.to_numeric(text, errors='coerce')


def make_week_start(date_series):
    iso = date_series.dt.isocalendar()
    return pd.to_datetime(
        iso.year.astype(str) + '-W' + iso.week.astype(str).str.zfill(2) + '-1',
        format='%G-W%V-%u',
        errors='coerce',
    )


def aggregate_weekly(df):
    df = df.copy()
    df['week_start'] = make_week_start(df['date'])

    for col in BASE_PREDICTORS + ['latitude', 'longitude']:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if {'wind_speed_ms', 'wind_direction_deg'}.issubset(df.columns):
        angle = np.deg2rad(df['wind_direction_deg'])
        df['_wind_u'] = df['wind_speed_ms'] * np.sin(angle)
        df['_wind_v'] = df['wind_speed_ms'] * np.cos(angle)

    mean_cols = [c for c in BASE_PREDICTORS + ['latitude', 'longitude'] if c in df.columns]
    agg = {c: 'mean' for c in mean_cols if c != 'wind_direction_deg'}

    if '_wind_u' in df.columns:
        agg['_wind_u'] = 'mean'
        agg['_wind_v'] = 'mean'

    weekly = df.groupby(['station', 'week_start'], as_index=False).agg(agg)

    if '_wind_u' in weekly.columns:
        weekly['wind_direction_deg'] = (
            np.degrees(np.arctan2(weekly['_wind_u'], weekly['_wind_v'])) + 360.0
        ) % 360.0
        weekly['wind_speed_ms'] = np.sqrt(weekly['_wind_u'] ** 2 + weekly['_wind_v'] ** 2)
        weekly = weekly.drop(columns=['_wind_u', '_wind_v'])

    iso = weekly['week_start'].dt.isocalendar()
    weekly['year'] = iso.year.astype(int)
    weekly['week'] = iso.week.astype(int)
    return weekly


def add_derived_features(weekly):
    out = weekly.copy()

    nh4 = out['ammonia'] if 'ammonia' in out else np.nan
    nox = out['nitrate_nitrite'] if 'nitrate_nitrite' in out else np.nan
    tp = out['total_phosphorus'] if 'total_phosphorus' in out else np.nan

    nitrate_ug_l = nox * 1000.0
    inorganic_n = nh4 + nitrate_ug_l
    out['ammonia_fraction_inorganic_n'] = nh4 / inorganic_n.replace(0, np.nan)
    out['inorganic_n_tp_ratio'] = inorganic_n / tp.replace(0, np.nan)
    out['sin_week'] = np.sin(2.0 * np.pi * out['week'] / 52.0)
    out['cos_week'] = np.cos(2.0 * np.pi * out['week'] / 52.0)

    return out


def add_toxin_history(weekly):
    out = weekly.copy()
    lookup = {
        (row.station, row.week_start): row.particulate_microcystin
        for row in out[['station', 'week_start', 'particulate_microcystin']].itertuples(index=False)
    }

    for lag in range(4):
        out[f'mc_hist_{lag}'] = [
            lookup.get((s, t - pd.Timedelta(weeks=lag)), np.nan)
            for s, t in zip(out['station'], out['week_start'])
        ]

    return out


def add_targets(weekly):
    out = weekly.copy()
    lookup = {
        (row.station, row.week_start): row.particulate_microcystin
        for row in out[['station', 'week_start', 'particulate_microcystin']].itertuples(index=False)
    }

    for horizon in HORIZONS:
        future = [
            lookup.get((s, t + pd.Timedelta(days=horizon)), np.nan)
            for s, t in zip(out['station'], out['week_start'])
        ]
        future = pd.Series(future, index=out.index, dtype=float)
        out[f'future_mc_{horizon}'] = future

        for name, threshold in THRESHOLDS.items():
            target = (future >= threshold).astype(float)
            target[future.isna()] = np.nan
            out[f'y_{name}_{horizon}'] = target
            out[f'mask_{name}_{horizon}'] = future.notna().astype(int)

    return out


def select_predictors(weekly):
    candidates = [
        c for c in ECOLOGICAL_BASE + ['particulate_microcystin'] + RIVER_FEATURES + MET_FEATURES
        if c in weekly.columns
    ]

    train = weekly[weekly['year'].isin(TRAIN_YEARS)]
    missing = train[candidates].isna().mean().sort_values(ascending=False)
    missing.rename('missing_fraction').to_csv(MISSING_PATH)

    retained = [c for c in candidates if missing[c] <= MISSING_LIMIT]
    return retained


def spearman_screen(weekly, retained):
    train = weekly[weekly['year'].isin(TRAIN_YEARS)]
    candidates = list(dict.fromkeys(retained + DERIVED_FEATURES))
    continuous = [c for c in candidates if c in train.columns and train[c].nunique(dropna=True) > 1]
    corr = train[continuous].corr(method='spearman')
    corr.to_csv(CORR_PATH)

    rows = []
    for i, a in enumerate(continuous):
        for b in continuous[i + 1:]:
            rho = corr.loc[a, b]
            if pd.notna(rho) and abs(rho) >= 0.90:
                rows.append({'predictor_1': a, 'predictor_2': b, 'spearman_rho': rho})

    pd.DataFrame(rows).to_csv(HIGH_CORR_PATH, index=False)


def fit_preprocessing(weekly, retained):
    model_features = list(dict.fromkeys(retained + DERIVED_FEATURES + [f'mc_hist_{i}' for i in range(4)]))
    model_features = [c for c in model_features if c in weekly.columns]

    train = weekly[weekly['year'].isin(TRAIN_YEARS)]
    medians = train[model_features].median(axis=0, skipna=True).fillna(0.0)
    filled = train[model_features].fillna(medians)
    means = filled.mean(axis=0)
    stds = filled.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)

    out = weekly.copy()
    for col in model_features:
        out[f'{col}__raw'] = out[col]
        out[col] = out[col].fillna(medians[col])
        out[col] = (out[col] - means[col]) / stds[col]

    params = {
        'retained_measured_predictors': retained,
        'model_features': model_features,
        'medians': medians.to_dict(),
        'means': means.to_dict(),
        'stds': stds.to_dict(),
        'ecological_features': [c for c in ECOLOGICAL_BASE + DERIVED_FEATURES if c in model_features],
        'river_features': [c for c in RIVER_FEATURES if c in model_features],
        'meteorological_features': [c for c in MET_FEATURES if c in model_features],
        'history_features': [f'mc_hist_{i}' for i in range(4)],
        'stations': STATIONS,
        'train_years': list(TRAIN_YEARS),
        'validation_years': VAL_YEARS,
        'test_years': list(TEST_YEARS),
        'horizons': HORIZONS,
        'thresholds': THRESHOLDS,
    }

    with open(PARAM_PATH, 'w') as f:
        json.dump(params, f, indent=2)

    return out, params


def main():
    df = pd.read_csv(MASTER_PATH, low_memory=False)
    df = combine_equivalent_fields(df)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'station']).copy()
    df['station'] = df['station'].map(normalize_station)
    df = df[df['station'].isin(STATIONS)].copy()

    weekly = aggregate_weekly(df)
    weekly = add_derived_features(weekly)
    weekly = add_toxin_history(weekly)
    weekly = add_targets(weekly)

    retained = select_predictors(weekly)
    spearman_screen(weekly, retained)
    processed, params = fit_preprocessing(weekly, retained)

    processed['split'] = np.select(
        [
            processed['year'].isin(TRAIN_YEARS),
            processed['year'].isin(VAL_YEARS),
            processed['year'].isin(TEST_YEARS),
        ],
        ['train', 'validation', 'test'],
        default='excluded',
    )

    processed.to_csv(PROCESSED_PATH, index=False)

    print('Rows:', len(processed))
    print('Retained measured predictors:', len(retained))
    print('Ecological features:', len(params['ecological_features']))
    print('Saved:', PROCESSED_PATH)


if __name__ == '__main__':
    main()
