from pathlib import Path
import copy
import json
import numpy as np
import pandas as pd

from model_training import OUTDIR, CKPT_DIR, SEEDS, ForecastDataset, load_data
from model_evaluation import (
    load_checkpoint,
    predict_split,
    average_seed_predictions,
    safe_auprc,
)

INTERP_DIR = OUTDIR / 'interpretation'
INTERP_DIR.mkdir(parents=True, exist_ok=True)
RNG_SEED = 2026


def permute_together(df, columns, row_mask, seed):
    out = df.copy()
    idx = out.index[row_mask].to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(idx))
    values = out.loc[idx, columns].to_numpy(copy=True)
    out.loc[idx, columns] = values[order]
    return out


def standardize_column(df, column, raw_values, params):
    mean = params['means'][column]
    std = params['stds'][column]
    df.loc[raw_values.index, column] = (raw_values - mean) / std


def permute_ammonia(df, params, row_mask, seed):
    out = df.copy()
    idx = out.index[row_mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))

    ammonia_raw = out.loc[idx, 'ammonia__raw'].to_numpy(copy=True)[perm]
    nitrate_raw = out.loc[idx, 'nitrate_nitrite__raw'].to_numpy(copy=True)
    tp_raw = out.loc[idx, 'total_phosphorus__raw'].to_numpy(copy=True)

    ammonia_raw = pd.Series(ammonia_raw, index=idx)
    nitrate_raw = pd.Series(nitrate_raw, index=idx)
    tp_raw = pd.Series(tp_raw, index=idx)

    inorganic = ammonia_raw + nitrate_raw * 1000.0
    fraction = ammonia_raw / inorganic.replace(0, np.nan)
    ratio = inorganic / tp_raw.replace(0, np.nan)

    standardize_column(out, 'ammonia', ammonia_raw, params)
    standardize_column(out, 'ammonia_fraction_inorganic_n', fraction, params)
    standardize_column(out, 'inorganic_n_tp_ratio', ratio, params)
    return out


def permute_nitrate(df, params, row_mask, seed):
    out = df.copy()
    idx = out.index[row_mask]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(idx))

    ammonia_raw = out.loc[idx, 'ammonia__raw'].to_numpy(copy=True)
    nitrate_raw = out.loc[idx, 'nitrate_nitrite__raw'].to_numpy(copy=True)[perm]
    tp_raw = out.loc[idx, 'total_phosphorus__raw'].to_numpy(copy=True)

    ammonia_raw = pd.Series(ammonia_raw, index=idx)
    nitrate_raw = pd.Series(nitrate_raw, index=idx)
    tp_raw = pd.Series(tp_raw, index=idx)

    inorganic = ammonia_raw + nitrate_raw * 1000.0
    fraction = ammonia_raw / inorganic.replace(0, np.nan)
    ratio = inorganic / tp_raw.replace(0, np.nan)

    standardize_column(out, 'nitrate_nitrite', nitrate_raw, params)
    standardize_column(out, 'ammonia_fraction_inorganic_n', fraction, params)
    standardize_column(out, 'inorganic_n_tp_ratio', ratio, params)
    return out


def make_permuted_data(df, params, analysis, year=None):
    row_mask = df['split'].eq('test')
    if year is not None:
        row_mask &= df['year'].eq(year)

    if analysis == 'nitrogen_group':
        columns = [
            c for c in [
                'ammonia',
                'nitrate_nitrite',
                'river_nitrate',
                'ammonia_fraction_inorganic_n',
                'inorganic_n_tp_ratio',
            ]
            if c in df.columns
        ]
        return permute_together(df, columns, row_mask, RNG_SEED + (year or 0))

    if analysis == 'prior_toxin_history':
        columns = [c for c in ['mc_hist_1', 'mc_hist_2', 'mc_hist_3'] if c in df.columns]
        return permute_together(df, columns, row_mask, RNG_SEED + 10 + (year or 0))

    if analysis == 'lake_nitrate_nitrite':
        return permute_nitrate(df, params, row_mask, RNG_SEED + 20 + (year or 0))

    if analysis == 'lake_ammonia':
        return permute_ammonia(df, params, row_mask, RNG_SEED + 30 + (year or 0))

    raise ValueError(f'Unknown analysis: {analysis}')


def predictions_for_data(df, params):
    predictions = []
    for seed in SEEDS:
        model, _ = load_checkpoint(CKPT_DIR / f'hydrotoxnet_seed{seed}.pt', params)
        dataset = ForecastDataset(df, params, 'test')
        predictions.append(predict_split(model, dataset, seed, 'test'))
    return average_seed_predictions(pd.concat(predictions, ignore_index=True))


def importance_table(reference, permuted, analysis, year=None):
    rows = []
    for target in ['Elevated', 'HigherRisk']:
        for horizon in [7, 14, 21]:
            base = reference[
                (reference['target'] == target)
                & (reference['horizon'] == horizon)
            ]
            perm = permuted[
                (permuted['target'] == target)
                & (permuted['horizon'] == horizon)
            ]

            if year is not None:
                base = base[base['year'] == year]
                perm = perm[perm['year'] == year]

            keys = ['station', 'year', 'week_start', 'target', 'horizon', 'y_true']
            paired = base.merge(perm, on=keys, suffixes=('_base', '_perm'))
            y = paired['y_true'].to_numpy(dtype=int)
            base_score = safe_auprc(y, paired['y_prob_base'].to_numpy())
            perm_score = safe_auprc(y, paired['y_prob_perm'].to_numpy())

            rows.append({
                'analysis': analysis,
                'year': year if year is not None else 'all',
                'target': target,
                'horizon': horizon,
                'baseline_auprc': base_score,
                'permuted_auprc': perm_score,
                'importance': base_score - perm_score,
                'n': len(paired),
            })

    return pd.DataFrame(rows)


def main():
    df, params = load_data()
    reference_path = OUTDIR / 'predictions' / 'seed_averaged_predictions.csv'
    if not reference_path.exists():
        raise FileNotFoundError('Run model_evaluation.py first.')

    reference = pd.read_csv(reference_path)
    reference = reference[reference['split'] == 'test'].copy()

    analyses = [
        'nitrogen_group',
        'prior_toxin_history',
        'lake_nitrate_nitrite',
        'lake_ammonia',
    ]

    all_rows = []
    for analysis in analyses:
        print('Running:', analysis)
        permuted_df = make_permuted_data(df, params, analysis)
        permuted_predictions = predictions_for_data(permuted_df, params)
        permuted_predictions.to_csv(
            INTERP_DIR / f'{analysis}_permuted_predictions.csv', index=False
        )
        all_rows.append(
            importance_table(reference, permuted_predictions, analysis)
        )

    annual_rows = []
    for year in [2019, 2020, 2021, 2022]:
        for analysis in ['nitrogen_group', 'prior_toxin_history']:
            print('Running annual:', analysis, year)
            permuted_df = make_permuted_data(df, params, analysis, year=year)
            permuted_predictions = predictions_for_data(permuted_df, params)
            annual_rows.append(
                importance_table(reference, permuted_predictions, analysis, year=year)
            )

    summary = pd.concat(all_rows, ignore_index=True)
    annual = pd.concat(annual_rows, ignore_index=True)
    summary.to_csv(INTERP_DIR / 'permutation_importance.csv', index=False)
    annual.to_csv(INTERP_DIR / 'annual_permutation_importance_2019_2022.csv', index=False)

    print(summary.round(4).to_string(index=False))
    print('Saved:', INTERP_DIR)


if __name__ == '__main__':
    main()
