from pathlib import Path
import json
import pandas as pd
import torch

from model_training import (
    OUTDIR,
    SEEDS,
    ForecastDataset,
    load_data,
    train_one_seed,
)
from model_evaluation import (
    average_seed_predictions,
    select_thresholds,
    evaluate_predictions,
    predict_split,
    paired_bootstrap_auprc,
    add_holm_adjustment,
)

ABLATION_DIR = OUTDIR / 'ablations'
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    'without_ecological_susceptibility': {
        'flags': {'use_ecological': False},
    },
    'without_local_toxin_persistence': {
        'flags': {'use_toxin_history': False},
    },
    'without_spatial_transport_external_forcing': {
        'flags': {'use_spatial_external': False},
    },
    'without_physical_spatial_prior': {
        'flags': {'use_physical_prior': False},
    },
    'without_learned_correction': {
        'flags': {'use_edge_correction': False},
    },
    'without_nitrogen': {
        'flags': {},
        'remove_nitrogen': True,
    },
    'without_prior_toxin_history': {
        'flags': {},
        'remove_prior_history': True,
    },
}


def load_selected_config():
    with open(OUTDIR / 'selected_config.json') as f:
        return json.load(f)


def run_variant(name, settings, df, params, config):
    variant_dir = ABLATION_DIR / name
    checkpoint_dir = variant_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    flags = settings.get('flags', {})
    remove_nitrogen = settings.get('remove_nitrogen', False)
    remove_prior_history = settings.get('remove_prior_history', False)
    predictions = []

    for seed in SEEDS:
        checkpoint_path = checkpoint_dir / f'{name}_seed{seed}.pt'
        model, _, _ = train_one_seed(
            df,
            params,
            config,
            seed,
            flags=flags,
            remove_nitrogen=remove_nitrogen,
            remove_prior_history=remove_prior_history,
            save_path=checkpoint_path,
        )

        for split in ['validation', 'test']:
            dataset = ForecastDataset(
                df,
                params,
                split,
                remove_nitrogen=remove_nitrogen,
                remove_prior_history=remove_prior_history,
            )
            predictions.append(predict_split(model, dataset, seed, split))

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    seed_predictions = pd.concat(predictions, ignore_index=True)
    seed_predictions.to_csv(variant_dir / 'seed_predictions.csv', index=False)

    averaged = average_seed_predictions(seed_predictions)
    averaged.to_csv(variant_dir / 'seed_averaged_predictions.csv', index=False)

    validation = averaged[averaged['split'] == 'validation']
    test = averaged[averaged['split'] == 'test']
    thresholds = select_thresholds(validation)
    metrics = evaluate_predictions(test, thresholds)
    metrics.to_csv(variant_dir / 'test_metrics.csv', index=False)

    return test, metrics


def main():
    df, params = load_data()
    config = load_selected_config()

    reference_path = OUTDIR / 'predictions' / 'seed_averaged_predictions.csv'
    if not reference_path.exists():
        raise FileNotFoundError('Run model_evaluation.py first.')

    reference = pd.read_csv(reference_path)
    reference = reference[reference['split'] == 'test'].copy()

    all_metrics = []
    all_tests = []

    for name, settings in VARIANTS.items():
        print('Running:', name)
        test, metrics = run_variant(name, settings, df, params, config)
        metrics.insert(0, 'variant', name)
        all_metrics.append(metrics)

        comparison = paired_bootstrap_auprc(reference, test)
        comparison.insert(0, 'variant', name)
        comparison = add_holm_adjustment(comparison)
        comparison.to_csv(ABLATION_DIR / name / 'paired_bootstrap.csv', index=False)
        all_tests.append(comparison)

    pd.concat(all_metrics, ignore_index=True).to_csv(
        ABLATION_DIR / 'ablation_metrics.csv', index=False
    )
    pd.concat(all_tests, ignore_index=True).to_csv(
        ABLATION_DIR / 'ablation_paired_bootstrap.csv', index=False
    )

    print('Saved:', ABLATION_DIR)


if __name__ == '__main__':
    main()
