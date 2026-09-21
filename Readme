# HydroToxNet: Forecasting Microcystin Risk in Western Lake Erie

HydroToxNet is a physics-guided spatiotemporal neural network for forecasting particulate microcystin risk in western Lake Erie 7, 14, and 21 days ahead. The model combines local ecological conditions, recent toxin history, information from neighboring monitoring stations, and river and meteorological forcing to predict Elevated and HigherRisk toxin conditions.

---

## How It Works

HydroToxNet combines three information sources. The **Ecological Susceptibility** component represents local water quality, bloom biomass, nutrient, and seasonal conditions. The **Local Toxin Persistence** component uses particulate microcystin measurements from the current week and the previous three weeks. The **Spatial Transport and External Forcing** component uses information from other monitoring stations together with river and meteorological conditions.

For each source and target station pair, the spatial component begins with a physics-guided prior based on source-to-target distance, wind alignment, and particulate microcystin concentration at the source station. A learned edge correction then adjusts these spatial relationships using observed data. The three model components are combined to generate Elevated and HigherRisk forecasts at 7, 14, and 21 days.

---

## Components

**1. Data Collection** — `data_collection.py`

Collects and combines NOAA GLERL/CIGLR lake monitoring data, USGS Maumee River data, NOAA NDBC wind data, and monitoring station information into the master dataset.

**2. Data Preprocessing** — `data_preprocessing.py`

Performs weekly aggregation, predictor coverage screening, Spearman correlation screening, derived feature construction, toxin-history construction, target generation, median imputation, and standardization using training data only.

**3. HydroToxNet Model** — `model.py`

Defines the Ecological Susceptibility, Local Toxin Persistence, Spatial Transport and External Forcing, learned edge correction, fusion network, and multi-horizon output heads.

**4. Model Training** — `model_training.py`

Runs the hyperparameter search and trains HydroToxNet using weighted masked binary cross-entropy, early stopping, and five random seeds.

**5. Model Evaluation** — `model_evaluation.py`

Averages predictions across the five seeds, selects classification thresholds using the validation data, computes evaluation metrics, and performs paired bootstrap tests with Holm adjustment.

**6. Ablation Study** — `ablation.py`

Removes individual model components and predictor groups to measure their contribution, including the ecological component, toxin-history component, spatial and external component, physical spatial prior, learned correction, nitrogen predictors, and prior toxin history.

**7. Baseline Comparison** — `baseline.py`

Compares HydroToxNet with Logistic Regression, Random Forest, XGBoost, LSTM, and Graph Attention Network models using the same chronological split, targets, and preprocessing procedure.

**8. Model Interpretation** — `model_interpretation.py`

Computes permutation importance for grouped nitrogen predictors, prior toxin history, lake nitrate plus nitrite, and lake ammonia. It also evaluates how predictor importance changes across toxin states, forecast horizons, and test years.

---

## Results

HydroToxNet predicts two microcystin risk levels:

- **Elevated:** particulate microcystin ≥ 1.0 µg/L
- **HigherRisk:** particulate microcystin ≥ 1.6 µg/L

| Target | 7 Days | 14 Days | 21 Days |
|---|---:|---:|---:|
| **Elevated AUPRC** | **0.845** | **0.839** | **0.700** |
| **HigherRisk AUPRC** | **0.716** | **0.634** | **0.495** |

HydroToxNet achieved the highest AUPRC across all six toxin-state and forecast-horizon settings compared with the evaluated baseline models.

---

## Data

The study uses western Lake Erie observations from **2012 through 2022**.

The eight lake monitoring stations used for model development are:

`WE02`, `WE04`, `WE06`, `WE08`, `WE09`, `WE12`, `WE13`, and `WE15`

Data sources include:

- NOAA GLERL/CIGLR lake monitoring data
- USGS Maumee River watershed data
- NOAA NDBC station 45005 wind data

The chronological data split is:

| Split | Years |
|---|---|
| Training | 2012–2017 |
| Validation | 2018 |
| Test | 2019–2022 |

The processed master dataset is:

`lake_erie_multimodal_master_2012_2022.csv`

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn torch xgboost
```

Requirements: Python, PyTorch, NumPy, pandas, SciPy, scikit-learn, and XGBoost.

The scripts are designed to run in **Google Colab** with Google Drive mounted at:

`/content/drive/MyDrive/hab_stgnn/`

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{liang2026hydrotoxnet,
  title={HydroToxNet: Forecasting Microcystin Risk in Western Lake Erie Reveals Changing Importance of Nitrogen and Toxin History},
  author={Liang, Ethan},
  year={2026}
}
```

Publication venue, DOI, and page information will be added after publication.
