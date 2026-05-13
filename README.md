# Demand Forecasting with Deep Neural Networks

Predicting hourly bike sharing demand using a deep MLP trained with backpropagation — comparing optimisers and loss functions.

---

## Overview

This project trains a deep Multilayer Perceptron (MLP) to predict hourly bike rental demand (`cnt`) from weather and calendar features. The focus is on the experimental methodology: systematically comparing optimiser and loss function choices to understand their effect on convergence and generalisation.

**Dataset:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — 17,379 hourly records across 2 years, with weather and calendar features.

---

## Experiments

Four controlled runs, holding architecture, preprocessing, and early stopping constant:

| Run | Optimiser | Loss | Val RMSE (log1p) |
|---|---|---|---|
| adam_mse_lr1e-3 | Adam | MSE | **0.505** |
| adam_mae_lr1e-3 | Adam | MAE | 0.619 |
| mom_mae_lr3e-4 | SGD + Momentum | MAE | 0.853 |
| mom_mse_lr3e-4 | SGD + Momentum | MSE | 1.020 |

**Key findings:**
- Adam consistently outperformed SGD with Momentum across both loss functions
- MSE slightly outperformed MAE — penalising large errors more heavily benefits demand spike prediction
- The model systematically underpredicts peak commuting hours (17:00), suggesting temporal lag features would improve performance

---

## Architecture

```
Input (61 features)
  → Linear(256) → ReLU
  → Linear(128) → ReLU
  → Linear(64)  → ReLU
  → Linear(1)
```

Gradient clipping (norm=1.0) and early stopping applied throughout.

---

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset and place at `data/hour.csv`:

```bash
mkdir -p data
wget -O bike_sharing.zip https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
unzip -o bike_sharing.zip
cp Bike-Sharing-Dataset/hour.csv data/hour.csv
```

Then run: `notebooks/demand_forecasting.ipynb`

---

## Stack

`torch` · `numpy` · `pandas` · `matplotlib` · `scikit-learn`
