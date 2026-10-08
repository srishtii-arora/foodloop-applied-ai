"""Train an exploratory ridge benchmark on EPA 2018 national food-waste flows.

The target is EPA's estimated flow amount, not an observed donation outcome.
This script uses a grouped holdout by producing sector to test generalization
to sectors withheld from training. It is deliberately transparent and small.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "Food_Waste_national_2018_m2_v1.3.2_9b1bb41.csv"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SEED = 42
CATEGORICAL = ["naics_2", "naics_3", "naics_4", "consumed_sector"]


def clean_sector(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "UNKNOWN"


def prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA, low_memory=False)
    required = {"FlowAmount", "SectorProducedBy", "SectorConsumedBy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["FlowAmount"] = pd.to_numeric(df["FlowAmount"], errors="coerce")
    df = df.dropna(subset=["FlowAmount"]).loc[lambda x: x.FlowAmount >= 0].copy()
    producer = df["SectorProducedBy"].map(clean_sector)
    df["producer_code"] = producer
    df["naics_2"] = producer.str[:2]
    df["naics_3"] = producer.str[:3]
    df["naics_4"] = producer.str[:4]
    df["consumed_sector"] = df["SectorConsumedBy"].map(clean_sector)
    df["target_log1p_kg"] = np.log1p(df["FlowAmount"].astype(float))
    return df.reset_index(drop=True)


def grouped_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.array(sorted(df["producer_code"].unique()))
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(groups)
    n_test = max(1, round(len(groups) * 0.20))
    test_groups = set(shuffled[:n_test])
    remaining = shuffled[n_test:]
    n_valid = max(1, round(len(remaining) * 0.15))
    valid_groups = set(remaining[:n_valid])
    train_groups = set(remaining[n_valid:])
    g = df["producer_code"]
    train = np.flatnonzero(g.isin(train_groups).to_numpy())
    valid = np.flatnonzero(g.isin(valid_groups).to_numpy())
    test = np.flatnonzero(g.isin(test_groups).to_numpy())
    return train, valid, test


def encode(train: pd.DataFrame, other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    vocab = {col: sorted(train[col].fillna("UNKNOWN").astype(str).unique()) for col in CATEGORICAL}
    names = ["intercept"]
    offsets: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL:
        offsets[col] = {}
        for val in vocab[col]:
            offsets[col][val] = len(names)
            names.append(f"{col}={val}")

    def matrix(frame: pd.DataFrame) -> np.ndarray:
        x = np.zeros((len(frame), len(names)), dtype=np.float64)
        x[:, 0] = 1.0
        for col in CATEGORICAL:
            mapping = offsets[col]
            for row, value in enumerate(frame[col].fillna("UNKNOWN").astype(str).to_numpy()):
                index = mapping.get(value)
                if index is not None:
                    x[row, index] = 1.0
        return x

    return matrix(train), matrix(other), names


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0  # leave the intercept unpenalized
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    denom = np.sum((actual - actual.mean()) ** 2)
    return {
        "mae_log1p_kg": float(np.mean(np.abs(residual))),
        "rmse_log1p_kg": float(np.sqrt(np.mean(residual**2))),
        "r2_log1p_kg": float(1 - np.sum(residual**2) / denom) if denom else 0.0,
    }


def main() -> None:
    df = prepare()
    train_idx, valid_idx, test_idx = grouped_split(df)
    train = df.iloc[train_idx]
    valid = df.iloc[valid_idx]
    test = df.iloc[test_idx]

    x_train, x_valid, names = encode(train, valid)
    y_train = train["target_log1p_kg"].to_numpy()
    y_valid = valid["target_log1p_kg"].to_numpy()
    candidates = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        w = fit_ridge(x_train, y_train, alpha)
        candidates.append((metrics(y_valid, x_valid @ w)["rmse_log1p_kg"], alpha))
    validation_rmse, best_alpha = min(candidates)

    train_valid = pd.concat([train, valid], ignore_index=True)
    x_fit, x_test, names = encode(train_valid, test)
    y_fit = train_valid["target_log1p_kg"].to_numpy()
    y_test = test["target_log1p_kg"].to_numpy()
    weights = fit_ridge(x_fit, y_fit, best_alpha)
    predictions = x_test @ weights
    baseline = np.full_like(y_test, np.median(y_fit))

    pd.DataFrame({
        "producer_sector": test["producer_code"].to_numpy(),
        "consumed_sector": test["consumed_sector"].to_numpy(),
        "flow_amount_kg_actual": test["FlowAmount"].to_numpy(),
        "log1p_kg_actual": y_test,
        "log1p_kg_predicted": predictions,
        "log1p_kg_baseline": baseline,
        "absolute_log_error": np.abs(y_test - predictions),
    }).to_csv(OUT / "test_predictions.csv", index=False)

    metrics_out = {
        "project": "FoodLoop sector-flow regression benchmark",
        "dataset": "U.S. Food Waste Flows Between Sectors, 2018 v1.3.2, method 2",
        "target": "log1p(FlowAmount in kg)",
        "rows_after_cleaning": int(len(df)),
        "producing_sector_groups": int(df["producer_code"].nunique()),
        "split": "70/15/15 by producing-sector group (train/validation/test); seed 42",
        "n_train": int(len(train_idx)), "n_validation": int(len(valid_idx)), "n_test": int(len(test_idx)),
        "model": "ridge regression over one-hot NAICS sector prefixes and consumed-sector code",
        "tuning": {"alpha_candidates": [0.1, 1.0, 10.0, 100.0], "selected_alpha": best_alpha, "validation_rmse_log1p_kg": validation_rmse},
        "test_model": metrics(y_test, predictions),
        "test_median_baseline": metrics(y_test, baseline),
        "interpretation": "Exploratory fit to EPA modeled 2018 national flow estimates; not donation-level data or a forecast of current/local surplus.",
        "seed": SEED,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    report = f"""# FoodLoop ML benchmark results

## Scope

- Dataset: EPA U.S. Food Waste Flows Between Sectors, 2018 v1.3.2, method 2.
- Rows after cleaning: {len(df):,}; producing-sector groups: {df['producer_code'].nunique():,}.
- Target: `log1p(FlowAmount in kg)`.
- Split: grouped by producing-sector code; {len(train_idx):,} train, {len(valid_idx):,} validation, {len(test_idx):,} test rows.
- Model: ridge regression on two-, three-, and four-digit NAICS producer-sector prefixes plus consumed-sector code.
- Selected regularization: alpha = {best_alpha:g}.

## Held-out test metrics

| Metric (log1p kg) | Ridge | Median baseline |
|---|---:|---:|
| MAE | {metrics_out['test_model']['mae_log1p_kg']:.3f} | {metrics_out['test_median_baseline']['mae_log1p_kg']:.3f} |
| RMSE | {metrics_out['test_model']['rmse_log1p_kg']:.3f} | {metrics_out['test_median_baseline']['rmse_log1p_kg']:.3f} |
| R-squared | {metrics_out['test_model']['r2_log1p_kg']:.3f} | {metrics_out['test_median_baseline']['r2_log1p_kg']:.3f} |

This fixed split holds out producer-sector codes from training. The result is an exploratory fit to EPA modeled 2018 national sector flows. It does not measure performance on current local surplus, donation transactions, or recipient acceptance. See `MODEL_CARD.md` and `data/README.md` before interpreting the numbers.
"""
    (OUT / "benchmark.md").write_text(report, encoding="utf-8")
    model = {"intercept": float(weights[0]), "alpha": best_alpha, "feature_names": names, "weights": weights.tolist(), "target": "log1p(FlowAmount kg)"}
    (OUT / "ridge_model.json").write_text(json.dumps(model), encoding="utf-8")
    coef = pd.DataFrame({"feature": names, "coefficient": weights}).sort_values("coefficient", key=abs, ascending=False)
    coef.head(40).to_csv(OUT / "top_coefficients.csv", index=False)
    print(json.dumps(metrics_out, indent=2))


if __name__ == "__main__":
    main()
