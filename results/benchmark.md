# FoodLoop ML benchmark results

## Scope

- Dataset: EPA U.S. Food Waste Flows Between Sectors, 2018 v1.3.2, method 2.
- Rows after cleaning: 8,315; producing-sector groups: 1,025.
- Target: `log1p(FlowAmount in kg)`.
- Split: grouped by producing-sector code; 5,682 train, 991 validation, 1,642 test rows.
- Model: ridge regression on two-, three-, and four-digit NAICS producer-sector prefixes plus consumed-sector code.
- Selected regularization: alpha = 10.

## Held-out test metrics

| Metric (log1p kg) | Ridge | Median baseline |
|---|---:|---:|
| MAE | 0.942 | 1.980 |
| RMSE | 1.258 | 2.425 |
| R-squared | 0.730 | -0.001 |

This fixed split holds out producer-sector codes from training. The result is an exploratory fit to EPA modeled 2018 national sector flows. It does not measure performance on current local surplus, donation transactions, or recipient acceptance. See `MODEL_CARD.md` and `data/README.md` before interpreting the numbers.
