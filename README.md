# FoodLoop | Applied AI portfolio project

FoodLoop explores how data analysis and a human-reviewed matching workflow could support local food rescue (SDG 2). It is an internship portfolio project with two clearly separated parts:

1. **ML benchmark:** a reproducible ridge-regression study on the U.S. EPA's modeled 2018 food-waste flows by sector.
2. **Applied AI demo:** a responsive offer-to-recipient workflow with hard eligibility filters, explainable rules-based ranking, and human confirmation.

The ML benchmark is not a donation-acceptance model and does not drive the demo's recipient ranking. The included dataset is modeled national sector-flow data, not current local surplus or donation transactions.

## Quick start

### Open the demo

- Windows: double-click `run_foodloop.bat`.
- Any recent browser: open `index.html`.
- Or run `python -m http.server 8000` from this folder and visit <http://localhost:8000>.

The browser demo has no runtime dependencies. Use an offer category, quantity, pickup window, and area to see eligible recipients ranked by a visible weighted score. Selecting a match displays a handoff reminder. The app sends no messages and records no real donations.

## Reproduce the ML benchmark

Use Python 3.10 or later. From this folder:

```bash
python -m pip install -r requirements.txt
python analysis/train_model.py
```

On Windows, `train_model.bat` installs the requirements and runs the pipeline. The raw EPA CSVs are included under `data/raw/`, so training needs no data download after dependencies are installed.

The script cleans `FlowAmount`, derives two-, three-, and four-digit producer-sector prefixes, and predicts `log1p(FlowAmount in kg)`. It uses a deterministic 70/15/15 split by producing-sector groups, selects ridge regularization on validation data, then compares the final test result with a median-only baseline. Results are saved in `results/metrics.json`, predictions in `results/test_predictions.csv`, coefficients in `results/top_coefficients.csv`, and the fitted weights in `results/ridge_model.json`.

### Checked-in benchmark result

On the included EPA method 2 file, the grouped test set contains 1,642 rows from producing-sector groups held out from training. The ridge model reports MAE **0.942** and RMSE **1.258** on the log1p target, versus **1.980** and **2.425** for the median baseline; test R² is **0.730** versus **-0.001**. These are one fixed split on EPA's modeled 2018 estimates. They do not establish accuracy for present-day local food waste, donated quantities, or recipient acceptance.

## Data and limitations

The included files are EPA Flow-By-Sector method 1 and method 2 outputs. The benchmark trains on method 2. EPA describes the data as modeled national 2018 flows attributed to waste-generating and waste-management sectors. See [`data/README.md`](data/README.md) for source links, license, and citation. See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use and limitations.

Because the source is aggregated and covers one modeled year, this project cannot answer how much food a particular donor will offer, whether a recipient will accept it, or whether a pickup will be safe or completed. A real operational model would need permissioned event-level partner data, a time-based evaluation, and food-safety and equity review. Until then, the demo uses explicit matching rules and human decisions.

## Screenshots and deliverables

- `screenshots/` contains screenshots captured from the running demo.
- `deliverables/` contains the Lean Canvas PDF, concept note, and presentation.

## Publish to GitHub

1. Create an empty repository on GitHub, for example `foodloop-applied-ai`.
2. In this project folder, run:

```bash
git init
git add .
git commit -m "Build FoodLoop applied AI portfolio project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/foodloop-applied-ai.git
git push -u origin main
```

Replace the URL with the repository URL GitHub provides. To publish the browser demo too, enable **Settings → Pages → Deploy from a branch**, choose `main` and `/(root)`, and save. Make the repository public only if your submission needs an accessible public link.
