# Bike Sharing Demand Forecast

Hourly bike-rental forecasting with AutoGluon tabular AutoML for the Kaggle Bike Sharing Demand competition. The Kaggle score went from 1.86413 (default model) to 0.42063 (feature-engineered model).

## Results

| Stage | Model change | Validation RMSE (AutoGluon, best model) | Kaggle score |
|---|---|---|---|
| Initial | AutoGluon defaults, raw `count` target | 92.39 (count units) | 1.86413 |
| Secondary | `log1p(count)` target, datetime and flag features | 0.276 (log units) | **0.42063** |
| Tertiary | Interaction and categorical features, Bayesian tuning, `best_quality` preset | 0.268 (log units) | 0.42555 |

- Kaggle score: lower is better. The secondary model is the best Kaggle result.
- The tertiary model had a slightly lower validation RMSE than the secondary, but a slightly worse Kaggle score.
- Initial validation RMSE is on the raw count scale, so it is not directly comparable to the two log-scale rows.

![Kaggle scores by stage](graphs/kaggle%20scores.png)

## Problem

Predict the number of bike rentals (`count`) in each hour from the date, time and weather columns. Kaggle scores the predictions on the hidden test set.

## Data

Kaggle Bike Sharing Demand competition data, in `data/`:

- `train.csv`: 10,886 hourly rows from 2011-01-01 to 2012-12-19, with `casual`, `registered` and `count`.
- `test.csv`: 6,493 hourly rows from 2011-01-20 to 2012-12-31, with the same weather and date columns but no rental counts.

## Approach

- **Log target:** `count` is strongly right-skewed, so the model predicts `log1p(count)`. Predictions are mapped back with `expm1` and clipped at zero.
- **Features:** hour, day, month, weekday and year from the timestamp, plus `weekend_bool` and `rushhour_bool` (hours 7 to 9 and 16 to 19). `casual` and `registered` are dropped because the test set does not have them.
- **Interaction features:** temp × humidity, temp × windspeed and workingday × hour. In the tertiary run, weather, year, month and weekday are also cast to categorical.
- **Models:** AutoGluon `TabularPredictor` with RMSE as the metric. The secondary run uses the default presets. The tertiary run uses `presets='best_quality'` and Bayesian hyperparameter search (`searcher='bayesopt'`, 20 trials).
- **Best model:** an AutoGluon weighted ensemble (`WeightedEnsemble_L2`). In the secondary run its weights were CatBoost 0.48, NeuralNetTorch 0.28, LightGBMLarge 0.12, and 0.04 each for KNN, LightGBM and XGBoost.

![Hourly trends](graphs/hourly%20trends.png)

## Repository contents

| Path | Description |
|------|-------------|
| `sharingdemand.ipynb` | Main notebook: data loading, EDA, three AutoGluon runs, Kaggle submissions and result charts |
| `project_report.ipynb` | Written report, answering the project questions (renamed from `preoject_report.ipynb`) |
| `project_report.pdf` | Rendered version of the report |
| `data/` | Kaggle `train.csv` and `test.csv` |
| `graphs/` | EDA charts (for example `hourly trends.png`, `temp on rentals.png`) and the Kaggle score chart |
| `submissions/` | Kaggle submission files. The three scored submissions are `initial_submission.csv`, `secondary_submission.csv` and `tertiary_submission.csv`. `sampleSubmission.csv` is the Kaggle template. |

## How to run

This is a notebook-only project. There is no script or test suite. It was built with Python 3.10 and AutoGluon 1.2 on Windows.

```bash
pip install "autogluon~=1.2" pandas matplotlib seaborn jupyter kaggle
jupyter notebook sharingdemand.ipynb
```

Run the notebook from the repository root. Cell 15 reads `train.csv` and `test.csv` from the working directory, not from `data/`. Cells 3 and 7 point to a Windows Kaggle config path, so change them to your own path. Cells 9 to 11 download and unzip the competition data with the Kaggle CLI, which needs an API token at `~/.kaggle/kaggle.json`. The three `kaggle competitions submit` cells submit to the live competition, so skip them when re-running.

## Limitations and next steps

- Validation uses AutoGluon's default random holdout, not a time-based split, so validation scores may be optimistic for time-ordered data. A time-based split is the first thing to add.
- The notebook's Kaggle table and chart use 0.42091 for the tertiary model. The submission log in the notebook output shows 0.42555 for the saved `tertiary_submission.csv`.
- The features are not lagged. Adding lag and rolling-window demand features, and checking errors by hour and weather, is the next step.
- The environment is not pinned (there is no requirements file), and the notebook has hard-coded local paths.

Coursework for Udacity (final project submission; the course name is not recorded in this repo).
