# Review guide: three focused improvements, October 18, 2026

These changes were implemented with AI assistance after the original 2025 SC² Student Conference presentation at Stanford. Review the code and saved outputs before using this revision in an application or interview. They do not validate the old paper or poster's numerical claims.

## 1. Geographic identity

Previously, `clean()` discarded state and `split_cities()` and `simulate()` treated city name alone as geography. Springfield in two states could therefore become one group. Now `housing_model.py` preserves state, trims it and uppercases it, normalizes city whitespace and case, and constructs `geo_id = cityname + '|' + state`. Empty city or missing/malformed two-letter state is excluded. This checks format, not whether a postal abbreviation is an actual jurisdiction; the invented fixture explicitly uses ZZ.

City/state identity now controls train/test groups, simulated platform allocation, shares and HHI. Both city/state ID and state enter the observed category features: an unseen city/state is encoded as unknown, while its state can be learned from other training cities. No held-out rents or rent aggregates enter predictors. Saved predictions retain row ID, normalized city, state and geographic ID. Deduplication includes state, preventing different states from being collapsed simply because other retained values match. IDs are input-row positions, not verified property identifiers.

`tests/fixtures/listings.csv` now has an explicitly invented state. Historical source files are unchanged. New tests cover same-named cities in different states, normalization, state exclusions, state-aware HHI and split boundaries. Existing tests still cover shared simulation/OLS data and train-only preprocessing.

## 2. Fixed multi-fold evaluation

`geographic_evaluation.py` adds an observed-only evaluation, separate from simulated targets. A fixed seed (42) shuffles sorted unique city/state groups into five folds. Every retained listing is tested once; its entire geographic group stays out of that fold's training set. Folds balance counts of geographic labels, not numbers of listings. This is five geographic folds, not five independent repeated experiments. Training sets overlap.

The protocol fixes 100-tree LightGBM settings and Ridge alpha=1, alongside training-mean and training-median baselines. There is no tuning, early stopping or choosing the best fold. Imputation, one-hot encoding, scaling and models are fitted on training rows only. Ridge uses sparse standard scaling (including one-hot columns), without centering, and the LSQR solver. Settings were specified in code before the real-data run; `docs/geographic-protocol.json` preserves them.

Reports include all fold MAE/RMSE, pooled listing-weighted errors, equal-city-weighted MAE, errors for every city/state with sample counts, and five fixed bands of actual asking rent. Band membership uses the held-out target only to diagnose errors, never to construct model features. Small-city errors can be unstable. Fold ranges are descriptive and are not confidence intervals. The command writes an incomplete/complete/failed status and input/code/output hashes and dependency versions.

### Real UCI snapshot results

99,826 raw rows; 98,438 cleaned rows; 3,549 city/state groups. There were 305 missing-state rows in the raw snapshot and 353 retained city names occurring in multiple states. The 1,388 total removals combine all filters and deduplication; the missing-state count is not an additional disjoint exclusion count. The retained row count happens to match the prior revision; that does not imply the groups or splits match.

| Model | Pooled MAE (USD/month) | Pooled RMSE | Fold RMSE range |
|---|---:|---:|---:|
| LightGBM | 358.74 | 557.31 | 450.96–642.12 |
| Ridge | 394.58 | 586.53 | 476.45–691.17 |
| Training mean | 532.12 | 759.85 | 654.42–905.93 |
| Training median | 512.67 | 775.87 | 640.15–943.13 |

LightGBM's equal-city-weighted MAE is $395.18. Its MAE in the $3,000–$9,999.99 band is $1,648.57 across 4,293 listings, so the aggregate score conceals substantial expensive-listing error. It does not win every rent band: the mean baseline has lower MAE in the $1,500–$1,999.99 band. Do not present this as uniformly accurate.

The previous single-split RMSE of $589.26 versus $671.92 for its mean baseline used different grouping and held-out rows. Comparing those values with the new pooled values is not a controlled estimate of improvement. The defensible conclusion is that this fixed geographic evaluation beats the simple baselines overall on this snapshot, with meaningful geographic and price-band variation. It is not a test on future listings or a separately reserved final dataset.

## 3. Clearer presentation and attribution

README now leads with asking-rent prediction, distinguishes advertised rent from transactions, and keeps platform behavior/outcomes explicitly simulated. The original student-conference presentation and later AI-assisted software maintenance are distinguished. Simulated time-on-market, rented status and views support checking whether code recovers its own assumptions, not claims that platforms cause real rental outcomes. The original unpublished paper is not newly validated.

For an interview, explain what you originally built, the original leakage/simulation limitations, and what you learned by reviewing these subsequent repairs. Do not describe all later AI-assisted implementation as work you independently wrote in 2025.

## Review and reproduce

1. Read `housing_model.py`: `clean`, `split_cities`, `simulate`, `preprocessor`.
2. Read `geographic_evaluation.py`: `PROTOCOL`, `folds`, `evaluate`.
3. Read `tests/test_geographic.py` and the updated geographic assertions in `tests/test_housing.py`.
4. Run `python -m pytest -q` and `python -m ruff check housing_model.py geographic_evaluation.py scripts tests`.
5. Run `python geographic_evaluation.py --input YOUR_UCI_SNAPSHOT.csv --out runs/geographic` in a new output directory. The original source snapshot SHA-256 is `dba71c5b6947a7524fc4bd4ab597daaa5f2c40ac1b0435129e392add607be291`.
6. Open `metrics.json`, `errors-by-city.csv`, `errors-by-rent-band.csv` and `cleaning-audit.json`. `predictions.csv` contains every held-out prediction and can reproduce the aggregate metrics.

The GitHub workflow adds a fixture geographic run and linting of the new evaluator. Local validation and delivery details are recorded in the accompanying `REVIEW-START-HERE.md`; hosted CI remains pending a push.
