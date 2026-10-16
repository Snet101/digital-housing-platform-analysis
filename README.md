# Reproducible rental prediction and platform simulation

This project combines observed apartment listings with an explicitly simulated platform environment. The original work was presented at the Stanford SC² Conference by Sanvi Nethikunta and Rithik Gumpu. This repair is a subsequent software revision; it does not retroactively validate the conference results.

## What is observed and what is generated?

| Variables | Meaning |
|---|---|
| Rent, bedrooms, bathrooms, square feet, category, city label, description, amenities | Observed only when using an actual UCI snapshot; all are invented in the offline fixture |
| Platform, landlord ID, multihoming, exposure, platform shares, HHI | Simulated allocation and derived simulation variables |
| Views, time on market, rented status | Generated outcomes, never observed rental performance |

The time generator is `clip(28 + curvature*(HHI-0.5)**2 - 3*exposure + price/500 + Normal(0,2), 1,60)`. Rental status is `time_on_market < 20`. Views include noise and are clipped at zero. A recovered concentration pattern reflects these assumptions, not evidence of a real platform effect. Platform names are illustrative labels, not observations of those companies.

## Run a small offline example

Python 3.12 is the validated interpreter. Use a new virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock.txt
python -m pip check
python -m pytest -q
python housing_model.py --input tests/fixtures/listings.csv --out runs/fixture
python scripts/check_notebook.py
```

On Linux/macOS activate with `source .venv/bin/activate`. Linux LightGBM may require the system OpenMP runtime (`libgomp1` on Ubuntu). The fixture contains 480 entirely invented rows. No network, external dataset, plotting window, or model download is required. Every output directory must be new; rerunning with the same output path fails rather than overwriting prior evidence.

`requirements.txt` specifies supported runtime ranges; `requirements-dev.txt` adds test/notebook tools. `requirements-dev.lock.txt` records the exact tested versions and platform-specific markers, without local paths. It is a version snapshot, not a hash-locked supply-chain guarantee.

## Use an actual UCI snapshot

The source is [Apartment for Rent Classified, UCI, DOI 10.24432/C5X623](https://archive.ics.uci.edu/dataset/555/apartment%2Bfor%2Brent%2Bclassified), licensed CC BY 4.0. Network acquisition is explicit and separate:

```powershell
python -m pip install ucimlrepo==0.0.7
python scripts/fetch_uci.py --out data/uci-snapshot
python housing_model.py --input data/uci-snapshot/input.csv --out runs/uci-seed42
```

The fetch saves input bytes, row count, source URL, and SHA256. Preserve `source.json` with the snapshot. Do not describe a fixture metric as a UCI result or claim an 80,000-row reproduction without the saved cleaned row count. Upstream availability and returned snapshot size may change.

## Evaluation boundary

The split holds out 25% of distinct city labels using a fixed seed. Cities do not overlap between training and test. A city label may conflate same-named cities in different states; it is not a verified neighborhood/geographic identifier. This evaluation tests prediction in unseen city labels, not future-time performance. Only positive rents below USD 10,000 with monthly price units, nonnegative beds/baths, positive square footage and nonempty category/city are retained. Missing descriptions and amenities become empty feature values. Listing duplicates are removed using the retained source columns; near-duplicate properties cannot be ruled out without reliable property identifiers.

Observed rent prediction uses only bedrooms, bathrooms, square feet, description length, amenities count, category and city label. It excludes **all** simulated variables and **all** rent-derived group statistics. Explicit feature lists replace the old broad prefix selector. One-hot categories and numeric imputers are fitted only on training rows; unseen test categories are handled without refitting.

Synthetic outcome models additionally use rent, simulated platform, multihoming, exposure, count-based platform share and HHI. These inputs are available in the simulated snapshot; no time, views or rented target is fed back into features. HHI describes the entire simulated city snapshot, including its listings, using counts alone. It is not a training-estimated rent statistic or a deployment estimate for unknown future listings. Simulation randomness and splitting are fixed by the recorded seed.

All tasks use identical train/test row IDs. Mean-regression and class-prior baselines accompany model metrics. AUC is null when the test set has one class; a single-class training set uses a constant classifier and is explicitly labeled. No test-set model tuning or causal interpretation is supported.

## Saved evidence

Each run saves:

- `simulation.csv`: cleaned row IDs, shared split, observed fields and generated realization.
- `predictions.csv`: held-out actual values, predictions and baseline predictions for every task.
- `metrics.json`: denominators, feature names, estimator, RMSE/AUC and baseline metrics.
- `ols.json`: observed log-rent regressions and trimmed training comparison, plus recovery of the known synthetic time formula.
- `provenance.json`: input/code/output hashes, seed, curvature, counts, versions and Git state when available.
- `status.json`: incomplete, failed or complete. Only complete runs are ready to analyze.

The notebook reads this same saved realization, verifies its hash, uses training rows only, and checks equality with saved OLS results. Set `HOUSING_RUN` to another complete run directory to inspect it. HC3 standard errors do not address all within-city or landlord dependence; coefficients are descriptive. Rank-deficient fits are recorded as unavailable.

## Sensitivity to assumptions

```powershell
python scripts/sensitivity.py --input tests/fixtures/listings.csv --out runs/sensitivity
```

This saves all nine combinations of seeds 41/42/43 and curvatures 0/10/20. Within each seed, the split, platform allocation and noise are shared across curvature assumptions. Across seeds, both the split and simulation change, so differences combine both effects. Do not select the best test run and report it as an unbiased estimate. Fixture runs test software, not market hypotheses.

## Historical files and deliberate scope

`docs/historical/` preserves the original script, README and notebook, including historical outputs. Their unseeded simulations, leaking rent-dispersion feature and separate OLS realization are not reproduced as validated results. The repaired CLI does not run those files. SHAP and smoothed partial-dependence plots are retained only in the historical script; they are not required dependencies or evidence of causal effects. The repaired version favors saved predictions, clear baselines, shared OLS data and assumption sensitivity.

See `docs/repair-notes.md` for verification and limits. GitHub Actions runs the offline checks after push.
