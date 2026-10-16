"""Reproducible rental prediction and explicitly synthetic model recovery."""

from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from lightgbm import LGBMRegressor, LGBMClassifier

OBSERVED_NUMERIC = ["bedrooms", "bathrooms", "square_feet", "description_length", "amenities_count"]
OBSERVED_CATEGORY = ["category", "cityname"]
SYNTHETIC_NUMERIC = OBSERVED_NUMERIC + ["price", "multihomed", "exposure_score", "platform_share", "hhi"]
SYNTHETIC_CATEGORY = OBSERVED_CATEGORY + ["platform"]
PLATFORMS = ["Craigslist", "Zillow", "Apartments.com"]
REQUIRED = [
    "price",
    "bedrooms",
    "bathrooms",
    "square_feet",
    "category",
    "cityname",
    "body",
    "amenities",
    "currency",
    "price_type",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def clean(raw):
    missing = set(REQUIRED) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    df = raw[REQUIRED].copy()
    df.insert(0, "row_id", np.arange(len(df)))
    for col in ["price", "bedrooms", "bathrooms", "square_feet"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in ["category", "cityname", "currency", "price_type"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    keep = (
        df.price.gt(0)
        & df.price.lt(10000)
        & df.square_feet.gt(0)
        & df.bedrooms.ge(0)
        & df.bathrooms.ge(0)
        & df.cityname.ne("")
        & df.category.ne("")
        & df.currency.str.upper().eq("USD")
        & df.price_type.str.casefold().eq("monthly")
    )
    df = df.loc[keep].copy()
    df["description_length"] = df.body.fillna("").astype(str).str.len()
    df["amenities_count"] = (
        df.amenities.fillna("").astype(str).map(lambda x: sum(bool(v.strip()) for v in x.split(",")))
    )
    # Identical retained source records must not count as independent observations.
    df = df.drop_duplicates(REQUIRED).reset_index(drop=True)
    if len(df) < 20 or df.cityname.nunique() < 4:
        raise ValueError("Need at least 20 cleaned records across four cities")
    return df


def split_cities(df, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train, test = next(splitter.split(df, groups=df.cityname))
    labels = np.full(len(df), "test", dtype=object)
    labels[train] = "train"
    assert not set(df.iloc[train].cityname) & set(df.iloc[test].cityname)
    return labels


def simulate(df, seed, curvature=10.0):
    """HHI uses listing counts only; no rent-derived aggregate is constructed."""
    rng = np.random.default_rng(seed)
    result = df.copy()
    cities = sorted(result.cityname.unique())
    mixes = dict(zip(cities, rng.dirichlet([1, 1, 1], size=len(cities)), strict=True))
    result["platform"] = [rng.choice(PLATFORMS, p=mixes[c]) for c in result.cityname]
    counts = result.groupby(["cityname", "platform"]).size().rename("count").reset_index()
    counts["platform_share"] = counts["count"] / counts.groupby("cityname")["count"].transform("sum")
    hhi = counts.assign(square=counts.platform_share**2).groupby("cityname").square.sum().rename("hhi")
    result = result.merge(
        counts[["cityname", "platform", "platform_share"]],
        on=["cityname", "platform"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.join(hhi, on="cityname", validate="many_to_one")
    assert len(result) == len(df) and result.row_id.is_unique
    result["landlord_id"] = rng.integers(1000, 2000, size=len(result))
    result["multihomed"] = (result.landlord_id % 7 == 0).astype(int)
    result["exposure_score"] = result.platform.map(dict(zip(PLATFORMS, [1.0, 1.5, 1.3], strict=True))) * (
        1 + 0.8 * result.multihomed
    )
    result["views"] = (result.exposure_score * 2000 / result.price + rng.normal(0, 40, len(result))).clip(lower=0)
    result["time_on_market"] = (
        28
        + curvature * (result.hhi - 0.5) ** 2
        - 3 * result.exposure_score
        + result.price / 500
        + rng.normal(0, 2, len(result))
    ).clip(1, 60)
    result["rented"] = (result.time_on_market < 20).astype(int)
    return result


def preprocessor(synthetic=False):
    numeric = SYNTHETIC_NUMERIC if synthetic else OBSERVED_NUMERIC
    category = SYNTHETIC_CATEGORY if synthetic else OBSERVED_CATEGORY
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=True), category),
        ],
        sparse_threshold=1.0,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_evaluate(df, seed):
    train, test = df.loc[df.split.eq("train")], df.loc[df.split.eq("test")]
    metrics, predictions = {}, df.loc[df.split.eq("test"), ["row_id", "cityname"]].copy()
    for target in ["price", "time_on_market", "views", "rented"]:
        binary = target == "rented"
        synthetic = target != "price"
        prep = preprocessor(synthetic)
        x_train = prep.fit_transform(train)
        x_test = prep.transform(test)
        settings = dict(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            verbosity=-1,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
        )
        if binary and train[target].nunique() < 2:
            model = DummyClassifier(strategy="prior")
        else:
            model = (LGBMClassifier if binary else LGBMRegressor)(**settings)
        baseline = DummyClassifier(strategy="prior") if binary else DummyRegressor(strategy="mean")
        model.fit(x_train, train[target])
        baseline.fit(x_train, train[target])

        def predict(estimator):
            if not binary:
                return estimator.predict(x_test)
            probabilities = estimator.predict_proba(x_test)
            return (
                probabilities[:, list(estimator.classes_).index(1)] if 1 in estimator.classes_ else np.zeros(len(test))
            )

        pred, base = predict(model), predict(baseline)
        result = {
            "kind": "synthetic_model_recovery" if synthetic else "observed_rent_prediction",
            "test_rows": len(test),
            "features": list(prep.get_feature_names_out()),
            "estimator": type(model).__name__,
        }
        if binary:
            defined = test[target].nunique() == 2
            result.update(
                auc=float(roc_auc_score(test[target], pred)) if defined else None,
                baseline_auc=float(roc_auc_score(test[target], base)) if defined else None,
                auc_note=None if defined else "Undefined: test partition has one class",
                accuracy=float(accuracy_score(test[target], pred >= 0.5)),
                baseline_accuracy=float(accuracy_score(test[target], base >= 0.5)),
            )
        else:
            result.update(
                rmse=float(np.sqrt(mean_squared_error(test[target], pred))),
                baseline_rmse=float(np.sqrt(mean_squared_error(test[target], base))),
            )
        predictions[f"{target}_actual"] = test[target].to_numpy()
        predictions[f"{target}_prediction"] = pred
        predictions[f"{target}_baseline"] = base
        metrics[target] = result
    return metrics, predictions


def ols_checks(df):
    """Alternate estimator on the exact saved training realization, not new data."""
    import statsmodels.formula.api as smf

    train = df.loc[df.split.eq("train")].copy()
    forms = {
        "observed_log_rent": "np.log(price) ~ bedrooms + bathrooms + square_feet + description_length + amenities_count",
        "synthetic_time_recovery": "time_on_market ~ I((hhi - 0.5)**2) + exposure_score + price",
    }
    rows = []
    for name, formula in forms.items():
        subsets = {"all_train": train}
        if name == "observed_log_rent":
            lo, hi = train.price.quantile([0.01, 0.99])
            subsets["trimmed_train"] = train.loc[train.price.between(lo, hi)]
        for subset, data in subsets.items():
            design = smf.ols(formula, data=data)
            rank = np.linalg.matrix_rank(design.exog)
            if len(data) <= design.exog.shape[1] + 1 or rank < design.exog.shape[1]:
                rows.append(dict(analysis=name, subset=subset, status="insufficient_rank_or_rows", n=len(data)))
                continue
            fit = design.fit(cov_type="HC3")
            for term in fit.params.index:
                se = float(fit.bse[term])
                rows.append(
                    dict(
                        analysis=name,
                        subset=subset,
                        term=term,
                        coefficient=float(fit.params[term]),
                        hc3_se=se if np.isfinite(se) else None,
                        n=int(fit.nobs),
                        status="ok" if np.isfinite(se) else "undefined_standard_error",
                    )
                )
    return rows


def run(input_path, out, seed=42, curvature=10.0):
    if seed < 0 or not np.isfinite(curvature):
        raise ValueError("Seed must be nonnegative and curvature finite")
    raw = pd.read_csv(input_path)
    cleaned = clean(raw)
    cleaned["split"] = split_cities(cleaned, seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "status.json", {"state": "incomplete"})
    try:
        df = simulate(cleaned, seed, curvature)
        df.to_csv(out / "simulation.csv", index=False)
        # OLS uses round-tripped saved input, also used by the notebook.
        df = pd.read_csv(out / "simulation.csv")
        metrics, predictions = fit_evaluate(df, seed)
        predictions.to_csv(out / "predictions.csv", index=False)
        save_json(out / "metrics.json", metrics)
        save_json(out / "ols.json", ols_checks(df))
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        save_json(
            out / "provenance.json",
            {
                "input_path": str(Path(input_path).resolve()),
                "input_sha256": digest(input_path),
                "code_sha256": digest(__file__),
                "git_commit": commit,
                "git_dirty": dirty,
                "simulation_sha256": digest(out / "simulation.csv"),
                "seed": seed,
                "curvature": curvature,
                "raw_rows": len(raw),
                "cleaned_rows": len(df),
                "excluded_or_duplicate_rows": len(raw) - len(df),
                "train_rows": int(df.split.eq("train").sum()),
                "test_rows": int(df.split.eq("test").sum()),
                "split": "held-out cities (25% of city groups); all tasks share row IDs",
                "python": platform.python_version(),
                "packages": {
                    p: importlib.metadata.version(p)
                    for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "statsmodels"]
                },
                "output_sha256": {p.name: digest(p) for p in out.iterdir() if p.name != "status.json"},
            },
        )
        save_json(out / "status.json", {"state": "complete"})
    except Exception as exc:
        save_json(out / "status.json", {"state": "failed", "error": str(exc)})
        raise
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="New output directory; existing directories are rejected"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curvature", type=float, default=10.0)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.out, args.seed, args.curvature), indent=2))


if __name__ == "__main__":
    main()
