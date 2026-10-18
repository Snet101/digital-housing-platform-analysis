"""Predeclared, observed asking-rent evaluation; no simulated targets."""
import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from housing_model import clean, digest, preprocessor, save_json

PROTOCOL = {
    "folds": 5, "seed": 42, "split": "shuffled KFold over sorted unique city/state labels",
    "ridge": {"alpha": 1.0, "solver": "lsqr", "tol": 0.0001},
    "lightgbm": {"n_estimators": 100, "learning_rate": 0.05, "num_leaves": 15,
                 "min_child_samples": 10, "random_state": 42, "n_jobs": 1,
                 "deterministic": True, "force_col_wise": True, "verbosity": -1},
    "rent_band_edges": [0, 1000, 1500, 2000, 3000, 10000],
    "selection": "All folds reported; no tuning, early stopping, or split selection",
}


def folds(df):
    groups = np.array(sorted(df.geo_id.unique()))
    if len(groups) < PROTOCOL["folds"]:
        raise ValueError("Need at least five city/state groups")
    for number, (train, test) in enumerate(KFold(5, shuffle=True, random_state=42).split(groups)):
        yield number, df.geo_id.isin(groups[train]), df.geo_id.isin(groups[test])


def scores(frame):
    error = frame.prediction - frame.actual
    return {"rows": len(frame), "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2)))}


def evaluate(source, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "status.json", {"state": "incomplete"})
    save_json(out / "protocol.json", PROTOCOL)
    try:
        raw = pd.read_csv(source)
        df = clean(raw)
        audit = {"raw_rows": len(raw), "cleaned_rows": len(df), "removed_rows": len(raw)-len(df),
                 "missing_state_rows_raw": int(raw.state.isna().sum()),
                 "geographic_groups": df.geo_id.nunique(),
                 "city_names_in_multiple_states": int((df.groupby("cityname").state.nunique() > 1).sum())}
        save_json(out / "cleaning-audit.json", audit)
        frames = []
        for number, train_mask, test_mask in folds(df):
            train, test = df.loc[train_mask], df.loc[test_mask]
            prep = preprocessor()
            x_train, x_test = prep.fit_transform(train), prep.transform(test)
            # Sparse scaling is fitted only on training data, including one-hot columns.
            scale = StandardScaler(with_mean=False)
            ridge_train, ridge_test = scale.fit_transform(x_train), scale.transform(x_test)
            models = {"mean": DummyRegressor(strategy="mean"),
                      "median": DummyRegressor(strategy="median"),
                      "ridge": Ridge(**PROTOCOL["ridge"]),
                      "lightgbm": LGBMRegressor(**PROTOCOL["lightgbm"])}
            for name, model in models.items():
                a, b = (ridge_train, ridge_test) if name == "ridge" else (x_train, x_test)
                model.fit(a, train.price)
                frame = test[["row_id", "cityname", "state", "geo_id"]].copy()
                frame["fold"], frame["model"] = number, name
                frame["actual"], frame["prediction"] = test.price, model.predict(b)
                frames.append(frame)
        predictions = pd.concat(frames, ignore_index=True)
        predictions.to_csv(out / "predictions.csv", index=False)
        fold_scores = [{"model": m, "fold": int(f), **scores(g)}
                       for (m, f), g in predictions.groupby(["model", "fold"])]
        summary = {m: scores(g) for m, g in predictions.groupby("model")}
        for m in summary:
            for metric in ["mae", "rmse"]:
                values = [r[metric] for r in fold_scores if r["model"] == m]
                summary[m][metric + "_fold_range"] = [min(values), max(values)]
        save_json(out / "metrics.json", {"pooled": summary, "folds": fold_scores})
        city = [{"model": m, "geo_id": geo, **scores(g)}
                for (m, geo), g in predictions.groupby(["model", "geo_id"])]
        pd.DataFrame(city).to_csv(out / "errors-by-city.csv", index=False)
        predictions["rent_band"] = pd.cut(predictions.actual, PROTOCOL["rent_band_edges"], right=False)
        bands = [{"model": m, "rent_band": str(band), **scores(g)}
                 for (m, band), g in predictions.groupby(["model", "rent_band"], observed=True)]
        pd.DataFrame(bands).to_csv(out / "errors-by-rent-band.csv", index=False)
        save_json(out / "macro-city-mae.json", pd.DataFrame(city).groupby("model").mae.mean().to_dict())
        save_json(out / "provenance.json", {
            "input_sha256": digest(source), "evaluation_sha256": digest(__file__),
            "housing_model_sha256": digest(Path(__file__).with_name("housing_model.py")),
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm"]},
            "outputs": {p.name: digest(p) for p in out.iterdir() if p.name != "status.json"},
        })
        save_json(out / "status.json", {"state": "complete"})
    except Exception as exc:
        save_json(out / "status.json", {"state": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.input, args.out)
