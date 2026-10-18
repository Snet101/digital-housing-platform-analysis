import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import housing_model as h

FIXTURE = Path(__file__).parent / "fixtures/listings.csv"


@pytest.fixture
def data():
    df = h.clean(pd.read_csv(FIXTURE))
    df["split"] = h.split_cities(df, 42)
    return df


def test_seed_reproducibility_and_shared_population(data):
    first = h.simulate(data, 42)
    pd.testing.assert_frame_equal(first, h.simulate(data, 42))
    assert not first.platform.equals(h.simulate(data, 43).platform)
    assert first.row_id.tolist() == data.row_id.tolist()
    assert first.split.tolist() == data.split.tolist()


def test_city_boundary(data):
    assert not set(data.loc[data.split.eq("train"), "geo_id"]) & set(data.loc[data.split.eq("test"), "geo_id"])


def test_hhi_counts_not_listing_weighted(data):
    df = h.simulate(data, 42)
    shares = df.groupby(["geo_id", "platform"]).size() / df.groupby("geo_id").size()
    expected = shares.pow(2).groupby("geo_id").sum()
    np.testing.assert_allclose(df.hhi, df.geo_id.map(expected))
    assert df.hhi.between(1 / 3 - 1e-12, 1).all()
    assert df.platform_share.between(0, 1).all()


def test_validation_rents_cannot_change_features(data):
    changed = data.copy()
    changed.loc[changed.split.eq("test"), "price"] *= 3
    first, second = h.simulate(data, 42), h.simulate(changed, 42)
    a, b = h.preprocessor(), h.preprocessor()
    x = a.fit_transform(first.loc[first.split.eq("train")])
    y = b.fit_transform(second.loc[second.split.eq("train")])
    np.testing.assert_array_equal(x.toarray(), y.toarray())
    np.testing.assert_array_equal(a.transform(first).toarray(), b.transform(second).toarray())
    assert all(
        "price" not in name and "platform" not in name and "hhi" not in name for name in a.get_feature_names_out()
    )


def test_preprocessing_unseen_category_and_no_test_fit(data):
    train, test = data.loc[data.split.eq("train")], data.loc[data.split.eq("test")].copy()
    prep = h.preprocessor()
    x = prep.fit_transform(train)
    test["category"] = "never_seen"
    transformed = prep.transform(test)
    assert transformed.shape[1] == x.shape[1]
    assert np.isfinite(transformed.data).all()
    assert "never_seen" not in prep.get_feature_names_out().tolist()


def test_cleaning_empty_amenities_and_duplicates():
    raw = pd.read_csv(FIXTURE)
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    result = h.clean(raw)
    assert len(result) == 480 and result.row_id.is_unique
    assert result.iloc[0].amenities_count == 0
    raw.loc[0, "price"] = np.inf
    assert len(h.clean(raw)) == 480  # duplicate valid original is retained instead


@pytest.mark.parametrize("column", ["price", "cityname", "state", "body"])
def test_missing_schema_rejected(column):
    with pytest.raises(ValueError, match="Missing"):
        h.clean(pd.read_csv(FIXTURE).drop(columns=column))


def test_curvature_is_assumption_not_discovery(data):
    a, b = h.simulate(data, 42, 0), h.simulate(data, 42, 10)
    np.testing.assert_allclose(b.time_on_market - a.time_on_market, 10 * (a.hhi - 0.5) ** 2)
    pd.testing.assert_series_equal(a.price, b.price)


def test_single_class_auc_explicit(data):
    df = h.simulate(data, 42)
    df["rented"] = 0
    metrics, predictions = h.fit_evaluate(df, 42)
    assert metrics["rented"]["auc"] is None
    assert metrics["rented"]["estimator"] == "DummyClassifier"
    assert predictions.rented_prediction.eq(0).all()


def test_run_outputs_repeat_and_overwrite(tmp_path):
    a, b = tmp_path / "first", tmp_path / "second"
    assert h.run(FIXTURE, a) == h.run(FIXTURE, b)
    assert (a / "simulation.csv").read_bytes() == (b / "simulation.csv").read_bytes()
    assert (a / "predictions.csv").read_bytes() == (b / "predictions.csv").read_bytes()
    assert json.loads((a / "status.json").read_text())["state"] == "complete"
    meta = json.loads((a / "provenance.json").read_text())
    assert meta["cleaned_rows"] == 480 and meta["train_rows"] + meta["test_rows"] == 480
    assert meta["simulation_sha256"] == h.digest(a / "simulation.csv")
    assert h.ols_checks(pd.read_csv(a / "simulation.csv")) == json.loads((a / "ols.json").read_text())
    with pytest.raises(FileExistsError):
        h.run(FIXTURE, a)


def test_failure_status(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(h, "fit_evaluate", fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        h.run(FIXTURE, out)
    assert json.loads((out / "status.json").read_text())["state"] == "failed"


def test_only_monthly_usd_rents():
    raw = pd.read_csv(FIXTURE)
    raw.loc[0, "currency"] = "EUR"
    raw.loc[1, "price_type"] = "Weekly"
    df = h.clean(raw)
    assert len(df) == 478
    assert not {0, 1} & set(df.row_id)


def test_historical_files_preserved():
    root = Path(__file__).resolve().parents[1]
    hashes = json.loads((root / "docs/historical-sha256.json").read_text())
    for name, expected in hashes.items():
        assert h.digest(root / "docs/historical" / name) == expected
