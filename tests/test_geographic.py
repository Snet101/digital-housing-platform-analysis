import numpy as np
import pandas as pd
import housing_model as h
import geographic_evaluation as e


def test_same_city_different_states_and_normalization():
    raw = pd.read_csv('tests/fixtures/listings.csv')
    raw.loc[0:19, 'cityname'] = '  Shared   City '
    raw.loc[0:9, 'state'] = ' ca '
    raw.loc[10:19, 'state'] = 'NY'
    df = h.clean(raw)
    assert {'shared city|CA', 'shared city|NY'} <= set(df.geo_id)
    simulated = h.simulate(df, 42)
    shares = simulated.groupby(['geo_id', 'platform']).size() / simulated.groupby('geo_id').size()
    expected = shares.pow(2).groupby('geo_id').sum()
    np.testing.assert_allclose(simulated.hhi, simulated.geo_id.map(expected))
    seen = []
    for _, train, test in e.folds(df):
        assert not set(df.loc[train, 'geo_id']) & set(df.loc[test, 'geo_id'])
        seen.extend(df.loc[test, 'row_id'])
    assert sorted(seen) == sorted(df.row_id)


def test_missing_state_excluded():
    raw = pd.read_csv('tests/fixtures/listings.csv')
    raw.loc[0, 'state'] = np.nan
    assert 0 not in set(h.clean(raw).row_id)


def test_evaluation_complete_and_mean_train_only(tmp_path):
    import json
    source = 'tests/fixtures/listings.csv'
    e.evaluate(source, tmp_path / 'evaluation')
    pred = pd.read_csv(tmp_path / 'evaluation/predictions.csv')
    df = h.clean(pd.read_csv(source))
    for number, train, _ in e.folds(df):
        values = pred.loc[(pred.model == 'mean') & (pred.fold == number), 'prediction']
        np.testing.assert_allclose(values, df.loc[train, 'price'].mean())
    assert pred.groupby('model').row_id.nunique().eq(len(df)).all()
    assert pred.groupby(['model', 'row_id']).size().eq(1).all()
    assert json.loads((tmp_path / 'evaluation/status.json').read_text())['state'] == 'complete'
