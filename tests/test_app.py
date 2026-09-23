import numpy as np
import pandas as pd
import pytest

from app.app import create_app
from contraception_risk import features as F
from contraception_risk.predict import RiskModel

BASE = {'country': 'GN', 'region': 'GN_conakry', 'urban': '1', 'method': 'injectable', 'age': '27', 'cheb': '3',
        'kids_lt5': '2', 'months_since_birth': '8', 'prior_methods': '1', 'prior_disc': '0', 'marstat': 'married',
        'hus_pref': 'more', 'fertpref': 'more', 'dec_health': 'husband', 'hc_talkfp': 'no', 'religion': 'muslim',
        'edu_level': '0', 'working': '1', 'mobphone': '1', 'radio_fq': '2', 'fp_radio': '0', 'fp_tv': '0'}


@pytest.fixture(scope='module')
def model():
    return RiskModel()


@pytest.fixture()
def client():
    return create_app().test_client()


def test_form_columns_match_model(model):
    X = F.to_model_frame(F.from_form(BASE), model.regions)
    assert list(X.columns) == model.model.booster_.feature_name()


def test_method_changes_risk_as_expected(model):
    p = {m: model.predict(dict(BASE, method=m))['proba'] for m in ('implant', 'injectable', 'pill')}
    assert p['implant'] < p['injectable'] and p['implant'] < p['pill']


def test_prior_discontinuation_increases_risk(model):
    assert model.predict(dict(BASE, prior_disc='2', prior_reason='side_effects'))['proba'] > model.predict(BASE)['proba']


def test_result_structure(model):
    r = model.predict(BASE)
    assert 0 < r['proba'] < 1 and r['tier'] in ('faible', 'moyen', 'élevé')
    assert abs(sum(s for _, s, _ in r['reasons']) - 1) < 1e-6
    assert r['counsel'] and r['region_note'] is None


def test_excluded_region_is_flagged(model):
    r = model.predict(dict(BASE, region='GN_kankan'))
    assert 0 < r['proba'] < 1 and 'reste du pays' in r['region_note']


def test_unmarried_forces_niu():
    X = F.from_form(dict(BASE, marstat='never', hus_pref='more', dec_health='husband'))
    assert X.hus_pref[0] == 'niu' and X.dec_health[0] == 'niu'


def test_no_child_ignores_birth_interval():
    X = F.from_form(dict(BASE, cheb='0', kids_lt5='0', months_since_birth='12'))
    assert np.isnan(X.months_since_birth[0])


def test_optional_fields_can_be_missing(model):
    r = model.predict({k: BASE[k] for k in ['country', 'region', 'method', 'age']})
    assert 0 < r['proba'] < 1


@pytest.mark.parametrize('bad', [{'age': '60'}, {'region': 'SN_dakar'}, {'marstat': 'xyz'}, {'cheb': '25'},
                                 {'region': ''}, {'method': ''}, {'kids_lt5': '5'}, {'prior_disc': '9'}])
def test_invalid_input_rejected(model, bad):
    with pytest.raises(ValueError):
        model.predict(dict(BASE, **bad))


def test_episode_and_form_encodings_agree():
    E = pd.DataFrame([{'method': 'injectable', 'age_start': 27.0, 'parity_start': 3, 'kids_lt5_start': 2,
                       'months_since_birth': 8.0, 'prior_need_stops': 0, 'prior_se_stops': 0, 'prior_episodes': 1,
                       'v501': 1, 'v602': 1, 'v621': 2, 'v743a': 4, 'v394': 1, 'v395': 0, 'v130_lab': 'islam',
                       'v106': 0, 'v025': 1, 'v714': 1, 'v169a': 1, 'v158': 2, 'v384a': 0, 'v384b': 0,
                       'v000': 'GN7', 'v024_lab': 'conakry'}])
    a = F.from_episodes(E)[F.FEATURES].iloc[0]
    b = F.from_form(BASE)[F.FEATURES].iloc[0]
    pd.testing.assert_series_equal(a, b, check_names=False, check_dtype=False)


def test_pages(client):
    assert client.get('/').status_code == 200
    assert client.get('/modele').status_code == 200
    assert client.get('/health').json['status'] == 'ok'
    r = client.post('/predict', data=BASE)
    assert r.status_code == 200 and 'Risque' in r.get_data(as_text=True)
    assert client.post('/predict', data=dict(BASE, age='abc')).status_code == 400
