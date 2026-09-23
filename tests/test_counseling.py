import pytest

from app.app import create_app
from contraception_risk import counseling
from contraception_risk.predict import RiskModel
from tests.test_app import BASE

COUNSEL = {'visit_type': 'start', 'method': 'pill', 'side_effects_now': '0', 'prior_disc': '1',
           'prior_method': 'pill', 'prior_reason': 'side_effects'}


@pytest.fixture(scope='module')
def model():
    return RiskModel()


def result(top_reason, tier='moyen'):
    return {'reasons': [('x', 0.6, top_reason), ('y', 0.4, 'autre')], 'tier': tier}


def test_counseling_only_fields_do_not_change_score(model):
    extra = {'visit_type': 'switch', 'side_effects_now': '1', 'prior_method': 'pill'}
    assert model.predict(BASE)['proba'] == model.predict(dict(BASE, **extra))['proba']


def test_prior_discontinuation_same_method_alerts():
    a = counseling.advise(COUNSEL, result('autre'))
    assert a['method']['label'] == 'Pilule'
    assert sum(level == 'warn' for level, _ in a['alerts']) == 2


def test_failure_with_user_dependent_method_suggests_larc():
    texts = [t for _, t in counseling.advise({'method': 'natural'}, result('echec'))['alerts']]
    assert any('longue durée' in t for t in texts)
    texts = [t for _, t in counseling.advise({'method': 'implant'}, result('echec'))['alerts']]
    assert not any('longue durée' in t for t in texts)


def test_no_method_no_card():
    a = counseling.advise({}, result('autre'))
    assert a['method'] is None and a['alerts'] == []


@pytest.mark.parametrize('bad', [{'method': 'xyz'}, {'visit_type': 'x'}, {'side_effects_now': '2'}])
def test_invalid_counseling_values(bad):
    with pytest.raises(ValueError):
        counseling.validate(bad)


def test_result_page_shows_method_and_alerts():
    c = create_app().test_client()
    html = c.post('/predict', data=dict(BASE, **COUNSEL)).get_data(as_text=True)
    assert 'Méthode : Pilule' in html and "Points d'attention" in html
    assert c.post('/predict', data=dict(BASE, method='xyz')).status_code == 400
