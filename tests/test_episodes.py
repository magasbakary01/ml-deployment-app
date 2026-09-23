import numpy as np
import pandas as pd

from training.episodes import episode_outcome, parse_calendar, woman_episodes


def cal(chrono, L=20):
    """Écrit un calendrier dans l'ordre chronologique (ancien -> récent) et le retourne au format DHS (récent d'abord)."""
    return chrono.ljust(L, '0')[:L][::-1]


def test_parse_orders_chronologically_and_dates():
    # 3 mois sans méthode, 5 mois de pilule arrêtée pour effets secondaires, puis non-utilisation
    v1 = cal('000' + '11111' + '0' * 12)
    v2 = cal('0' * 7 + '5' + '0' * 12)
    eps = parse_calendar(v1, v2, v017=1000, v019=20)
    assert len(eps) == 1
    e = eps[0]
    assert (e['method_code'], e['start_cmc'], e['end_cmc'], e['duration']) == ('1', 1003, 1007, 5)
    assert e['ended'] and e['reason_code'] == '5' and e['next_code'] == '0'
    assert episode_outcome(e) == ('arret_besoin', 'effets_secondaires')


def test_want_pregnancy_and_switch_are_not_need_stops():
    v1 = cal('0' + '333' + 'P' * 9 + 'B' + '0' * 6)
    v2 = cal('0' + '002' + '0' * 16)
    assert episode_outcome(parse_calendar(v1, v2, 1000, 20)[0])[0] == 'arret_sans_besoin'
    v1 = cal('0' + '111' + 'NNNNNNNNNNNNNNNN')
    v2 = cal('0' + '004' + '0' * 16)
    assert episode_outcome(parse_calendar(v1, v2, 1000, 20)[0])[0] == 'changement'


def test_censoring_and_continuation():
    # implant commencé 5 mois avant l'entretien : suivi < 12 mois -> exclu
    e = parse_calendar(cal('0' * 15 + 'NNNNN'), cal(''), 1000, 20)[0]
    assert not e['ended'] and episode_outcome(e) == (None, None)
    # injectable utilisé 14 mois puis arrêté : continuation à 12 mois
    e = parse_calendar(cal('00' + '3' * 14 + '0000'), cal('0' * 15 + '5'), 1000, 20)[0]
    assert episode_outcome(e)[0] == 'continue'


def test_woman_episodes_history_and_start_features():
    # pilule arrêtée (effets sec.) puis, après une naissance, injectable arrêté (mari)
    L = 40
    v1 = cal('0' + '1111' + '0' * 5 + 'B' + '000' + '3333' + '0' * 22, L)
    v2 = cal('0' * 4 + '5' + '0' * 12 + '3' + '0' * 22, L)
    w = {'caseid': 'x', 'vcal_1': v1, 'vcal_2': v2, 'v017': 1000, 'v019': L, 'v008': 1039, 'v011': 700,
         **{f'b3_{i:02d}': np.nan for i in range(1, 21)}, 'b3_01': 1010, 'b3_02': 950}
    rows = pd.DataFrame(woman_episodes(w))
    assert list(rows.method) == ['pill', 'injectable']
    assert list(rows.y) == [1, 1]
    assert list(rows.reason) == ['effets_secondaires', 'mari']
    assert list(rows.prior_need_stops) == [0, 1] and list(rows.prior_se_stops) == [0, 1]
    assert list(rows.parity_start) == [1, 2]
    assert rows.months_since_birth.iloc[1] == 1014 - 1010
    assert abs(rows.age_start.iloc[0] - (1001 - 700) / 12) < 1e-9


def test_left_censored_episode_excluded():
    w = {'caseid': 'x', 'vcal_1': cal('3' * 20), 'vcal_2': cal(''), 'v017': 1000, 'v019': 20, 'v008': 1019,
         'v011': 700, **{f'b3_{i:02d}': np.nan for i in range(1, 21)}}
    assert woman_episodes(w) == []


def test_learn_codebook_from_labels():
    from training.episodes import learn_codebook
    # DHS-8 : '4' = effets secondaires, 'M' = saignements ; méthode '3' = injectable
    rows = []
    for code, lab in [('4', 'side effects'), ('M', 'changes in menstrual bleeding'), ('2', 'wanted to become pregnant')]:
        for _ in range(6):
            rows.append({'vcal_1': cal('0' + '333' + '0' * 16), 'vcal_2': cal('0' * 3 + code + '0' * 16),
                         'v017': 1000, 'v019': 20, 'v359_lab': 'injections', 'v360_lab': lab})
    book, rep = learn_codebook(pd.DataFrame(rows))
    assert book['reason'] == {'4': 'effets_secondaires', 'M': 'effets_secondaires', '2': 'veut_grossesse'}
    assert book['method']['3'] == 'injectable'


def test_leading_blank_months_after_interview_are_skipped():
    # 3 mois postérieurs à l'entretien (vides), puis 20 mois de calendrier ; injectable en cours à l'entretien
    chrono = '0' * 10 + '3' * 10
    v1 = '   ' + chrono[::-1]
    eps = parse_calendar(v1, '   ' + '0' * 20, v017=1000, v019=20, v018=4)
    assert len(eps) == 1 and not eps[0]['ended'] and eps[0]['start_cmc'] == 1010 and eps[0]['end_cmc'] == 1019


def test_missing_reason_is_not_counted_as_need_stop():
    e = parse_calendar(cal('0' + '111' + '0' * 16), cal(''), 1000, 20)[0]
    assert episode_outcome(e)[0] == 'motif_inconnu'
