"""Définition unique des variables du modèle, partagée par l'entraînement et l'application.

Le modèle prédit, au moment où une femme démarre (ou poursuit) une méthode, le risque qu'elle l'arrête dans
les 12 mois alors qu'elle en a encore besoin. Deux sources produisent le même tableau de variables :
  - ``from_episodes`` : épisodes d'utilisation tirés du calendrier DHS (``training/episodes.py``) ;
  - ``from_form``     : formulaire rempli en consultation.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

COUNTRY_NAMES = {'BF': 'Burkina Faso', 'CI': "Côte d'Ivoire", 'GH': 'Ghana', 'GN': 'Guinée', 'ML': 'Mali',
                 'NG': 'Nigeria', 'SN': 'Sénégal'}

CATEGORIES = {
    'method': ['pill', 'injectable', 'implant', 'iud', 'condom', 'lam', 'natural', 'other'],
    'marstat': ['never', 'married', 'union', 'widowed', 'divorced', 'separated'],
    'fertpref': ['more', 'undecided', 'nomore', 'infecund'],
    'hus_pref': ['same', 'more', 'fewer', 'dk', 'niu'],
    'dec_health': ['alone', 'joint', 'husband', 'other', 'niu'],
    'hc_talkfp': ['no', 'yes', 'novisit'],
    'religion': ['muslim', 'christian', 'traditional', 'none', 'other'],
    'country': list(COUNTRY_NAMES),
}
IN_UNION = {'married', 'union'}

# Variables décrivant l'épisode et son contexte au démarrage (issues du calendrier)
START_FEATURES = ['method', 'age', 'cheb', 'kids_lt5', 'months_since_birth', 'prior_stops', 'prior_se_stops',
                  'prior_methods']
# Caractéristiques de la femme (mesurées à l'enquête dans les données d'entraînement)
WOMAN_FEATURES = ['marstat', 'fertpref', 'hus_pref', 'dec_health', 'hc_talkfp', 'religion', 'edu_level', 'urban',
                  'working', 'mobphone', 'radio_fq', 'fp_radio', 'fp_tv']
CONTEXT_FEATURES = ['country', 'region']
FEATURES = START_FEATURES + WOMAN_FEATURES + CONTEXT_FEATURES

LABELS = {
    'method': "Méthode choisie", 'age': "Âge", 'cheb': "Nombre d'enfants nés",
    'kids_lt5': "Enfants nés au cours des 5 dernières années", 'months_since_birth': "Délai depuis la dernière naissance",
    'prior_stops': "Arrêts antérieurs d'une méthode", 'prior_se_stops': "Arrêts antérieurs pour effets secondaires",
    'prior_methods': "Méthodes utilisées auparavant", 'marstat': "Situation matrimoniale",
    'fertpref': "Désir d'autres enfants", 'hus_pref': "Souhait du mari/partenaire",
    'dec_health': "Décision sur sa propre santé", 'hc_talkfp': "PF abordée lors d'une visite au centre de santé",
    'religion': "Religion", 'edu_level': "Niveau d'instruction", 'urban': "Milieu de résidence",
    'working': "Activité professionnelle", 'mobphone': "Téléphone portable", 'radio_fq': "Écoute de la radio",
    'fp_radio': "Message PF entendu à la radio", 'fp_tv': "Message PF entendu à la télévision",
    'zone': "Pays et région",
}
CAPS = {'prior_stops': 3, 'prior_se_stops': 2, 'prior_methods': 4}


def slug(text):
    t = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '-', t).strip('-')


def region_key(country, label):
    return f"{country}_{slug(label)}"


def religion_category(label):
    t = str(label).lower()
    if re.search(r'musl|islam', t):
        return 'muslim'
    if re.search(r'christ|cathol|protest|pentecos|evangel|method|presbyt|anglic|advent|apostol|jehov|mormon', t):
        return 'christian'
    if re.search(r'tradition|animis|spiritual', t):
        return 'traditional'
    if re.search(r'no religion|none|sans', t):
        return 'none'
    return 'other' if t not in ('nan', 'none', '') else np.nan


def _na(s, codes):
    return s.where(~s.isin(codes))


# ---------------------------------------------------------------- épisodes DHS -> variables
def from_episodes(E: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=E.index)
    X['method'] = E.method
    X['age'] = E.age_start
    X['cheb'] = E.parity_start.astype(float)
    X['kids_lt5'] = E.kids_lt5_start.astype(float)
    X['months_since_birth'] = E.months_since_birth
    X['prior_stops'] = E.prior_need_stops.clip(upper=CAPS['prior_stops']).astype(float)
    X['prior_se_stops'] = E.prior_se_stops.clip(upper=CAPS['prior_se_stops']).astype(float)
    X['prior_methods'] = E.prior_episodes.clip(upper=CAPS['prior_methods']).astype(float)
    X['marstat'] = E.v501.map({0: 'never', 1: 'married', 2: 'union', 3: 'widowed', 4: 'divorced', 5: 'separated'})
    X['fertpref'] = E.v602.map({1: 'more', 2: 'undecided', 3: 'nomore', 5: 'infecund'})
    in_union = X.marstat.isin(IN_UNION)
    X['hus_pref'] = E.v621.map({1: 'same', 2: 'more', 3: 'fewer', 8: 'dk'}).where(in_union, 'niu')
    X['dec_health'] = E.v743a.map({1: 'alone', 2: 'joint', 3: 'other', 4: 'husband', 5: 'other', 6: 'other'}
                                  ).where(in_union, 'niu')
    X['hc_talkfp'] = np.where(E.v394 == 0, 'novisit', E.v395.map({0: 'no', 1: 'yes'}))
    X.loc[E.v394.isna() | ((E.v394 == 1) & E.v395.isna()), 'hc_talkfp'] = np.nan
    X['religion'] = E.v130_lab.map(religion_category) if 'v130_lab' in E else np.nan
    X['edu_level'] = _na(E.v106, [8, 9]).astype(float)
    X['urban'] = (E.v025 == 1).astype(float)
    X['working'] = _na(E.v714, [8, 9]).astype(float)
    X['mobphone'] = _na(E.v169a, [8, 9]).astype(float)
    X['radio_fq'] = _na(E.v158, [8, 9]).astype(float)
    X['fp_radio'] = _na(E.v384a, [8, 9]).astype(float)
    X['fp_tv'] = _na(E.v384b, [8, 9]).astype(float)
    X['country'] = E.v000.str[:2]
    X['region'] = [region_key(c, r) for c, r in zip(X.country, E.v024_lab)]
    return X


# ---------------------------------------------------------------- formulaire -> variables
def _num(v, lo, hi):
    if v in (None, ''):
        return np.nan
    x = float(v)
    if not lo <= x <= hi:
        raise ValueError(f"valeur {x} hors de l'intervalle [{lo}, {hi}]")
    return x


def from_form(form) -> pd.DataFrame:
    """Transforme les réponses du formulaire (dict de chaînes) en une ligne de variables.

    Lève ValueError si une valeur est invalide. Les champs facultatifs non renseignés deviennent NaN."""
    def cat(name):
        v = form.get(name) or None
        if v is not None and v not in CATEGORIES[name]:
            raise ValueError(f"modalité inconnue pour {name}: {v}")
        return v

    prior_stops = _num(form.get('prior_disc'), 0, CAPS['prior_stops'])
    row = {
        'method': cat('method'),
        'age': _num(form.get('age'), 15, 49),
        'cheb': _num(form.get('cheb'), 0, 20),
        'kids_lt5': _num(form.get('kids_lt5'), 0, 20),
        'months_since_birth': _num(form.get('months_since_birth'), 0, 600),
        'prior_stops': prior_stops,
        'prior_se_stops': (np.nan if np.isnan(prior_stops) else
                           float(prior_stops > 0 and form.get('prior_reason') == 'side_effects')),
        'prior_methods': _num(form.get('prior_methods'), 0, CAPS['prior_methods']),
        'marstat': cat('marstat'), 'fertpref': cat('fertpref'), 'hus_pref': cat('hus_pref'),
        'dec_health': cat('dec_health'), 'hc_talkfp': cat('hc_talkfp'), 'religion': cat('religion'),
        'edu_level': _num(form.get('edu_level'), 0, 3), 'urban': _num(form.get('urban'), 0, 1),
        'working': _num(form.get('working'), 0, 1), 'mobphone': _num(form.get('mobphone'), 0, 1),
        'radio_fq': _num(form.get('radio_fq'), 0, 3), 'fp_radio': _num(form.get('fp_radio'), 0, 1),
        'fp_tv': _num(form.get('fp_tv'), 0, 1),
        'country': cat('country'), 'region': form.get('region') or None,
    }
    if row['method'] is None:
        raise ValueError("la méthode choisie est obligatoire")
    if row['country'] is None or row['region'] is None:
        raise ValueError("pays et région sont obligatoires")
    if not row['region'].startswith(row['country'] + '_'):
        raise ValueError("la région ne correspond pas au pays")
    if row['marstat'] is not None and row['marstat'] not in IN_UNION:
        row['hus_pref'] = row['dec_health'] = 'niu'
    if not np.isnan(row['cheb']):
        if not np.isnan(row['age']) and row['cheb'] > row['age'] - 10:
            raise ValueError("nombre d'enfants incompatible avec l'âge")
        if row['cheb'] == 0:
            row['months_since_birth'] = np.nan
        if not np.isnan(row['kids_lt5']) and row['kids_lt5'] > row['cheb']:
            raise ValueError("plus d'enfants nés depuis 5 ans que d'enfants au total")
    return pd.DataFrame([row])


def to_model_frame(X: pd.DataFrame, regions: list) -> pd.DataFrame:
    """Ordonne les colonnes et fixe les modalités : l'entraînement et la prédiction voient le même encodage."""
    X = X[FEATURES].copy()
    for c, cats in CATEGORIES.items():
        X[c] = pd.Categorical(X[c], categories=cats)
    X['region'] = pd.Categorical(X['region'], categories=regions)
    return X
