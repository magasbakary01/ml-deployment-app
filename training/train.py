"""Entraîne les modèles de risque d'arrêt de la contraception et écrit les artefacts dans ``models/``.

Usage :
    python -m training.episodes data/dhs_ir/*.zip --out data/episodes.csv   # 1. base par épisode
    python -m training.train [--episodes data/episodes.csv] [--out models]    # 2. modèles

Unité d'analyse : un épisode d'utilisation d'une méthode (calendrier DHS), suivi 12 mois complets.
Cible : arrêt de la méthode en situation de besoin dans les 12 mois (effets secondaires, opposition du
partenaire, grossesse sous méthode, accès, coût…). Les arrêts pour désir de grossesse, baisse du besoin ou
changement de méthode sont exclus ; la référence est la poursuite de la méthode 12 mois.
"""
import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from contraception_risk import features as F

ROOT = Path(__file__).resolve().parents[1]

REASON_CLASSES = ['continue', 'effets_secondaires', 'mari', 'echec', 'autre']
TIERS = [0.25, 0.50]                   # seuils faible / moyen / élevé

# Données exclues pour qualité insuffisante (motif documenté)
EXCLUSIONS = [
    {'survey': 'GNIR71', 'v024_lab': 'kankan',
     'motif': "Guinée 2018, région de Kankan : calendrier anormal (3,0 épisodes par utilisatrice contre 1,2 à 1,5 "
              "ailleurs ; 861 épisodes de 3 mois ou moins ; 86 % d'arrêts)"},
]

PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=63, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, reg_lambda=5.0, n_estimators=300, verbose=-1,
              max_cat_to_onehot=4, cat_smooth=20, random_state=0)


def load(path):
    """Retourne (épisodes conservés, tous les épisodes) après application des exclusions."""
    E = pd.read_csv(path, low_memory=False)
    E = E[E.v000.str[:2].isin(F.COUNTRY_NAMES)].reset_index(drop=True)
    excluded = pd.Series(False, index=E.index)
    for ex in EXCLUSIONS:
        mask = pd.Series(True, index=E.index)
        for k, v in ex.items():
            if k != 'motif':
                mask &= E[k] == v
        excluded |= mask
    return E[~excluded].reset_index(drop=True), E


def region_table(E_all, E_kept):
    """Régions proposées dans l'application, avec leur nom (libellés DHS de v024) et le nombre d'épisodes
    d'entraînement. Une région exclue garde sa place (0 épisode) : le modèle s'appuie alors sur le reste du pays."""
    def keys(E):
        return pd.Series([F.region_key(c, l) for c, l in zip(E.v000.str[:2], E.v024_lab)], index=E.index)
    r = E_all.assign(region=keys(E_all), pays=E_all.v000.str[:2]).sort_values('v007')
    r = r.groupby('region').agg(pays=('pays', 'last'), nom=('v024_lab', 'last'))
    r['nom'] = r.nom.str.title()
    r['episodes'] = keys(E_kept).value_counts().reindex(r.index).fillna(0).astype(int)
    return r


def tier(p):
    return np.select([p < TIERS[0], p < TIERS[1]], ['faible', 'moyen'], 'élevé')


def evaluate(X, y, reason, groups):
    """Validation croisée 5 plis ; les épisodes d'un même village (grappe DHS) restent dans le même pli."""
    oof = np.zeros(len(y))
    rc = pd.Categorical(reason, categories=REASON_CLASSES).codes
    oof_r = np.zeros((len(y), len(REASON_CLASSES)))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups):
        oof[te] = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
        m = lgb.LGBMClassifier(**{**PARAMS, 'objective': 'multiclass'}).fit(X.iloc[tr], rc[tr])
        oof_r[te] = m.predict_proba(X.iloc[te])
    country = X.country.astype(str).values
    cols = [f for f in F.FEATURES if f not in ('country', 'region')]
    loco = {}
    for c in np.unique(country):
        tr, te = country != c, country == c
        p = lgb.LGBMClassifier(**PARAMS).fit(X.loc[tr, cols], y[tr]).predict_proba(X.loc[te, cols])[:, 1]
        loco[c] = round(roc_auc_score(y[te], p), 3)
    t = tier(oof)
    method = X.method.astype(str).values
    return {
        'n': int(len(y)), 'prevalence': round(float(y.mean()), 3),
        'auc': round(roc_auc_score(y, oof), 3), 'pr_auc': round(average_precision_score(y, oof), 3),
        'brier': round(brier_score_loss(y, oof), 4),
        'niveaux_de_risque': {k: {'part': round(float(np.mean(t == k)), 3),
                                  'taux_arret_observe': round(float(y[t == k].mean()), 3)} for k in ['faible', 'moyen', 'élevé']},
        'taux_par_decile': pd.Series(y.values).groupby(pd.qcut(oof, 10, labels=False)).mean().round(3).tolist(),
        'par_pays': {c: {'n': int((country == c).sum()), 'prevalence': round(float(y[country == c].mean()), 3),
                         'auc': round(roc_auc_score(y[country == c], oof[country == c]), 3)} for c in np.unique(country)},
        'auc_pays_non_vu': loco,
        'par_methode': {m: {'n': int((method == m).sum()), 'observe': round(float(y[method == m].mean()), 3),
                            'predit': round(float(oof[method == m].mean()), 3)} for m in np.unique(method)},
        'motif_auc': {c: round(roc_auc_score(rc == i, oof_r[:, i]), 3) for i, c in enumerate(REASON_CLASSES)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', default=ROOT / 'data' / 'episodes.csv')
    ap.add_argument('--out', default=ROOT / 'models')
    ap.add_argument('--skip-eval', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    E, E_all = load(a.episodes)
    regions = region_table(E_all, E)
    X = F.to_model_frame(F.from_episodes(E), list(regions.index))
    y, reason = E.y, E.reason
    groups = E.survey + '_' + E.v001.astype(str)
    print(f"{len(y)} épisodes, {E.survey.nunique()} enquêtes ; taux d'arrêt en situation de besoin : {y.mean():.3f}")

    metrics = {} if a.skip_eval else evaluate(X, y, reason, groups)
    if metrics:
        print(json.dumps({k: metrics[k] for k in ['auc', 'pr_auc', 'brier', 'niveaux_de_risque', 'par_pays']},
                         ensure_ascii=False, indent=1))

    model = lgb.LGBMClassifier(**PARAMS).fit(X, y)
    rc = pd.Categorical(reason, categories=REASON_CLASSES).codes
    reason_model = lgb.LGBMClassifier(**{**PARAMS, 'objective': 'multiclass'}).fit(X, rc)

    joblib.dump({'model': model, 'reason_model': reason_model, 'reason_classes': REASON_CLASSES,
                 'features': F.FEATURES, 'regions': list(regions.index), 'tiers': TIERS,
                 'trained_on': str(date.today()), 'lightgbm': lgb.__version__}, out / 'model.joblib')
    regions.to_csv(out / 'regions.csv', encoding='utf-8')
    metrics.update({'date': str(date.today()), 'enquetes': sorted(E.survey.unique()),
                    'cible': "arrêt de la méthode en situation de besoin dans les 12 mois suivant le démarrage",
                    'exclusions': [ex['motif'] for ex in EXCLUSIONS], 'variables': F.FEATURES, 'parametres': PARAMS})
    (out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"Artefacts écrits dans {out}")


if __name__ == '__main__':
    main()
