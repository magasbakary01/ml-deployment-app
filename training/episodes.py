"""Construction d'une base par épisode d'utilisation contraceptive à partir du calendrier DHS (fichiers IR).

Chaque ligne = un épisode d'utilisation d'une méthode, décrit AU MOMENT DU DÉMARRAGE :
méthode, âge, parité, délai depuis la dernière naissance, arrêts antérieurs (tirés du calendrier),
plus les caractéristiques de la femme mesurées à l'enquête (instruction, milieu, région…).
Issue : arrêt en situation de besoin dans les 12 mois suivant le démarrage.

Usage :  python -m training.episodes data/dhs_ir/*.zip --out data/episodes.parquet

Conventions du calendrier DHS :
  - ``vcal_1`` : une lettre par mois, du plus récent au plus ancien. Les ``v018 - 1`` premières positions
    (mois postérieurs à l'entretien) sont vides : le mois de l'entretien est en position v018, et le calendrier
    couvre ``v019`` mois à partir de là (v017 = CMC du premier mois, v017 + v019 - 1 = v008).
  - ``vcal_2`` : motif d'arrêt, renseigné au dernier mois d'utilisation d'un épisode.
Les codes des lettres changent d'une phase DHS à l'autre (ex. « 4 » = méthode plus efficace en DHS-7,
effets secondaires en DHS-8). Ils sont donc appris pour chaque enquête (``learn_codebook``) en croisant
le dernier arrêt du calendrier avec ses versions étiquetées : v359 (méthode) et v360 (motif).
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NON_USE = {'0', 'B', 'P', 'T', ' '}                  # non-utilisation, naissance, grossesse, fin de grossesse

# Catégories de motif d'arrêt
NOT_IN_NEED = {'veut_grossesse', 'moins_besoin'}       # arrêts sans besoin : exclus de la cible
SWITCH_REASON = 'plus_efficace'                        # changement de méthode : exclu de la cible
REASON_TO_CLASS = {'effets_secondaires': 'effets_secondaires', 'mari': 'mari', 'echec': 'echec',
                   'acces': 'autre', 'autre': 'autre'}

# Libellés DHS (v359 / v360) -> catégories du projet ; premier motif qui correspond
REASON_LABELS = [
    (r'wanted to become pregnant|want.*pregnan', 'veut_grossesse'),
    (r'became pregnant|method fail', 'echec'),
    (r'infrequent|away|menopaus|difficult|dissolution|separat|widow|divorc', 'moins_besoin'),
    (r'husband|partner', 'mari'),
    (r'side effect|health|menstrua|mesntrual|bleeding', 'effets_secondaires'),
    (r'more effective|switch', 'plus_efficace'),
    (r'access|availab|cost|inconvenient|far', 'acces'),
    (r'.', 'autre'),                                   # fataliste, autre, ne sait pas
]
METHOD_LABELS = [
    (r'steril', 'sterilisation'), (r'implant|norplant', 'implant'), (r'iud|coil|dispositif', 'iud'),
    (r'inject', 'injectable'), (r'emergency', 'other'), (r'pill', 'pill'), (r'condom', 'condom'),
    (r'lactational|lam\b', 'lam'), (r'abstinence|withdrawal|standard days|sdm|rhythm|traditional|folk', 'natural'),
    (r'.', 'other'),
]

# Codes DHS-7 standard : utilisés si une enquête ne permet pas d'apprendre ses codes (et dans les tests)
DEFAULT_CODEBOOK = {
    'method': {'1': 'pill', '2': 'iud', '3': 'injectable', '5': 'condom', 'N': 'implant', 'L': 'lam', 'C': 'condom',
               '8': 'natural', '9': 'natural', 'S': 'natural', 'W': 'natural', 'A': 'natural', '6': 'sterilisation',
               '7': 'sterilisation', '4': 'other', 'F': 'other', 'E': 'other', 'M': 'other'},
    'reason': {'1': 'echec', '2': 'veut_grossesse', '3': 'mari', '4': 'plus_efficace', '5': 'effets_secondaires',
               '6': 'acces', '7': 'acces', '8': 'acces', '9': 'autre', 'A': 'moins_besoin', 'C': 'moins_besoin',
               'D': 'moins_besoin', 'F': 'moins_besoin', 'W': 'autre', 'K': 'autre'},
}

IR_COLUMNS = ['caseid', 'v000', 'v001', 'v005', 'v007', 'v008', 'v011', 'v017', 'v018', 'v019', 'v024', 'v025', 'v106',
              'v130', 'v190', 'v501', 'v714', 'v602', 'v621', 'v743a', 'v394', 'v395', 'v384a', 'v384b', 'v158',
              'v169a', 'vcal_1', 'vcal_2', 'v359', 'v360'] + [f'b3_{i:02d}' for i in range(1, 21)]


def _classify(label, rules):
    label = str(label).lower()
    return next(cat for pattern, cat in rules if re.search(pattern, label))


def is_method(c):
    return c not in NON_USE


# ---------------------------------------------------------------- lecture d'un calendrier
def parse_calendar(vcal1: str, vcal2: str, v017: int, v019: int, v018: int = 1):
    """Découpe le calendrier en épisodes d'utilisation, du plus ancien au plus récent.

    Retourne une liste de dict : method_code, start_cmc, end_cmc, duration, ended, reason_code, next_code,
    left_censored. Un épisode encore en cours au mois de l'entretien a ended=False (censuré)."""
    L, off = int(v019), int(v018) - 1              # on saute les mois postérieurs à l'entretien
    s1 = (vcal1 or '')[off:off + L].ljust(L)[::-1]   # ordre chronologique : index 0 = mois le plus ancien
    s2 = (vcal2 or '')[off:off + L].ljust(L)[::-1]
    episodes, i = [], 0
    while i < L:
        c = s1[i]
        if is_method(c):
            j = i
            while j + 1 < L and s1[j + 1] == c:
                j += 1
            ended = j + 1 < L
            episodes.append({
                'method_code': c, 'start_cmc': int(v017) + i, 'end_cmc': int(v017) + j, 'duration': j - i + 1,
                'ended': ended, 'reason_code': s2[j].strip() if ended else '',
                'next_code': s1[j + 1] if ended else '',
                'left_censored': i == 0,               # a pu commencer avant le calendrier
            })
            i = j + 1
        else:
            i += 1
    return episodes


def episode_outcome(ep, codebook=DEFAULT_CODEBOOK, horizon=12):
    """Issue à ``horizon`` mois : ('continue'|'arret_besoin'|'arret_sans_besoin'|'changement'|None, motif)."""
    if ep['ended'] and ep['duration'] <= horizon:
        if ep['reason_code'] in ('', '0'):
            return 'motif_inconnu', None             # exclu de la cible plutôt que compté comme arrêt
        reason = codebook['reason'].get(ep['reason_code'], 'autre')
        if is_method(ep['next_code']) or reason == SWITCH_REASON:
            return 'changement', reason
        if reason in NOT_IN_NEED:
            return 'arret_sans_besoin', reason
        return 'arret_besoin', reason
    if ep['duration'] >= horizon:
        return 'continue', None
    return None, None                               # épisode en cours depuis moins de 12 mois : censuré


# ---------------------------------------------------------------- une femme -> ses épisodes
def woman_episodes(w, codebook=DEFAULT_CODEBOOK, horizon=12):
    eps = parse_calendar(w['vcal_1'], w['vcal_2'], w['v017'], w['v019'], w.get('v018', 1))
    births = sorted(w[f'b3_{i:02d}'] for i in range(1, 21) if pd.notna(w.get(f'b3_{i:02d}')))
    rows, prior_need_stops, prior_se_stops = [], 0, 0
    for k, ep in enumerate(eps):
        start = ep['start_cmc']
        method = codebook['method'].get(ep['method_code'], 'other')
        # suivi complet de 12 mois exigé : sinon les arrêts précoces seraient sur-représentés
        if method != 'sterilisation' and not ep['left_censored'] and w['v008'] - start >= horizon:
            outcome, reason = episode_outcome(ep, codebook, horizon)
            if outcome in ('continue', 'arret_besoin'):
                before = [b for b in births if b < start]
                rows.append({
                    'caseid': w['caseid'], 'start_cmc': start, 'method': method,
                    'age_start': (start - w['v011']) / 12,
                    'parity_start': len(before),
                    'kids_lt5_start': sum(b >= start - 60 for b in before),
                    'months_since_birth': start - before[-1] if before else np.nan,
                    'prior_need_stops': prior_need_stops,
                    'prior_se_stops': prior_se_stops,
                    'prior_episodes': k,
                    'months_before_interview': w['v008'] - start,
                    'y': int(outcome == 'arret_besoin'),
                    'reason': REASON_TO_CLASS.get(reason, 'autre') if outcome == 'arret_besoin' else 'continue',
                })
        # historique mis à jour APRÈS l'épisode, pour les épisodes suivants
        if ep['ended'] and ep['reason_code'] not in ('', '0'):
            r = codebook['reason'].get(ep['reason_code'], 'autre')
            if r not in NOT_IN_NEED and r != SWITCH_REASON and not is_method(ep['next_code']):
                prior_need_stops += 1
                prior_se_stops += r == 'effets_secondaires'
    return rows


# ---------------------------------------------------------------- lecture des fichiers IR et apprentissage des codes
def read_ir(path):
    """Lit un fichier IR DHS (zip Stata) : codes bruts + libellés de v359/v360 (colonnes *_lab)."""
    zf = zipfile.ZipFile(path)
    name = next(n for n in zf.namelist() if n.upper().endswith('.DTA'))
    raw = zf.read(name)
    with pd.read_stata(io.BytesIO(raw), iterator=True, convert_categoricals=False) as it:
        available = it.variable_labels()
        cols = [c for c in IR_COLUMNS if c in available]
        df = it.read(columns=cols)
    lab_cols = [c for c in ('v359', 'v360', 'v024', 'v130') if c in available]
    if lab_cols:
        labs = pd.read_stata(io.BytesIO(raw), columns=lab_cols)
        for c in lab_cols:
            df[c + '_lab'] = labs[c].astype(str).where(labs[c].notna())
    df['survey'] = re.sub(r'DT$', '', Path(path).stem.upper())
    return df


def learn_codebook(df, min_n=5, min_purity=0.9):
    """Apprend la signification des lettres du calendrier pour une enquête.

    Pour chaque femme, le dernier arrêt du calendrier correspond à v359 (méthode) et v360 (motif) :
    chaque lettre prend le libellé qui lui est associé dans au moins ``min_purity`` des cas.
    Retourne (codebook, rapport)."""
    if 'v360_lab' not in df:
        return DEFAULT_CODEBOOK, {'source': 'défaut DHS-7 (v359/v360 absents)'}
    last_m, last_r = [], []
    v018 = df.v018 if 'v018' in df else [1] * len(df)
    for v1, v2, a, b, c in zip(df.vcal_1, df.vcal_2, df.v017, df.v019, v018):
        eps = [e for e in parse_calendar(v1, v2, a, b, c) if e['ended'] and e['reason_code'] not in ('', '0')]
        last_m.append(eps[-1]['method_code'] if eps else None)
        last_r.append(eps[-1]['reason_code'] if eps else None)
    book, report = {'method': {}, 'reason': {}}, {'source': 'appris (v359/v360)', 'ambigus': []}
    for key, codes, lab, rules in [('method', last_m, 'v359_lab', METHOD_LABELS),
                                   ('reason', last_r, 'v360_lab', REASON_LABELS)]:
        t = pd.crosstab(pd.Series(codes, index=df.index), df[lab])
        for code, row in t.iterrows():
            if row.sum() < min_n:
                continue
            if row.max() / row.sum() < min_purity:
                report['ambigus'].append((key, code, row.idxmax(), round(row.max() / row.sum(), 2)))
                continue
            book[key][code] = _classify(row.idxmax(), rules)
    # lettres de méthode jamais arrêtées en dernier (ex. stérilisation) : repli sur le code standard
    for code, cat in DEFAULT_CODEBOOK['method'].items():
        book['method'].setdefault(code, cat)
    report['codes'] = book
    return book, report


def check_codes(df):
    """Contrôle qualité d'une enquête : calendrier présent, cohérence des dates."""
    if 'vcal_1' not in df:
        return {'n_femmes': len(df), 'calendrier_rempli': 0.0}
    return {
        'n_femmes': len(df),
        'calendrier_rempli': round(float((df.vcal_1.fillna('').str.strip() != '').mean()), 3),
        'dates_coherentes': round(float(((df.v017 + df.v019 - 1) == df.v008).mean()), 3),
        'position_entretien_ok': round(float(((df.vcal_1.fillna('').str.len() - df.vcal_1.fillna('').str.lstrip().str.len())
                                              == df.v018 - 1).mean()), 3),
    }


def build(paths, horizon=12):
    frames, report = [], {}
    keep = ['caseid', 'survey', 'v000', 'v001', 'v005', 'v007', 'v024', 'v025', 'v106', 'v130', 'v190', 'v501',
            'v714', 'v602', 'v621', 'v743a', 'v394', 'v395', 'v384a', 'v384b', 'v158', 'v169a', 'v024_lab', 'v130_lab']
    for p in paths:
        df = read_ir(p)
        s = df.survey.iloc[0]
        report[s] = check_codes(df)
        if report[s]['calendrier_rempli'] < 0.5:
            report[s]['statut'] = 'ignorée : pas de calendrier'
            continue
        df = df[df.vcal_1.fillna('').str.strip() != '']
        book, rep = learn_codebook(df)
        report[s].update(rep)
        rows = [r for w in df.to_dict('records') for r in woman_episodes(w, book, horizon)]
        ep = pd.DataFrame(rows)
        report[s].update({'episodes': len(ep), 'taux_arret_besoin': round(float(ep.y.mean()), 3)})
        frames.append(ep.merge(df[[c for c in keep if c in df]], on='caseid', how='left'))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--out', default='data/episodes.csv')
    a = ap.parse_args()
    ep, report = build(a.files)
    for s, r in report.items():
        codes = r.pop('codes', None)
        print(s, r)
        if codes:
            print('   motifs :', codes['reason'])
    if len(ep):
        ep.to_csv(a.out, index=False)
        print(f"\n{len(ep)} épisodes -> {a.out} ; taux d'arrêt en situation de besoin à 12 mois : {ep.y.mean():.3f}")
        print(ep.groupby('method').y.agg(['size', 'mean']).round(3).sort_values('size', ascending=False))


if __name__ == '__main__':
    main()
