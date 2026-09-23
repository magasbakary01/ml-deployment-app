"""Chargement des artefacts et prédiction pour une patiente."""
import json
from pathlib import Path

import joblib
import pandas as pd

from . import features as F

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
FEW_EPISODES = 50          # en dessous, la région est signalée comme peu représentée

TIER_LABELS = {'faible': 'Risque faible', 'moyen': 'Risque moyen', 'élevé': 'Risque élevé'}
REASON_LABELS = {
    'effets_secondaires': "Effets secondaires / inquiétudes de santé",
    'mari': "Opposition du mari / partenaire",
    'echec': "Grossesse sous méthode (échec)",
    'autre': "Accès, coût, inconfort ou autre raison",
}
COUNSEL = {
    'effets_secondaires': "Expliquer à l'avance les effets secondaires attendus et quoi faire s'ils surviennent ; "
                          "proposer une visite de suivi précoce et la possibilité de changer de méthode plutôt que d'arrêter.",
    'mari': "Proposer un counseling en couple ou des informations à partager avec le partenaire ; "
            "discuter de méthodes discrètes si la patiente le souhaite.",
    'echec': "Vérifier la bonne utilisation de la méthode ; proposer une méthode de longue durée (implant, DIU) "
             "moins dépendante de l'utilisatrice.",
    'autre': "Anticiper l'accès : prochain rendez-vous, lieu de réapprovisionnement, coût ; "
             "rappel par téléphone si possible.",
}
TIER_ACTIONS = {
    'faible': "Counseling standard et rendez-vous de routine.",
    'moyen': "Counseling renforcé ciblé sur le motif probable ; confirmer la date du prochain rendez-vous.",
    'élevé': "Counseling renforcé, suivi rapproché (appel ou visite à 1 mois) et plan en cas d'effets secondaires.",
}


class RiskModel:
    def __init__(self, models_dir=MODELS_DIR):
        models_dir = Path(models_dir)
        b = joblib.load(models_dir / 'model.joblib')
        self.model, self.reason_model = b['model'], b['reason_model']
        self.reason_classes, self.regions, self.tiers = b['reason_classes'], b['regions'], b['tiers']
        self.metrics = json.loads((models_dir / 'metrics.json').read_text(encoding='utf-8'))
        self.region_info = pd.read_csv(models_dir / 'regions.csv', encoding='utf-8').set_index('region')
        self.region_names = self.region_info['nom'].to_dict()

    def regions_by_country(self):
        out = {}
        for r, row in self.region_info.sort_values('nom').iterrows():
            out.setdefault(row.pays, []).append((r, row.nom))
        return out

    def tier(self, p):
        return 'faible' if p < self.tiers[0] else 'moyen' if p < self.tiers[1] else 'élevé'

    def predict(self, form):
        """Retourne la probabilité d'arrêt, le niveau de risque, le motif probable et les facteurs explicatifs.

        Lève ValueError si le formulaire est invalide."""
        X = F.from_form(form)
        if X.region[0] not in self.region_info.index:
            raise ValueError("région inconnue")
        X = F.to_model_frame(X, self.regions)

        p = float(self.model.predict_proba(X)[0, 1])
        pr = self.reason_model.predict_proba(X)[0]
        stop = {c: float(v) for c, v in zip(self.reason_classes, pr) if c != 'continue'}
        total = sum(stop.values())
        reasons = sorted(((REASON_LABELS[c], v / total, c) for c, v in stop.items()), key=lambda t: -t[1])

        # Contributions (valeurs SHAP de LightGBM) ; pays et région regroupés
        contrib = self.model.predict(X, pred_contrib=True)[0][:-1]
        s = pd.Series(contrib, index=X.columns)
        grouped = s.drop(F.CONTEXT_FEATURES).rename(F.LABELS)
        grouped[F.LABELS['zone']] = s[F.CONTEXT_FEATURES].sum()
        grouped = grouped[grouped.abs() > 0.02].sort_values()
        tier = self.tier(p)
        n_region = int(self.region_info.loc[X.region[0], 'episodes'])
        return {
            'proba': p, 'tier': tier, 'tier_label': TIER_LABELS[tier], 'action': TIER_ACTIONS[tier],
            'reasons': reasons, 'counsel': COUNSEL[reasons[0][2]],
            'up': [(k, v) for k, v in grouped[::-1].items() if v > 0][:3],
            'down': [(k, v) for k, v in grouped.items() if v < 0][:3],
            'observed_rate': self.metrics.get('niveaux_de_risque', {}).get(tier, {}).get('taux_arret_observe'),
            'region_note': None if n_region >= FEW_EPISODES else (
                "Aucune donnée d'entraînement fiable pour cette région : l'estimation s'appuie sur le reste du pays."
                if n_region == 0 else
                f"Région peu représentée dans les données ({n_region} épisodes) : estimation moins précise."),
        }
