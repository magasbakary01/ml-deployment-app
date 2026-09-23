# Risque d'arrêt de la contraception : outil d'aide au counseling

Application web qui estime, lors d'une consultation de planification familiale, le risque qu'une femme arrête sa
contraception alors qu'elle en a encore besoin. Elle indique aussi le motif d'arrêt le plus probable et une conduite
de counseling adaptée.

La feuille de route du projet (carnet électronique OpenSRP 2, pilote, passage à l'échelle) est dans
[`docs/FEUILLE_DE_ROUTE.md`](docs/FEUILLE_DE_ROUTE.md).

## Structure

```
contraception_risk/     variables du modèle (épisodes DHS -> variables, formulaire -> variables), prédiction, conseil
training/episodes.py    lit le calendrier DHS (fichiers IR) et construit la base par épisode d'utilisation
training/train.py       validation et entraînement ; écrit les fichiers de models/
models/                 model.joblib, metrics.json, regions.csv (noms des régions, libellés DHS)
app/                    application Flask (formulaire, résultat, page « À propos du modèle »)
tests/                  tests automatiques
data/dhs_ir/            fichiers femmes DHS (XXIRnnDT.zip) — non versionnés (conditions d'utilisation DHS)
data/episodes.csv       base par épisode générée — non versionnée
```

L'ancienne application (`Downloads/My_computer/.../MLdepoyment/`) et l'extrait IPUMS (`data/idhs_00009.csv`) ne
sont plus utilisés par le modèle.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Utilisation

```bash
python -m training.episodes data/dhs_ir/*.zip --out data/episodes.csv   # 1. base par épisode (~5 min)
python -m training.train                                                 # 2. validation + modèles (~5 min)
python -m pytest                                                         # tests
python -m app.app                                                        # http://127.0.0.1:5000
```

En production, utiliser un serveur WSGI et définir une clé secrète :

```bash
set SECRET_KEY=une-longue-cle-aleatoire
waitress-serve --port=8000 --call app.app:create_app
```

## Le modèle

- **Unité d'analyse :** un épisode d'utilisation d'une méthode, tiré du calendrier contraceptif mois par mois des
  DHS, et suivi 12 mois complets.
- **Cible :** arrêt de la méthode dans les 12 mois alors que la femme a encore besoin de contraception (effets
  secondaires, opposition du partenaire, grossesse sous méthode, accès, coût…). Les arrêts pour désir de
  grossesse, baisse du besoin ou changement de méthode sont exclus.
- **Données :** 7 enquêtes, soit 28 985 épisodes. Taux d'arrêt : 23 % (implant 12 %, injectable 39 %, pilule 36 %).
  - Guinée 2018 (GNIR71)
  - Burkina Faso 2021, Côte d'Ivoire 2021
  - Ghana 2022, Nigeria 2023-24, Sénégal 2023, Mali 2023-24
  - GNIR82 et NIIR82 ne contiennent pas de calendrier : elles sont ignorées.
- **Exclusion :** Guinée 2018, région de Kankan. Le calendrier y est anormal : 3 épisodes par utilisatrice contre
  1,2 à 1,5 ailleurs, et 86 % d'arrêts. La région reste proposée dans l'application ; l'estimation s'appuie
  alors sur le reste du pays et un avertissement s'affiche.
- **Codes du calendrier :** ils changent d'une phase DHS à l'autre. `training/episodes.py` les apprend donc pour
  chaque enquête à partir des variables étiquetées v359 (méthode) et v360 (motif).
- **Variables :**
  - au démarrage de la méthode : méthode, âge, parité, naissances des 5 dernières années, délai depuis la
    dernière naissance, arrêts et méthodes antérieurs ;
  - caractéristiques de la femme : couple, désir d'enfants, décision, information PF, instruction, milieu, activité,
    téléphone, radio, religion ;
  - contexte : pays et région.
- **Validation :** 5 plis ; les épisodes d'un même village (grappe DHS) restent dans le même pli.

| | Valeur |
|---|---|
| AUC | 0,752 (0,71 à 0,77 selon le pays ; Guinée : 0,769) |
| AUC sur un pays exclu de l'entraînement | 0,64 à 0,73 |
| Risque faible / moyen / élevé | 66 % / 25 % / 9 % des épisodes ; taux d'arrêt observé 13 % / 36 % / 58 % |

## Conseil selon la méthode (`contraception_risk/counseling.py`)

Le formulaire produit aussi une fiche de conseil propre à la méthode choisie et des alertes :
- antécédent d'arrêt ;
- reprise d'une méthode arrêtée pour effets secondaires ;
- motif « échec » prédit alors que la méthode dépend de l'utilisatrice → proposer une méthode de longue durée ;
- opposition probable du partenaire ;
- relais à prévoir après la MAMA.

Le type de visite, les effets secondaires actuels et la méthode arrêtée servent uniquement au conseil. La méthode
choisie et le nombre d'arrêts antérieurs entrent dans le score.

**Les textes de conseil doivent être validés par un clinicien avant usage.**

## Limites et suite

- Seules la méthode, l'âge, la parité, les naissances et l'historique sont mesurés au démarrage de l'épisode. Les
  autres caractéristiques sont mesurées à l'enquête, jusqu'à 5 ans plus tard.
- La Guinée ne compte que 994 épisodes exploitables : il faut une validation prospective dans les structures
  guinéennes.
- Ajouter d'autres enquêtes avec calendrier (priorité 3 de la liste DHS) renforcera le modèle.
