# Fichiers femmes DHS (Individual Recode, format Stata)

Ces fichiers servent à construire la base par épisode (`training/episodes.py`). Les zip sont à placer dans
`data/dhs_ir/`, sans les décompresser ; ce dossier est exclu de git, conformément aux conditions d'utilisation DHS.

**Téléchargement :** les liens ne fonctionnent que dans un navigateur **connecté au compte DHS** du projet
(https://dhsprogram.com/data/dataset_admin/login_main.cfm). Sans session, le site renvoie une page de connexion.

**Construction d'un lien :**

```
https://dhsprogram.com/customcf/legacy/data/download_dataset.cfm?Filename={FICHIER}DT.zip&Tp=1&Ctry_Code={PAYS}&surv_id={ID}&dm=1&dmode=nm
```

Exemple (Guinée 2018) :
https://dhsprogram.com/customcf/legacy/data/download_dataset.cfm?Filename=GNIR71DT.zip&Tp=1&Ctry_Code=GN&surv_id=539&dm=1&dmode=nm

**Pour les obtenir à nouveau :** sur dhsprogram.com, choisir « Individual Recode », « Stata System file »,
« All DHS », puis cocher les pays et cliquer sur « Build URL File List ». Ne pas cocher « Individual Raw » : ces
fichiers (IQ) n'ont pas de calendrier.

## Statut des fichiers

- ✅ **Utilisé** : intégré au modèle (au 23/09/2026).
- ⛔ **Sans calendrier** : fichier allégé, ignoré par le modèle.
- 🔜 **Priorité** : à télécharger ensuite pour renforcer le modèle.
- Sans mention : enquêtes anciennes, dont le calendrier est souvent absent ou incomplet.

| Pays (code) | Fichier : surv_id |
|---|---|
| Bénin (BJ) | BJIR31 : 87 · BJIR41 : 211 · BJIR51 : 289 · BJIR61 : 420 · BJIR71 : 491 🔜 |
| Burkina Faso (BF) | BFIR21 : 42 · BFIR31 : 116 · BFIR43 : 230 · BFIR62 : 329 🔜 · BFIR71 : 481 · BFIR7A : 526 · **BFIR81 : 562 ✅** |
| Côte d'Ivoire (CI) | CIIR35 : 62 · CIIR3A : 106 · CIIR51 : 231 · CIIR62 : 311 🔜 · **CIIR81 : 559 ✅** |
| Gambie (GM) | GMIR61 : 425 · GMIR81 : 555 🔜 |
| Ghana (GH) | GHIR02 : 25 · GHIR31 : 58 · GHIR41 : 151 · GHIR4B : 235 · GHIR5A : 301 · GHIR72 : 437 🔜 · GHIR7B : 516 · GHIR82 : 557 · **GHIR8C : 598 ✅** |
| Guinée (GN) | GNIR41 : 154 · GNIR52 : 249 · GNIR62 : 391 🔜 · **GNIR71 : 539 ✅** (Kankan exclue) · GNIR82 : 571 ⛔ |
| Liberia (LB) | LBIR01 : 7 · LBIR51 : 271 · LBIR5A : 330 · LBIR61 : 361 · LBIR6A : 435 · LBIR71 : 509 · LBIR7A : 537 · LBIR81 : 573 🔜 |
| Mali (ML) | MLIR01 : 12 · MLIR32 : 77 · MLIR41 : 159 · MLIR53 : 276 · MLIR6A : 405 🔜 · MLIR72 : 487 🔜 · MLIR7A : 517 · MLIR83 : 574 · **MLIR8A : 613 ✅** |
| Mauritanie (MR) | MRIR71 : 553 🔜 |
| Niger (NI) | NIIR22 : 53 · NIIR31 : 99 · NIIR51 : 277 · NIIR61 : 407 🔜 · NIIR82 : 575 ⛔ |
| Nigeria (NG) | NGIR21 : 32 · NGIR4B : 223 · NGIR53 : 302 · NGIR61 : 392 · NGIR6A : 438 🔜 · NGIR71 : 474 · NGIR7B : 528 🔜 · NGIR81 : 576 · **NGIR8B : 609 ✅** |
| Sénégal (SN) | SNIR02 : 3 · SNIR21 : 46 · SNIR32 : 81 · SNIR4A : 233 · SNIR51 : 293 · SNIR5A : 338 · SNIR61 : 365 · SNIR6D : 423 · SNIR71 : 457 🔜 · SNIR7A : 489 🔜 · SNIR7I : 524 🔜 · SNIR7Z : 534 🔜 · SNIR81 : 580 · SNIR8B : 581 · SNIR8I : 587 · **SNIR8S : 611 ✅** |
| Sierra Leone (SL) | SLIR51 : 324 · SLIR61 : 450 · SLIR73 : 515 · SLIR7A : 545 🔜 |
| Togo (TG) | TGIR01 : 22 · TGIR31 : 114 · TGIR61 : 328 · TGIR71 : 497 🔜 |

## Points d'attention

- Le programme `training/episodes.py` apprend automatiquement la signification des codes du calendrier de chaque
  enquête : les codes changent d'une phase DHS à l'autre.
- Il signale les enquêtes sans calendrier. Il faut aussi vérifier la qualité du calendrier par région : Kankan
  2018 a été exclue parce que son calendrier était anormal.
- Après tout ajout de fichiers, relancer :
  `python -m training.episodes data/dhs_ir/*.zip --out data/episodes.csv` puis `python -m training.train`.
