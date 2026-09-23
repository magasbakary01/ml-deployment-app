# Feuille de route : carnet électronique de la femme avec prédiction du risque d'arrêt de la contraception

*Version du 23 septembre 2026. Document de travail, à mettre à jour à chaque étape.*

## 1. Objectif

Doter les structures de santé guinéennes d'un **carnet électronique de la femme** fondé sur **OpenSRP 2**. Le carnet :

- fonctionne hors ligne ;
- est conforme aux SMART Guidelines de l'OMS ;
- est interopérable avec DHIS2.

Il intègre un **modèle d'aide à la décision** qui estime, pendant la consultation de planification familiale (PF),
trois choses : le risque qu'une femme arrête sa méthode dans les 12 mois alors qu'elle en a encore besoin, le
motif probable de cet arrêt et le conseil adapté.

## 2. Point de départ (acquis au 23/09/2026)

| Acquis | Détail |
|---|---|
| Modèle de prédiction | LightGBM entraîné sur 28 985 épisodes d'utilisation, tirés du calendrier DHS de 7 pays (Guinée 2018, Burkina Faso, Côte d'Ivoire, Ghana, Mali, Nigeria, Sénégal 2021-2024) |
| Performance validée | AUC 0,75 (Guinée 0,77) ; 0,64 à 0,73 sur un pays exclu de l'entraînement ; taux d'arrêt observés de 13 % / 36 % / 58 % selon le niveau de risque (faible / moyen / élevé) |
| Motif probable d'arrêt | Second modèle : effets secondaires, partenaire, échec de la méthode, accès/autre |
| Application de démonstration | Application web Flask : formulaire, résultat, conseil selon la méthode, page « À propos du modèle » ; 34 tests automatiques |
| Choix de plateforme | OpenSRP 2 (`opensrp/fhircore`, licence Apache 2.0), fonctionnement « local d'abord », DHIS2 pour les indicateurs agrégés |
| Publication | Magassouba et al., *Reproductive Health* (2026) 23:83 |

## 3. Architecture cible

```
Téléphone (OpenSRP 2, hors ligne)            Serveur local de la structure
┌──────────────────────────────┐   synchro   ┌──────────────────────────────────────┐
│ Questionnaire PF (FHIR)      │ ──────────► │ HAPI FHIR  ◄──►  Keycloak (comptes)  │
│ Modèle embarqué (ONNX)       │ ◄────────── │     │ notification                   │
│ → risque, motif, conseil     │             │     ▼                                │
│ → tâche de suivi à 1 mois    │             │ Service du modèle (API)              │
└──────────────────────────────┘             │ → RiskAssessment + Task (FHIR)       │
                                             │ Passerelle SMS (téléphone GSM)       │
                                             └──────────────┬───────────────────────┘
                                                            │ par moments
                                                            ▼
                                               Serveur central → DHIS2 (agrégé)
```

Principes :

- **Local d'abord.** Aucune coupure de réseau ne bloque la consultation.
- **Standards ouverts** : FHIR, SMART Guidelines OMS, CDS Hooks.
- **Logiciel libre** : aucune licence payante.
- **DHIS2 prévu dès la conception**, mais branché plus tard.

## 4. Les étapes

Les durées sont indicatives et courent à partir du démarrage de chaque étape.

### Étape 0 : cadre institutionnel et éthique (en continu, dès maintenant)

**Objectif :** obtenir l'appui du ministère et les autorisations du pilote.

- [ ] Présenter le projet au ministère : direction de la santé de la reproduction et système d'information sanitaire (DHIS2)
- [ ] Obtenir une lettre d'appui du ministère
- [ ] Identifier 2 ou 3 structures pilotes (connectivité, électricité, volume de consultations PF)
- [ ] Faire relire les textes de conseil par des cliniciens (sages-femmes, gynécologues)
- [ ] Rédiger le protocole de pilote et le soumettre au comité d'éthique
- [ ] Vérifier la conformité à la loi guinéenne sur les données personnelles
- [ ] Rechercher un partenaire technique OpenSRP (par exemple Ona) et un développeur Android/Kotlin

**Responsable :** porteur du projet. **Jalon :** lettre d'appui et avis éthique favorable.

### Étape 1 : le modèle en service FHIR et en version embarquable (quelques sessions de travail)

**Objectif :** rendre le modèle utilisable par OpenSRP 2 ou par toute plateforme FHIR.

- [ ] Questionnaire FHIR de consultation PF, en français, aligné sur le kit OMS de planification familiale
- [ ] Traduction réponses du questionnaire → variables du modèle, avec des tests de concordance avec l'encodage d'entraînement
- [ ] API du modèle :
  - reçoit une consultation FHIR ;
  - renvoie un `RiskAssessment` (probabilité, niveau, motif, conseil) ;
  - crée une `Task` de suivi si le risque est élevé ;
  - expose un point d'accès CDS Hooks.
- [ ] Export ONNX des deux modèles, avec un test de concordance (écart < 0,0001 sur des milliers de cas)
- [ ] Image Docker du service

**Responsable :** Claude. **Prérequis :** aucun. **Jalon :** API et modèles ONNX testés.

### Étape 2 : environnement de démonstration OpenSRP 2 (1 à 3 semaines)

**Objectif :** une démonstration complète sur un ordinateur et un téléphone.

- [ ] Docker Compose avec HAPI FHIR, Keycloak et le service du modèle
- [ ] Configuration OpenSRP 2 :
  - écrans ;
  - questionnaire PF ;
  - extraction des données vers les ressources FHIR ;
  - règles de création des tâches.
- [ ] Notification automatique du serveur vers le modèle à chaque consultation synchronisée
- [ ] Démonstration de bout en bout : consultation → synchronisation → risque affiché → tâche de suivi

**Responsables :** Claude, avec un informaticien. **Prérequis :** ordinateur avec Docker Desktop, Android Studio,
un téléphone Android de test. **Jalon :** démonstration présentable au ministère et aux financeurs.

### Étape 3 : modèle embarqué sur le téléphone (plusieurs semaines)

**Objectif :** afficher le risque pendant la consultation, même sans réseau.

- [ ] Code Kotlin qui charge le modèle ONNX dans une version adaptée d'OpenSRP 2
- [ ] Calcul et affichage du résultat à la validation du questionnaire
- [ ] Tests sur plusieurs téléphones (performance, mémoire, batterie)
- [ ] Procédure de mise à jour du modèle sur les appareils

**Responsables :** Claude pour le code, **développeur Android/Kotlin** pour la compilation, les tests et la
maintenance. **Jalon :** application pilote installable (APK).

### Étape 4 : rappels SMS (1 à 2 semaines)

**Objectif :** rappeler les rendez-vous et assurer le suivi à 1 mois des femmes à risque élevé.

- [ ] Service qui lit chaque jour les tâches de rappel sur le serveur local
- [ ] Envoi par un téléphone Android « passerelle » (réseau GSM, sans internet)
- [ ] Messages neutres, sans mention de la contraception ; consentement et numéro recueillis en consultation
- [ ] Étude d'une alternative vocale en langues locales (poular, maninka, soussou) ou d'un appel par l'agent communautaire

**Responsables :** Claude, avec l'équipe du projet. **Prérequis :** téléphone passerelle, carte SIM avec forfait
SMS, textes validés. **Jalon :** rappels testés avec des volontaires.

### Étape 5 : DHIS2, sécurité et préparation du pilote (2 à 4 semaines)

**Objectif :** un système prêt pour un usage réel.

- [ ] Script mensuel d'indicateurs agrégés vers DHIS2 (API ou fichier d'import), sur des indicateurs validés par le ministère
- [ ] Sécurité :
  - chiffrement des appareils et du serveur ;
  - comptes par rôle ;
  - sauvegardes automatiques et copie externe.
- [ ] Serveur local et énergie de secours (batterie ou solaire) pour chaque structure
- [ ] Guides d'installation, d'utilisation et de dépannage
- [ ] Formation des soignants

**Responsables :** Claude pour la partie technique, équipe du projet pour le matériel et la formation.
**Jalon :** structures pilotes équipées et formées.

### Étape 6 : pilote et évaluation d'impact (18 à 24 mois)

**Objectif :** démontrer que l'outil réduit les arrêts de contraception.

- [ ] Déploiement dans 2 ou 3 structures, idéalement comparées à des structures sans l'outil
- [ ] Suivi des femmes à 12 mois
- [ ] Indicateurs :
  - taux de poursuite de la méthode à 12 mois ;
  - satisfaction des femmes ;
  - temps de consultation ;
  - utilisation de l'outil par les soignants.
- [ ] Validation du modèle sur les données guinéennes du pilote, puis ré-entraînement
- [ ] Rapport d'évaluation et publication

**Responsables :** équipe du projet et université. **Jalon :** preuve d'impact, pour le passage à l'échelle.

### Étape 7 : passage à l'échelle (après le pilote)

- [ ] Extension nationale, sur financement public ou partenaires
- [ ] Nouveaux modules du carnet :
  - suivi prénatal ;
  - vaccination de l'enfant ;
  - dépistage du cancer du col.
- [ ] Extension régionale aux pays couverts par le modèle

## 5. Amélioration continue du modèle

- [ ] Ajouter les autres enquêtes DHS avec calendrier (Sénégal 2014-2017, Nigeria 2013 et 2018, Ghana 2014, Mali 2018…)
- [ ] Comparer les taux guinéens au rapport DHS Guinée 2018 (arrêts à 12 mois par méthode)
- [ ] Intégrer les variables issues du pilote : effets secondaires vécus, qualité du counseling, accord du partenaire, distance
- [ ] Surveiller la performance dans le temps et ré-entraîner régulièrement

## 6. Principaux risques

| Risque | Parade |
|---|---|
| Pas d'accord du ministère | Impliquer le ministère dès l'étape 0 ; interopérabilité DHIS2 ; conformité OMS |
| Manque de compétences Android/FHIR | Partenaire technique ; recrutement ou formation d'un développeur Kotlin |
| Surcharge de travail des soignants | Co-conception avec des sages-femmes ; questions calculées automatiquement à partir du carnet |
| Coupures de réseau et d'électricité | Fonctionnement local d'abord ; énergie de secours |
| Confidentialité des données de contraception | Chiffrement, accès par rôle, SMS neutres, consentement |
| Pression sur le choix de méthode | Messages d'information et d'accompagnement, jamais de recommandation contraignante ; formation des soignants |
| Financement après le pilote | Modèle de services, intégration aux programmes nationaux, diversification des bailleurs |

## 7. Prochaines actions

1. **Claude :** lancer l'étape 1 (modèle en service FHIR et ONNX).
2. **Porteur du projet :** premier contact avec le ministère ; identification des structures pilotes ; relecture clinique.
3. **Porteur du projet :** installer Docker Desktop et Android Studio ; prévoir un téléphone Android de test (étape 2).
4. **Porteur du projet :** rechercher un partenaire technique OpenSRP et un développeur Kotlin.
