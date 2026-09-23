"""Conseil selon la méthode choisie et alertes cliniques, en complément du score de risque.

La méthode choisie et les arrêts antérieurs sont aussi des variables du modèle ; ce module ajoute la fiche
de conseil propre à la méthode et des alertes lisibles par le soignant.

Contenu inspiré du manuel mondial de planification familiale de l'OMS. À FAIRE VALIDER PAR UN CLINICIEN
avant utilisation.
"""

VISIT_TYPES = {'start': "Premier démarrage", 'switch': "Changement de méthode", 'resupply': "Réapprovisionnement / suivi"}

METHODS = {
    'pill': {
        'label': "Pilule",
        'effects': "Saignements irréguliers ou spotting, nausées, maux de tête, surtout les 3 premiers mois ; "
                   "ils diminuent généralement ensuite.",
        'todo': "Prendre un comprimé chaque jour à la même heure. En cas d'oubli, suivre la conduite indiquée "
                "sur la notice ou revenir au centre.",
        'next': "Revenir avant la fin de la dernière plaquette.",
        'user_dependent': True,
    },
    'injectable': {
        'label': "Injectable",
        'effects': "Saignements irréguliers, puis souvent absence de règles : ce n'est pas dangereux. "
                   "Prise de poids possible. Le retour de la fécondité peut prendre plusieurs mois après l'arrêt.",
        'todo': "Noter la date de la prochaine injection.",
        'next': "Prochaine injection dans 3 mois ; revenir même en cas de retard.",
        'user_dependent': False,
    },
    'implant': {
        'label': "Implant",
        'effects': "Saignements irréguliers fréquents, surtout les premiers mois ; parfois absence de règles.",
        'todo': "L'implant peut être retiré à tout moment sur simple demande ; la fécondité revient rapidement.",
        'next': "Revenir en cas de gêne ou de questions ; durée d'action de 3 à 5 ans selon le type.",
        'user_dependent': False,
    },
    'iud': {
        'label': "DIU (stérilet)",
        'effects': "Règles plus abondantes ou plus longues et crampes les premiers mois (DIU au cuivre).",
        'todo': "Revenir en urgence en cas de douleur pelvienne forte, fièvre ou pertes anormales.",
        'next': "Visite de contrôle après les premières règles ou dans les 3 à 6 semaines.",
        'user_dependent': False,
    },
    'condom': {
        'label': "Préservatif",
        'effects': "Pas d'effet hormonal.",
        'todo': "Utiliser à chaque rapport. En cas de rupture, recourir à la contraception d'urgence.",
        'next': "Prévoir un stock suffisant ; envisager de l'associer à une autre méthode.",
        'user_dependent': True,
    },
    'lam': {
        'label': "MAMA (allaitement)",
        'effects': "Efficace seulement si l'allaitement est exclusif, sans retour des règles, et si le bébé a moins de 6 mois.",
        'todo': "Choisir dès maintenant la méthode qui prendra le relais.",
        'next': "Revenir avant les 6 mois du bébé, au retour des règles ou à l'introduction d'autres aliments.",
        'user_dependent': True,
    },
    'natural': {
        'label': "Méthode naturelle / retrait",
        'effects': "Aucun effet secondaire, mais un risque de grossesse plus élevé que les méthodes modernes.",
        'todo': "Informer sur les méthodes modernes et sur la contraception d'urgence.",
        'next': "Revenir à tout moment pour une autre méthode.",
        'user_dependent': True,
    },
    'other': {
        'label': "Autre méthode",
        'effects': "Voir les recommandations propres à la méthode.",
        'todo': "",
        'next': "",
        'user_dependent': False,
    },
}

PRIOR_REASONS = {'side_effects': "Effets secondaires / santé", 'husband': "Opposition du mari / partenaire",
                 'failure': "Grossesse sous méthode", 'access': "Accès / coût / inconfort", 'other': "Autre raison"}


def validate(form):
    """Vérifie les champs de counseling (tous facultatifs). Lève ValueError si une valeur est inconnue."""
    for name, allowed in [('visit_type', VISIT_TYPES), ('method', METHODS), ('prior_method', METHODS),
                          ('prior_reason', PRIOR_REASONS)]:
        v = form.get(name) or None
        if v is not None and v not in allowed:
            raise ValueError(f"modalité inconnue pour {name}: {v}")
    if (form.get('side_effects_now') or None) not in (None, '0', '1'):
        raise ValueError("valeur inconnue pour side_effects_now")


def advise(form, result):
    """Construit la fiche méthode et les alertes à partir du formulaire et de la prédiction du modèle."""
    method = form.get('method') or None
    card = METHODS.get(method)
    top_reason = result['reasons'][0][2]
    alerts = []

    if form.get('prior_disc') not in (None, '', '0'):
        txt = "A déjà arrêté une méthode alors qu'elle en avait besoin : c'est un facteur de risque d'arrêt reconnu."
        if form.get('prior_reason') in PRIOR_REASONS:
            txt += f" Motif : {PRIOR_REASONS[form['prior_reason']].lower()}."
        alerts.append(('warn', txt + " Revoir avec elle ce qui s'était passé."))
        if method and method == form.get('prior_method') and form.get('prior_reason') == 'side_effects':
            alerts.append(('warn', "Elle reprend la méthode arrêtée pour effets secondaires : expliquer comment les "
                                   "gérer et proposer une visite de contrôle précoce."))
    if form.get('side_effects_now') == '1':
        alerts.append(('warn', "Effets secondaires actuellement ressentis : les prendre en charge aujourd'hui et "
                               "proposer de changer de méthode plutôt que d'arrêter."))
    if top_reason == 'echec' and card and card['user_dependent']:
        alerts.append(('info', "Le motif d'arrêt le plus probable est une grossesse sous méthode, et la méthode "
                               "choisie dépend de l'utilisatrice : présenter les méthodes de longue durée (implant, DIU)."))
    if top_reason == 'effets_secondaires' and method in ('injectable', 'implant', 'iud', 'pill'):
        alerts.append(('info', "Le motif d'arrêt le plus probable est lié aux effets secondaires : insister sur les "
                               "effets attendus de cette méthode et sur leur caractère le plus souvent bénin."))
    if top_reason == 'mari':
        alerts.append(('info', "Le motif d'arrêt le plus probable est l'opposition du partenaire : proposer un "
                               "counseling en couple ; discuter de méthodes discrètes si elle le souhaite."))
    if method == 'lam':
        alerts.append(('info', "MAMA : planifier dès aujourd'hui la méthode de relais."))
    if form.get('visit_type') == 'resupply' and result['tier'] == 'élevé':
        alerts.append(('info', "Visite de réapprovisionnement d'une femme à risque élevé : vérifier sa satisfaction "
                               "et ses difficultés avec la méthode."))
    return {'method': card, 'visit': VISIT_TYPES.get(form.get('visit_type')), 'alerts': alerts}
