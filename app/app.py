"""Application Flask : estimation du risque d'arrêt de la contraception en consultation PF.

Lancement local :  python -m app.app      (http://127.0.0.1:5000)
Production      :  waitress-serve --port=8000 --call app.app:create_app
"""
import os
import secrets

from flask import Flask, flash, redirect, render_template, request, url_for

from contraception_risk import counseling
from contraception_risk import features as F
from contraception_risk.predict import RiskModel


def create_app(models_dir=None):
    app = Flask(__name__)
    app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    model = RiskModel(models_dir) if models_dir else RiskModel()

    def form_page(values=None):
        return render_template('index.html', countries=F.COUNTRY_NAMES, regions=model.regions_by_country(),
                               v=values or {'country': 'GN'}, visit_types=counseling.VISIT_TYPES,
                               methods={k: m['label'] for k, m in counseling.METHODS.items()},
                               prior_reasons=counseling.PRIOR_REASONS)

    @app.get('/')
    def index():
        return form_page()

    @app.post('/predict')
    def predict():
        values = request.form.to_dict()
        try:
            counseling.validate(values)
            result = model.predict(values)
        except ValueError as e:
            flash(f"Formulaire incomplet ou invalide : {e}")
            return form_page(values), 400
        region = model.region_names.get(values.get('region'), '')
        return render_template('result.html', r=result, a=counseling.advise(values, result), region=region,
                               country=F.COUNTRY_NAMES[values['country']])

    @app.get('/modele')
    def about():
        return render_template('about.html', m=model.metrics, countries=F.COUNTRY_NAMES, labels=F.LABELS)

    @app.get('/health')
    def health():
        return {'status': 'ok', 'model_date': model.metrics.get('date')}

    @app.errorhandler(404)
    def not_found(_):
        return redirect(url_for('index'))

    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))
