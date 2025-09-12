import os
from flask import Flask, render_template, request, flash
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your_secret_key_here'


# ---------- Chargement modèle + métadonnées ----------
def load_model_bundle():
    """
    Charge model.pkl. Accepte soit:
      - un dict {"model": ..., "columns": [...], "label_encoder_country": ...}
      - directement un objet modèle
    Essaie aussi de récupérer les noms de variables depuis LightGBM (feature_name_).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model file not found at {model_path}")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    # Uniformiser le bundle
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        expected_cols = bundle.get("columns")
    else:
        model = bundle
        expected_cols = None

    # Si pas de colonnes fournies, tenter de les lire depuis le modèle (LightGBM)
    if expected_cols is None:
        try:
            # cas direct (LGBMClassifier/LGBMRegressor)
            if hasattr(model, "feature_name_"):
                expected_cols = list(model.feature_name_)
            # cas Pipeline: essayer de trouver l’étape finale
            elif hasattr(model, "named_steps"):
                for key in ["model", "classifier", "clf", "estimator"]:
                    step = model.named_steps.get(key)
                    if step is not None and hasattr(step, "feature_name_"):
                        expected_cols = list(step.feature_name_)
                        break
        except Exception:
            pass

    return {"model": model, "expected_columns": expected_cols}


BUNDLE = load_model_bundle()
MODEL = BUNDLE["model"]
EXPECTED_COLUMNS = BUNDLE["expected_columns"]  # peut être None


# ---------- Définition des variables attendues par le formulaire ----------
# NOTE: on traite country et fpmethnow comme ENTIERs (codes numériques),
# pour éviter un encodage arbitraire différent de l'entraînement.
VARIABLE_TYPES = {
    'age': int,
    'hhkidlt5': int,
    'hheadage': int,
    'cheb': int,
    'marstat': int,
    'kiddesire': int,
    'wealthq': int,
    'edyrtotal': int,
    'husfertpref': int,
    'fpldisreas5y': int,
    'religion': int,
    'famstructr': int,
    'radiobrig': int,
    'decfemhcare': int,
    'fphctalkfp': int,
    'tvbrig': int,
    'currwork': int,
    'fp1stuslvkid': int,
    'fpradiohr': int,
    'fptvhr': int,
    'fpposthr': int,
    'newsbrig': int,
    'fphomvisity': int,
    'hheadsex': int,
    'newsfq': int,
    'urban': int,
    'mobphone': int,
    'fertpref': int,
    'country': int,     # ⚠️ int (pas LabelEncoder à la volée)
    'fpmethnow': int    # ⚠️ on one-hot-encode sur base des codes
}


def _safe_cast(value, to_type, default_int=0, default_str=""):
    """Cast robuste des champs; gère les vides/non remplis."""
    if value is None or value == "":
        return default_int if to_type is int else default_str
    try:
        return to_type(value)
    except Exception:
        return default_int if to_type is int else default_str


def _align_to_expected_columns(df: pd.DataFrame, expected_cols):
    """
    Aligne df sur expected_cols:
      - ajoute les colonnes manquantes (0)
      - retire les colonnes inconnues
      - respecte l'ordre
    """
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    return df[expected_cols]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    metrics = {}

    try:
        # 1) Récupérer et caster proprement les entrées
        form_data = {}
        missing = []
        for var, t in VARIABLE_TYPES.items():
            raw = request.form.get(var)
            if raw is None or raw == "":
                missing.append(var)
            form_data[var] = _safe_cast(raw, t)

        # Optionnel: prévenir si des champs sont vides (mais on continue avec 0)
        if missing:
            flash("Certains champs n'étaient pas remplis: " + ", ".join(missing))

        X = pd.DataFrame([form_data])

        # 2) Encodage de fpmethnow (one-hot) SEULEMENT si le modèle l'attend en dummies
        #    On essaie d'inférer via EXPECTED_COLUMNS si dispo.
        need_dummies_for_fpmethnow = False
        if EXPECTED_COLUMNS is not None:
            # si on voit des colonnes qui ressemblent à fpmethnow_XXX → dummies attendus
            need_dummies_for_fpmethnow = any(
                (col.startswith("fpmethnow_") or col.startswith("fpmethnow__")) for col in EXPECTED_COLUMNS
            )

        if need_dummies_for_fpmethnow:
            X = pd.get_dummies(X, columns=["fpmethnow"], prefix="fpmethnow")
        # sinon, on laisse fpmethnow en entier tel quel

        # 3) (Optionnel) on peut aussi dummy-iser d'autres variables si le modèle l'exige.
        #    Ici on suppose que le reste a été entraîné en codage numérique direct.

        # 4) Alignement des colonnes si le modèle les expose
        if EXPECTED_COLUMNS is not None:
            X = _align_to_expected_columns(X, EXPECTED_COLUMNS)
        else:
            # Pas d'info → on tente quand même, mais certains modèles exigeront
            # exactement les mêmes colonnes qu'à l'entraînement.
            pass

        # 5) Prédiction
        yhat = MODEL.predict(X)
        # sécuriser la conversion
        yhat_idx = int(np.array(yhat).ravel()[0])

        # Probabilités (si dispo)
        if hasattr(MODEL, "predict_proba"):
            proba = MODEL.predict_proba(X)
            proba = np.array(proba)[0]  # shape (2,)
            p0 = float(proba[0]) if len(proba) > 1 else float(1.0 - proba[0])
            p1 = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # fallback: score de décision → approx sigmoid si 1D
            if hasattr(MODEL, "decision_function"):
                score = float(np.array(MODEL.decision_function(X)).ravel()[0])
                # sigmoid approx
                p1 = 1.0 / (1.0 + np.exp(-score))
                p0 = 1.0 - p1
            else:
                # aucun proba dispo
                p1 = float(yhat_idx)
                p0 = 1.0 - p1

        target_names = ['No Discontinuation', 'Discontinuation']
        predicted_label = target_names[yhat_idx]

        condition = 'Discontinuation' if p1 >= 0.5 else 'No Discontinuation'

        metrics['Prediction'] = condition
        metrics['No Discontinuation Probability'] = round(p0, 4)
        metrics['Discontinuation Probability'] = round(p1, 4)

        # Scores demo (y_true inconnu ici)
        y_true = [1]
        label_mapping = {'No Discontinuation': 0, 'Discontinuation': 1}
        y_pred = [label_mapping[predicted_label]]
        metrics['Accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        metrics['Recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics['F1 Score'] = float(f1_score(y_true, y_pred, zero_division=0))

    except Exception as e:
        # Log console + message visible
        app.logger.exception("Prediction error")
        flash(f"Prediction error: {e}")

    return render_template('predict.html', metrics=metrics)
    

if __name__ == '__main__':
    # Important: l’hôte et le port peuvent être ajustés au besoin
    app.run(debug=True)
