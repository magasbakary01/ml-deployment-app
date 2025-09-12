
import os
from flask import Flask, render_template, request, flash
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__, template_folder='templates')
app.secret_key = SECRET_KEY

# Load the trained model
model_path = 'model.pkl'  # Ensure this path is correct
model = load_model_safe(model_path)

# Assuming label_encoder is available here as a pre-fitted LabelEncoder instance
label_encoder = None  # Replace None with the actual label encoder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the form data and convert to DataFrame
        prediction = predict_safe(request.form, model, label_encoder)
        # Process the prediction as needed and prepare the response
        # ...
        return render_template('predict.html', prediction=prediction)
    except ValueError as e:
        flash(str(e))
        return render_template('index.html')
    except Exception as e:
        flash(str(e))
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
