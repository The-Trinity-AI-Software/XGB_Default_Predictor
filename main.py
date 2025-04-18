# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:43:02 2025

@author: Ravi kiran Jonnalagadda

"""

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # ✅ Makes "app." imports work

from flask import Flask, render_template, request, send_file
import uuid
import pandas as pd
from app.model_utils import validate_input_data
from app.train_pipeline import train_model
from app.predict_pipeline import predict


app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
MODEL_PATH = os.path.join(BASE_DIR, "app", "xgb_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "app", "feature_metadata.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

AUC_SCORE = None
PREDICTION_CSV = os.path.join(PREDICTIONS_DIR, "default_predictions.csv")

@app.route('/')
def home():
    return render_template("index.html", auc=AUC_SCORE)

@app.route('/train', methods=['POST'])
def train():
    global AUC_SCORE
    file = request.files['train_file']
    train_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    file.save(train_path)

    xgb_params = {
        "max_depth": int(request.form.get("max_depth", 3)),
        "learning_rate": float(request.form.get("learning_rate", 0.1)),
        "n_estimators": int(request.form.get("n_estimators", 100)),
        "min_child_weight": int(request.form.get("min_child_weight", 1)),
        "max_leaves": int(request.form.get("max_leaves", 0)),
        "subsample": float(request.form.get("subsample", 1.0)),
        "scale_pos_weight": float(request.form.get("scale_pos_weight", 1.0))
    }

    AUC_SCORE = train_model(train_path, MODEL_PATH, FEATURE_PATH, xgb_params)
    return render_template("index.html", auc=AUC_SCORE)

@app.route('/predict', methods=['POST'])
def predict_view():
    test_file = request.files['test_file']
    test_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{test_file.filename}")
    test_file.save(test_path)

    predict(MODEL_PATH, FEATURE_PATH, test_path, PREDICTION_CSV)

    return render_template("index.html", auc=AUC_SCORE, predictions=True)

@app.route('/download')
def download_file():
    return send_file(PREDICTION_CSV, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
