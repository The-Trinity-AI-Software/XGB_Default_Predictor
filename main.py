# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:43:02 2025

@author: Ravi kiran Jonnalagadda

"""

import os
import sys
import uuid
import pandas as pd
from flask import Flask, render_template, request, send_file
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from app.train_pipeline import train_model
from app.predict_pipeline import predict_and_score
from app.model_utils import validate_input_data

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
MODEL_PATH = os.path.join(BASE_DIR, "app", "xgb_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "app", "feature_metadata.json")
PREDICTION_JSON = os.path.join(PREDICTIONS_DIR, "default_predictions.json")
ROC_IMAGE_PATH = os.path.join("static", "roc_curve.png")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs("static", exist_ok=True)

PIPELINE = None
AUC_SCORE = None
TRAIN_PREVIEW = None
TEST_PREDICTIONS = None
CLASS_REPORT = None
ROC_CURVE_READY = False

@app.route('/')
def home():
    return render_template("index.html", auc=AUC_SCORE, predictions=False, model_trained=False,
                           preview_data=None, output_data=None, roc_ready=False, class_report=None)

@app.route('/tune', methods=['POST'])
def tune_model():
    global PIPELINE, AUC_SCORE, TEST_PREDICTIONS, ROC_CURVE_READY, CLASS_REPORT

    xgb_params = {
        "max_depth": int(request.form.get("max_depth", 3)),
        "learning_rate": float(request.form.get("learning_rate", 0.1)),
        "n_estimators": int(request.form.get("n_estimators", 100)),
        "min_child_weight": int(request.form.get("min_child_weight", 1)),
        "max_leaves": int(request.form.get("max_leaves", 31)),
        "subsample": float(request.form.get("subsample", 1.0)),
        "scale_pos_weight": float(request.form.get("scale_pos_weight", 1.0)),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    # === Reuse latest training data ===
    latest_train_file = sorted(
        [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if "train" in f],
        key=os.path.getmtime
    )[-1]
    PIPELINE, _ = train_model(latest_train_file, FEATURE_PATH, xgb_params)

    # === Reuse latest test data ===
    latest_test_file = sorted(
        [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if "test" in f],
        key=os.path.getmtime
    )[-1]
    test_df = pd.read_csv(latest_test_file)
    validate_input_data(test_df, feature_metadata_path=FEATURE_PATH, require_target=False)

    result_df, AUC_SCORE = predict_and_score(PIPELINE, test_df, PREDICTION_JSON)
    TEST_PREDICTIONS = result_df.head(10).to_html(classes='table table-bordered', index=False)

    ROC_CURVE_READY = False
    CLASS_REPORT = None
    if 'actual_default' in test_df.columns:
        RocCurveDisplay.from_predictions(test_df['actual_default'], result_df['default_probability'])
        plt.savefig(ROC_IMAGE_PATH)
        ROC_CURVE_READY = True
        report = classification_report(test_df['actual_default'], result_df['default'], output_dict=True)
        CLASS_REPORT = pd.DataFrame(report).transpose().to_html(classes='table table-sm table-bordered')

    return render_template("index.html", auc=AUC_SCORE, predictions=True, model_trained=True,
                           preview_data=None, output_data=TEST_PREDICTIONS,
                           roc_ready=ROC_CURVE_READY, class_report=CLASS_REPORT)


@app.route('/train', methods=['POST'])
def train():
    global PIPELINE, TRAIN_PREVIEW
    file = request.files['train_file']
    train_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    file.save(train_path)

    xgb_params = {
        "max_depth": int(request.form.get("max_depth", 3)),
        "learning_rate": float(request.form.get("learning_rate", 0.1)),
        "n_estimators": int(request.form.get("n_estimators", 100)),
        "min_child_weight": int(request.form.get("min_child_weight", 1)),
        "max_leaves": int(request.form.get("max_leaves", 31)),
        "subsample": float(request.form.get("subsample", 1.0)),
        "scale_pos_weight": float(request.form.get("scale_pos_weight", 1.0)),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    PIPELINE, TRAIN_PREVIEW = train_model(train_path, FEATURE_PATH, xgb_params)
    preview_df = pd.read_csv(train_path).head(10).to_html(classes='table table-bordered', index=False)

    return render_template("index.html", auc=None, predictions=False, model_trained=True,
                           preview_data=preview_df, output_data=None, roc_ready=False, class_report=None)

@app.route('/predict', methods=['POST'])
def predict_view():
    global AUC_SCORE, TEST_PREDICTIONS, ROC_CURVE_READY, CLASS_REPORT

    test_file = request.files['test_file']
    test_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{test_file.filename}")
    test_file.save(test_path)

    test_df = pd.read_csv(test_path)
    validate_input_data(test_df, feature_metadata_path=FEATURE_PATH, require_target=False)

    result_df, AUC_SCORE = predict_and_score(PIPELINE, test_df, PREDICTION_JSON)
    TEST_PREDICTIONS = result_df.head(10).to_html(classes='table table-bordered', index=False)

    if 'actual_default' in test_df.columns:
        RocCurveDisplay.from_predictions(test_df['actual_default'], result_df['default_probability'])
        plt.savefig(ROC_IMAGE_PATH)
        ROC_CURVE_READY = True

        report = classification_report(test_df['actual_default'], result_df['default'], output_dict=True)
        CLASS_REPORT = pd.DataFrame(report).transpose().to_html(classes='table table-sm table-bordered')

    return render_template("index.html", auc=AUC_SCORE, predictions=True, model_trained=True,
                           preview_data=None, output_data=TEST_PREDICTIONS,
                           roc_ready=ROC_CURVE_READY, class_report=CLASS_REPORT)

@app.route('/download')
def download_file():
    return send_file(PREDICTION_JSON, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
