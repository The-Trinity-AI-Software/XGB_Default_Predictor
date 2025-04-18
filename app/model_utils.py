
import os
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, roc_auc_score




def validate_input_data(df, feature_metadata_path=None, require_target=True):
    """
    Validates required columns and optionally checks for target.
    """
    if feature_metadata_path:
        if not os.path.exists(feature_metadata_path):
            raise FileNotFoundError(f"Feature metadata file not found at {feature_metadata_path}")

        with open(feature_metadata_path, 'r') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Feature metadata file is empty: {feature_metadata_path}")
            metadata = json.loads(content)

        required_columns = metadata.get('numerical_features', []) + metadata.get('categorical_features', [])
    else:
        required_columns = df.columns.tolist()

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if require_target and 'default' not in df.columns:
        raise ValueError("Missing target column 'default' for training.")

    return True
