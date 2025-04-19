
import os
import json
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model_utils import validate_input_data

def train_model(input_path: str, feature_path: str, xgb_params: dict = None):
    df = pd.read_csv(input_path)
    validate_input_data(df, require_target=True)

    y = df['default']
    X = df.drop(columns=['default'])

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump({
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "all_input_columns": X.columns.tolist()
        }, f, indent=2)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(**xgb_params))
    ])

    pipeline.fit(X, y)

    model_path = os.path.join(os.path.dirname(feature_path), "xgb_model.pkl")
    joblib.dump(pipeline, model_path)

    return pipeline, df
