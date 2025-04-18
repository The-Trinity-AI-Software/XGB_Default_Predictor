
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import joblib
from app.model_utils import validate_input_data

def train_model(input_path: str, model_path: str, feature_path: str, xgb_params: dict = None) -> float:
    df = pd.read_csv(input_path)
    validate_input_data(df, require_target=True)

    y = df['default']
    X = df.drop(columns=['default'])

    # Save feature metadata
    joblib.dump({'features': X.columns.tolist()}, feature_path)

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    if not xgb_params:
        xgb_params = {
            "max_depth": 3,
            "min_child_weight": 1,
            "max_leaf_nodes": 31,
            "subsample": 0.8,
            "scale_pos_weight": 1.0,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "use_label_encoder": False,
            "eval_metric": "logloss"
        }

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBClassifier(**xgb_params))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)

    return pipeline.score(X, y)
