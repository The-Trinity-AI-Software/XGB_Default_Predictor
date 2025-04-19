
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

def validate_input_data(df, feature_metadata_path=None, require_target=True):
    if feature_metadata_path:
        if not os.path.exists(feature_metadata_path):
            raise FileNotFoundError(f"Feature metadata file not found at {feature_metadata_path}")

        with open(feature_metadata_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                raise ValueError(f"Feature metadata file is empty: {feature_metadata_path}")
            try:
                metadata = json.loads(content)
            except json.JSONDecodeError:
                raise ValueError(f"Feature metadata is not valid JSON")

        required_columns = metadata.get('all_input_columns') or (
            metadata.get('numerical_features', []) + metadata.get('categorical_features', [])
        )
    else:
        required_columns = df.columns.tolist()

    # Fill missing required columns with placeholder values
    if not require_target:
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col in metadata.get('numerical_features', []) else ""

    # Confirm all required columns now exist
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")

    if require_target and 'default' not in df.columns:
        raise ValueError("Training data must include 'default' as the target column.")

    return True
