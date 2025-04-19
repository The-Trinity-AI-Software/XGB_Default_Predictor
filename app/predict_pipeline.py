
import os
import sys
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

# Ensure relative imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.model_utils import validate_input_data

FEATURE_PATH = "app/feature_metadata.json"

def predict_and_score(model, test_df, output_json_path):
    """
    Predicts default and probability for given test_df.
    Returns: enriched DataFrame with all original columns, plus predictions.
    """
    import copy
    original_df = copy.deepcopy(test_df)

    # Validate structure (model-required columns)
    validate_input_data(test_df, feature_metadata_path=FEATURE_PATH, require_target=False)

    # Predict using only required features
    preds = model.predict(test_df)
    probs = model.predict_proba(test_df)[:, 1]

    # Reattach predictions to full original input
    original_df['default'] = preds
    original_df['default_probability'] = probs

    # Save enriched result
    original_df.to_json(output_json_path, orient="records", indent=2)

    # AUC if actual provided
    auc_score = roc_auc_score(original_df['actual_default'], original_df['default_probability']) \
        if 'actual_default' in original_df.columns else None

    return original_df, auc_score
