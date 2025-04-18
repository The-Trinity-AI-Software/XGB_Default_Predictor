
# ✅ Add this at the top to make relative imports work when run independently
import os
import sys
import pandas as pd
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ✅ Correct import for model_utils inside the app folder
from app.model_utils import validate_input_data


def predict(model_path: str, feature_path: str, test_path: str, output_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(test_path)
    validate_input_data(df, feature_metadata_path=feature_path, require_target=False)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df['predicted_default'] = preds
    df['default_probability'] = probs
    df.to_csv(output_path, index=False)

