import joblib
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models"

class FraudModel:
    def __init__(self):
        # Load models
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(MODEL_DIR / "xgb_model.json")
        self.lgb = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model.txt"))
        
        # Load category mappings created during training
        self.mappings = joblib.load(MODEL_DIR / "category_mappings.pkl")
        
        # Define features order strictly
        self.features = self.xgb.feature_names_in_

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Create a copy to avoid setting copy warnings
        data = df.copy()
        
        # Apply the same mappings as training
        for col, mapping in self.mappings.items():
            if col in data.columns:
                # Convert to string first to handle potential None/NaN inputs from API
                data[col] = data[col].astype(str).fillna('Missing')
                # Map values; use -1 for unseen categories (safety net)
                data[col] = data[col].map(mapping).fillna(-1)
        
        # Ensure all columns from training exist, fill missing with -1
        for col in self.features:
            if col not in data.columns:
                data[col] = -1
                
        return data[self.features].fillna(-1)

    def predict_proba(self, df: pd.DataFrame) -> float:
        processed_df = self.preprocess(df)
        
        # XGBoost prediction
        xgb_prob = self.xgb.predict_proba(processed_df)[:, 1]
        
        # LightGBM prediction
        lgb_prob = self.lgb.predict(processed_df)
        
        # Average
        return (xgb_prob + lgb_prob) / 2