import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

# Setup directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

print("1. Generating synthetic production data...")
# We create data matching the schema.py structure exactly
n_rows = 10000
data = {
    'TransactionAmt': np.random.exponential(scale=100, size=n_rows),
    'ProductCD': np.random.choice(['W', 'C', 'R', 'H', 'S'], size=n_rows),
    'card1': np.random.randint(1000, 18000, size=n_rows),
    'card2': np.random.randint(100, 600, size=n_rows),
    'card3': np.random.choice([150, 185, None], size=n_rows),
    'card4': np.random.choice(['visa', 'mastercard', 'amex', 'discover'], size=n_rows),
    'card5': np.random.randint(100, 250, size=n_rows),
    'card6': np.random.choice(['credit', 'debit'], size=n_rows),
    'addr1': np.random.randint(100, 500, size=n_rows),
    'addr2': np.random.choice([87, 60, 96], size=n_rows),
    'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', None], size=n_rows),
    'isFraud': np.random.choice([0, 1], size=n_rows, p=[0.97, 0.03]) # 3% fraud
}

# Fill generic C, D, M columns with random numbers
for i in range(1, 15):
    data[f'C{i}'] = np.random.poisson(1, size=n_rows)
for i in range(1, 16): # D columns usually represent time deltas
    data[f'dist{i}'] = np.random.randint(0, 500, size=n_rows) if i < 3 else None
for i in range(1, 10): # M columns
    data[f'M{i}'] = np.random.choice(['T', 'F', None], size=n_rows)

df = pd.DataFrame(data)

# Save raw data for reference
df.to_csv(DATA_DIR / "train.csv", index=False)

print("2. Preprocessing...")
# Simple preprocessing for the demo: Label Encode categoricals
# In a real real-world scenario, you'd use a Scikit-Learn Pipeline, but let's keep it compatible with the simple Model class
cat_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain'] + [f'M{i}' for i in range(1, 10)]

# We need to save mappings to handle inference correctly
mappings = {}
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Missing')
        unique_vals = df[col].unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        mappings[col] = mapping
        df[col] = df[col].map(mapping)

# Save mappings so API can use them
joblib.dump(mappings, MODEL_DIR / "category_mappings.pkl")

# Features to use
features = [c for c in df.columns if c != 'isFraud']
X = df[features].fillna(-1) # Simple imputation for tree models
y = df['isFraud']

print("3. Training Models...")
# Train XGBoost
xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, enable_categorical=False)
xgb_clf.fit(X, y)
xgb_clf.save_model(MODEL_DIR / "xgb_model.json")

# Train LightGBM
lgb_clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, verbose=-1)
lgb_clf.fit(X, y)
lgb_clf.booster_.save_model(MODEL_DIR / "lgb_model.txt")

print("4. Saving Reference Data for Drift Monitoring...")
# Save a sample of the training data (without target) for Evidently
ref_sample = df[features].sample(500)
joblib.dump(ref_sample, DATA_DIR / "reference_sample.pkl")

print("✅ DONE! System is ready to run.")