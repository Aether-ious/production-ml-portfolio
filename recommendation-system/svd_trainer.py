import pandas as pd
import requests
import zipfile
import io
import os
from pathlib import Path
from surprise import Reader, Dataset, SVD
import joblib

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RATINGS_PATH = DATA_DIR / 'ratings.csv'
SVD_MODEL_PATH = DATA_DIR / 'svd_model.pkl'
TEMP_ZIP_PATH = DATA_DIR / "ml-latest-small.zip"
DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def train_svd():
    print("1. Downloading Ratings Data...")
    # NOTE: Assuming you already ran setup_and_train.py which downloaded the ZIP
    if not os.path.exists(RATINGS_PATH):
        try:
            # Download and extract the ratings file
            response = requests.get(DATA_URL, stream=True)
            zip_ref = zipfile.ZipFile(io.BytesIO(response.content))
            zip_ref.extract('ml-latest-small/ratings.csv', path=DATA_DIR)
            os.rename(DATA_DIR / 'ml-latest-small' / 'ratings.csv', RATINGS_PATH)
            os.rmdir(DATA_DIR / 'ml-latest-small')
        except Exception as e:
            print(f"❌ Error during ratings download: {e}")
            return
        
    print("2. Loading and Preprocessing Ratings...")
    # CRITICAL FIX: Load IDs as STRINGS, as SVD works best with IDs as identifiers
    ratings_df = pd.read_csv(RATINGS_PATH, dtype={'userId': str, 'movieId': str}) 

    # Required format for Surprise: (user, item, rating)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings_df[['userId', 'movieId', 'rating']], reader
    )
    
    trainset = data.build_full_trainset()
    
    print("3. Training SVD Model (Collaborative Filtering)...")
    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    
    # Save the trained model's "brain"
    joblib.dump(svd, SVD_MODEL_PATH)
    print(f"✅ SVD Model trained and saved to {SVD_MODEL_PATH}")

if __name__ == "__main__":
    train_svd()