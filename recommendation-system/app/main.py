from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
from surprise.prediction_algorithms.matrix_factorization import SVD
import joblib
from decouple import config

# Load Data & Models Paths
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "movies.csv"
INDEX_PATH = BASE_DIR / "data" / "movie_vectors.index"
SVD_MODEL_PATH = BASE_DIR / "data" / "svd_model.pkl"

app = FastAPI(title="MovieLens Vector Recommendation API")

# Global variables to hold data
movies_df = None
index = None
svd_model = None 

# Hybrid Weight (Read from .env or default)
CF_WEIGHT = float(config('CF_WEIGHT', default=0.7)) 
CB_WEIGHT = 1.0 - CF_WEIGHT

class RecommendationRequest(BaseModel):
    watched_movie_id: int
    user_id: int = 1

def get_cf_score(user_id, movie_id):
    """Predicts a rating using the SVD model (CF), handling unseen IDs."""
    if svd_model is None:
        return 0
    
    try:
        # CRITICAL FIX: Ensure user_id and movie_id are converted to strings for SVD
        prediction = svd_model.predict(str(user_id), str(movie_id)).est
    except ValueError:
        # Robustness: Assign a neutral rating (midpoint of [0.5, 5]) if ID is not in training data
        prediction = 3.0 
        
    # Normalize score from [0.5, 5] to [0, 1]
    return (prediction - 0.5) / 4.5 

@app.on_event("startup")
def load_data():
    """Loads all models and data (FAISS, Movies, SVD)"""
    global movies_df, index, svd_model
    
    if not DATA_PATH.exists() or not INDEX_PATH.exists():
        raise RuntimeError("Core data files are missing! Run setup_and_train.py first.")

    # 1. Load the Movies DataFrame (CRITICAL FIX)
    try:
        # Ensure we load 'id' correctly for matching
        movies_df = pd.read_csv(DATA_PATH)
        
        if movies_df.empty or 'id' not in movies_df.columns:
            raise ValueError("movies.csv is empty or missing the required 'id' column.")
            
        # Enforce integer type on ID column to match POST request and prevent dtype errors
        movies_df['id'] = movies_df['id'].astype(int) 
        print(f"✅ DataFrame loaded with {len(movies_df)} movies.")
    except Exception as e:
        print(f"FATAL: Failed to read movies.csv: {e}")
        raise RuntimeError("Application failed to load movie data.")

    # 2. Load FAISS Index
    index = faiss.read_index(str(INDEX_PATH))
    
    # 3. Load SVD Model
    if SVD_MODEL_PATH.exists():
        svd_model = joblib.load(SVD_MODEL_PATH)
        print("✅ SVD Model (Collaborative Filtering) loaded.")
    else:
        print("⚠️ SVD Model not found. CF recommendations will be skipped.")


@app.get("/movies")
def list_movies():
    """Returns a list of random movies to browse and pick from."""
    # Renaming 'plot' back to 'features' for cleaner output
    sample = movies_df.sample(10).rename(columns={'plot': 'features'})
    return sample[['id', 'title', 'features']].to_dict(orient="records")

@app.post("/recommend")
def recommend_hybrid(request_body: RecommendationRequest): 
    """HYBRID Recommendation: Combines CBF (FAISS) and CF (SVD) scores."""
    
    # Extract values from the request body
    watched_movie_id = request_body.watched_movie_id
    user_id = request_body.user_id
    
    # --- 1. Content-Based Filtering (CBF) - FAISS Retrieval ---
    # CRITICAL: movies_df['id'] is now guaranteed to be an integer type
    movie_row = movies_df[movies_df['id'] == watched_movie_id]
    
    if movie_row.empty:
        raise HTTPException(status_code=404, detail="Watched movie ID not found in dataset.")

    index_pos = movie_row.index[0]
    vector = index.reconstruct(int(index_pos))
    D, I = index.search(np.array([vector]), 100) # Search a wide range (100 candidates)
    
    
    recommendation_scores = {}
    
    # --- 2. Score Combination and Hybridization ---
    for i, movie_index in enumerate(I[0]):
        if movie_index == index_pos: continue # Skip the movie itself

        movie_id = movies_df.iloc[movie_index]['id']
        
        # 2a. CBF Score (Normalized FAISS Distance)
        cbf_score = 1.0 / (1.0 + D[0][i]) 
        
        # 2b. CF Score (SVD Predicted Rating)
        cf_score = get_cf_score(user_id, movie_id)
        
        # 2c. Weighted Hybrid Score
        hybrid_score = (CB_WEIGHT * cbf_score) + (CF_WEIGHT * cf_score)
        
        recommendation_scores[movie_id] = hybrid_score

    # Sort by the final hybrid score
    sorted_recs = sorted(recommendation_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Retrieve the top 5 movie details
    top_5_ids = [mid for mid, score in sorted_recs[:5]]
    
    # Final results formatting
    results = movies_df[movies_df['id'].isin(top_5_ids)]
    results = results.rename(columns={'plot': 'features'})
    
    return {
        "model_type": f"Weighted Hybrid (CBF: {CB_WEIGHT} | CF: {CF_WEIGHT})",
        "because_you_watched": movie_row['title'].values[0],
        "we_recommend": results[['id', 'title', 'features']].to_dict(orient="records")
    }