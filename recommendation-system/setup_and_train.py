import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import zipfile
import io


import os
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
TEMP_ZIP_PATH = DATA_DIR / "ml-latest-small.zip"

# MovieLens official download link
DATA_URL = "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def setup_and_train():
    print("1. Downloading MovieLens Data (1MB)...")
    try:
        response = requests.get(DATA_URL, stream=True)
        response.raise_for_status()
        
        # Save zip file to data folder
        with open(TEMP_ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("   ✅ Download complete.")

    except Exception as e:
        print(f"   ❌ Error downloading MovieLens: {e}")
        return

    print("2. Extracting movies.csv...")
    try:
        with zipfile.ZipFile(TEMP_ZIP_PATH, 'r') as zip_ref:
            # Only extract the movies.csv file
            zip_ref.extract('ml-latest-small/movies.csv', path=DATA_DIR)
        
        # Rename and move the file to the correct location
        os.rename(
            DATA_DIR / 'ml-latest-small' / 'movies.csv',
            DATA_DIR / 'movies.csv'
        )
        # Clean up the unnecessary folder
        os.rmdir(DATA_DIR / 'ml-latest-small')
        os.remove(TEMP_ZIP_PATH) # Remove the downloaded zip file
        
        df = pd.read_csv(DATA_DIR / 'movies.csv')
    except Exception as e:
        print(f"   ❌ Error during extraction/reading: {e}")
        return

    # --- Processing ---
    # The movies.csv file only has title and genre, no plot. 
    # For a content-based recommender, we will use the combined string of Title and Genre as the "plot" for the AI to read.
    print("3. Creating Features (Title + Genre)...")
    df['plot'] = df['title'] + " " + df['genres'].str.replace('|', ' ', regex=False)
    
    # We must use the 'plot' column as the text input for our AI model
    df_clean = df.rename(columns={'movieId': 'id'}).drop(columns=['genres'])
    df_clean.to_csv(DATA_DIR / 'movies.csv', index=False)
    
    print("4. Converting Real Titles/Genres to Vectors (Embeddings)...")
    # This downloads the AI model (only happens once)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_clean['plot'].tolist(), show_progress_bar=True)

    print("5. Building FAISS Index...")
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, str(DATA_DIR / "movie_vectors.index"))

    print(f"✅ DONE! {len(df_clean)} movies indexed and ready.")

if __name__ == "__main__":
    setup_and_train()