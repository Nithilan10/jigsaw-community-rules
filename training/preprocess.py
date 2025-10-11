import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler

# THIS LINE IS CRITICAL:
from sklearn.feature_extraction.text import TfidfVectorizer 

from scipy.spatial.distance import cosine
import numpy as np
from typing import Tuple, Dict, Any


LEXICAL_CUES = r'\b(you should|you must|i suggest|my advice|best way is to)\b'
SEMANTIC_KEYWORDS = r'\b(sue|lawyer|court|filing|testimony|statute|jurisdiction|legal advice)\b'
PROMO_CUES = r'\b(free|limited|giveaway|discount|click here|watch now|c0mpanyname)\b'
OBFUSCATED_NAMES = r'\b(gamify|c0in|fr3e|cIick|Iink)\b' 


# --- Helper Functions (No Change) ---

def _get_exclamation_frequency(comment: str) -> float:
    """Calculates the density of exclamation marks in the comment."""
    comment_len = len(comment)
    if comment_len == 0:
        return 0.0
    return comment.count('!') / comment_len

def _check_legal_advice_interaction(comment: str) -> int:
    """Checks for the presence of both an advice cue AND a legal keyword."""
    text = comment.lower()
    has_cue = re.search(LEXICAL_CUES, text) is not None
    has_keyword = re.search(SEMANTIC_KEYWORDS, text) is not None
    return 1 if has_cue and has_keyword else 0

def _calculate_promo_persuasion_feature(comment: str) -> int:
    """Checks for the presence of promotional cues OR obfuscated names (Spam/Promo signal)."""
    text = comment.lower()
    promo_count = len(re.findall(PROMO_CUES, text))
    obfuscated_count = len(re.findall(OBFUSCATED_NAMES, text))
    return 1 if promo_count > 0 or obfuscated_count > 0 else 0


# --- Feature Calculation Functions (No Change) ---

def calculate_simple_features(df: pd.DataFrame, scaler: MinMaxScaler = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Calculates and scales continuous structural features."""
    
    df['comment_text'] = df['comment_text'].astype(str).fillna('')
    
    
    df['comment_length'] = df['comment_text'].apply(lambda x: len(x.split()))

    
    df['exclamation_frequency'] = df['comment_text'].apply(_get_exclamation_frequency)

    
    continuous_features = ['comment_length', 'exclamation_frequency']
    
    if scaler is None:
        
        scaler = MinMaxScaler()
        df[continuous_features] = scaler.fit_transform(df[continuous_features])
    else:
    
        df[continuous_features] = scaler.transform(df[continuous_features])
    
    return df, scaler

def calculate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates rule-specific binary interaction features."""
    
    df['legal_advice_interaction_feature'] = df['comment_text'].apply(_check_legal_advice_interaction)
    df['promo_persuasion_feature'] = df['comment_text'].apply(_calculate_promo_persuasion_feature)
    
    return df

def calculate_similarity_features(
    df: pd.DataFrame, 
    tfidf_vectorizer: TfidfVectorizer, 
    mean_vectors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates the cosine similarity of each comment to the mean violation and safe vectors.
    """
    
    # Use TRANSFORM to vectorize based on the fitted vocabulary
    X_current_tfidf = tfidf_vectorizer.transform(df['comment_text'])
    
    mean_violation = mean_vectors['violation']
    mean_safe = mean_vectors['safe']
    
    violation_similarity = []
    safe_similarity = []
    
    # Efficiently calculate distances for all vectors
    for vector in X_current_tfidf:
        # Check for zero vectors (happens if comment is filtered entirely by stop words, etc.)
        vector_array = vector.toarray()[0]
        if np.sum(vector_array) == 0:
            # If a comment is empty after vectorization, set similarity to neutral values
            violation_similarity.append(0.0) 
            safe_similarity.append(0.0)
            continue
            
        # NOTE: Using sparse vector support with scipy.spatial.distance.cosine
        dist_to_violation = cosine(vector_array, mean_violation)
        violation_similarity.append(1 - dist_to_violation)

        dist_to_safe = cosine(vector_array, mean_safe)
        safe_similarity.append(1 - dist_to_safe)

    df['similarity_to_violation'] = violation_similarity
    df['similarity_to_safe'] = safe_similarity
    
    return df

# --- Master Preprocessing Function (The fix is here) ---

def preprocess_data(
    file_path: str = None, 
    df_to_process: pd.DataFrame = None, 
    tfidf_params: Dict[str, Any] = None,
    tfidf_model: TfidfVectorizer = None,
    mean_vectors: Dict[str, Any] = None,
    scaler: MinMaxScaler = None
) -> Tuple[pd.DataFrame, TfidfVectorizer, Dict[str, Any], MinMaxScaler]:
    """
    Orchestrates the loading, feature engineering, and scaling of the data.
    Accepts either a file_path (for initial load) or a DataFrame (for splits).
    """
    
    # 1. Load Data (Fixes UnboundLocalError)
    if df_to_process is not None:
        # Use the provided DataFrame (for train/validation splits)
        df = df_to_process.copy() 
    elif file_path is not None:
        # Load from file (for initial train.csv loading)
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            # Return dummy results and exit safely
            return pd.DataFrame(), TfidfVectorizer(max_features=1), {}, MinMaxScaler()
    else:
        # CRITICAL FIX: Ensure the function exits if no data is provided
        raise ValueError("Must provide either 'file_path' or 'df_to_process'.")

    # --- Data Cleaning and Label Definition ---
    
    # CRITICAL FIX: RENAME 'body' to 'comment_text'
    if 'body' in df.columns:
        df = df.rename(columns={'body': 'comment_text'})
    df.columns = df.columns.str.strip() 
    if 'comment_text' not in df.columns:
        print(f"FATAL ERROR: Text column 'comment_text' not found. Available columns: {list(df.columns)}")
        raise KeyError('comment_text')
    
    # FIX: Explicitly set the label column to the one that actually exists
    LABEL_COLUMNS = ['rule_violation'] 
    
    # 2. Calculate Simple Features and Scale 
    df, scaler = calculate_simple_features(df, scaler)

    # 3. Calculate Interaction Features
    df = calculate_interaction_features(df)
    
    # 4. TF-IDF and Mean Vector Calculation (If processing the TRAINING SET)
    if tfidf_model is None:
        print("Fitting TFIDF and calculating mean vectors for the first time...")
        
        # A. Fit TF-IDF
        tfidf_params = tfidf_params if tfidf_params else {'max_features': 5000, 'stop_words': 'english', 'ngram_range': (1, 2)}
        tfidf_model = TfidfVectorizer(**tfidf_params)
        X_tfidf = tfidf_model.fit_transform(df['comment_text']).toarray()
        
        # B. Calculate Mean Vectors
        violation_mask = df['rule_violation'] == 1 
        
        MEAN_VIOLATION_VECTOR = X_tfidf[violation_mask].mean(axis=0)
        MEAN_SAFE_VECTOR = X_tfidf[~violation_mask].mean(axis=0)
        
        mean_vectors = {'violation': MEAN_VIOLATION_VECTOR, 'safe': MEAN_SAFE_VECTOR}
        
    # 5. Calculate Similarity Features (For all datasets)
    # This line now works for both cases because tfidf_model and mean_vectors 
    # are guaranteed to be defined (either passed in or calculated in Step 4)
    df = calculate_similarity_features(df, tfidf_model, mean_vectors)
    
    # 6. Final Column Selection (Fixes df_final not defined)
    columns_to_keep = ['comment_text'] + [
        'comment_length', 'exclamation_frequency', 
        'legal_advice_interaction_feature', 'promo_persuasion_feature', 
        'similarity_to_violation', 'similarity_to_safe'
    ] + LABEL_COLUMNS 

    # CRITICAL FIX: Define df_final before returning it
    df_final = df[columns_to_keep] 

    print(f"Preprocessing complete. Final features: {list(df_final.columns)}")
    
    return df_final, tfidf_model, mean_vectors, scaler