import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler, normalize, RobustScaler
from sklearn.metrics.pairwise import cosine_similarity

# THIS LINE IS CRITICAL:
from sklearn.feature_extraction.text import TfidfVectorizer 

from scipy.spatial.distance import cosine
import numpy as np
from typing import Tuple, Dict, Any


LEXICAL_CUES = r'\b(you should|you must|i suggest|my advice|best way is to)\b'
SEMANTIC_KEYWORDS = r'\b(sue|lawyer|court|filing|testimony|statute|jurisdiction|legal advice)\b'
PROMO_CUES = r'\b(free|limited|giveaway|discount|click here|watch now|c0mpanyname)\b'
OBFUSCATED_NAMES = r'\b(gamify|c0in|fr3e|cIick|Iink)\b' 


# --- Enhanced Text Processing Functions ---

def _clean_and_normalize_text(text: str) -> str:
    """
    Enhanced text cleaning and normalization for better feature extraction.
    """
    if not isinstance(text, str):
        return ''
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs but keep the fact that URLs were present
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Normalize repeated characters (e.g., "soooooo" -> "sooo")
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    
    # Normalize punctuation
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    
    # Remove excessive punctuation but keep some for sentiment
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

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

def calculate_simple_features(df: pd.DataFrame, scaler: RobustScaler = None) -> Tuple[pd.DataFrame, RobustScaler]:
    """Calculates and scales continuous structural features with enhanced preprocessing."""
    
    # Enhanced text cleaning
    df['comment_text'] = df['comment_text'].astype(str).fillna('')
    df['comment_text'] = df['comment_text'].apply(_clean_and_normalize_text)
    
    # More robust length calculation (character and word count)
    df['comment_length'] = df['comment_text'].apply(lambda x: len(x.split()))
    df['comment_char_length'] = df['comment_text'].apply(lambda x: len(x))
    
    # Enhanced exclamation frequency with better normalization
    df['exclamation_frequency'] = df['comment_text'].apply(_get_exclamation_frequency)
    
    # Additional text quality features
    df['avg_word_length'] = df['comment_text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
    df['punctuation_ratio'] = df['comment_text'].apply(lambda x: sum(1 for c in x if c in '!?.,;:') / len(x) if len(x) > 0 else 0)
    
    # Features to scale
    continuous_features = ['comment_length', 'comment_char_length', 'exclamation_frequency', 'avg_word_length', 'punctuation_ratio']
    
    # Handle outliers and missing values
    for feature in continuous_features:
        # Replace infinite values with NaN
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
        # Fill NaN with median
        df[feature] = df[feature].fillna(df[feature].median())
    
    if scaler is None:
        # Use RobustScaler for better outlier handling
        scaler = RobustScaler()
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
    
    # --- NEW FEATURE: Boundary Proximity Score ---
    if 'semantic_difference' in mean_vectors:
        # Retrieve the difference vector
        difference_vec = mean_vectors['semantic_difference']
        
        # Reshape the single difference vector for proper comparison (1 row, N features)
        difference_vec_reshaped = difference_vec.reshape(1, -1)
        
        # Calculate similarity between all comments and the semantic difference vector
        similarity_scores = cosine_similarity(X_current_tfidf, difference_vec_reshaped)
        
        # The final score indicates proximity to the violation concept
        df['boundary_proximity_score'] = similarity_scores.flatten()
        
        print(f"Calculated boundary proximity scores for {len(df)} comments")
    else:
        print("Warning: semantic_difference vector not found in mean_vectors. Skipping boundary proximity score.")
        df['boundary_proximity_score'] = 0.0
    
    return df

def calculate_archetype_vector(texts, tfidf_model):
    """
    Calculate the median vector from a collection of texts using the fitted TF-IDF model.
    This creates an archetype vector that represents the typical characteristics of the text collection.
    """
    if len(texts) == 0:
        # Return zero vector if no texts provided
        return np.zeros(tfidf_model.transform(['']).shape[1])
    
    # Transform texts to TF-IDF vectors
    X_exa = tfidf_model.transform(texts)
    X_exa_dense = X_exa.toarray()
    
    # Calculate median vector
    return np.median(X_exa_dense, axis=0)

def calculate_consistency_features(
    df: pd.DataFrame, 
    tfidf_vectorizer: TfidfVectorizer, 
    mean_vectors: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculates the consistency deviation feature by measuring how much the internal 
    similarity of a specific row's examples deviates from the average consistency 
    found across the entire dataset.
    """
    
    # Check if the required columns exist
    if 'positive_example_1' not in df.columns or 'positive_example_2' not in df.columns:
        print("Warning: positive_example_1 or positive_example_2 columns not found. Skipping consistency features.")
        return df
    
    # 1. Transform BOTH text columns using the FITTED tfidf_vectorizer
    text_1 = df['positive_example_1'].fillna('')
    text_2 = df['positive_example_2'].fillna('')

    X_ex1 = tfidf_vectorizer.transform(text_1)
    X_ex2 = tfidf_vectorizer.transform(text_2)

    # 2. Normalize and compute row-wise Cosine Similarity
    X_ex1_norm = normalize(X_ex1, norm='l2', axis=1)
    X_ex2_norm = normalize(X_ex2, norm='l2', axis=1)

    # Calculate cosine similarity between the two positive examples
    consistency_scores = (X_ex1_norm.multiply(X_ex2_norm)).sum(axis=1)
    df['example_consistency'] = np.asarray(consistency_scores).flatten()
    
    # 3. Calculate Global Statistics and Apply Scaling
    if 'consistency_mean' not in mean_vectors:
        # Training phase: Calculate and save the mean and standard deviation
        consistency_mean = df['example_consistency'].mean()
        consistency_std = df['example_consistency'].std()
        
        # Store these statistics in the mean_vectors dictionary
        mean_vectors['consistency_mean'] = consistency_mean
        mean_vectors['consistency_std'] = consistency_std
        
        print(f"Training: Calculated consistency_mean={consistency_mean:.4f}, consistency_std={consistency_std:.4f}")
    else:
        # Validation/Test phase: Load the saved mean and std dev
        consistency_mean = mean_vectors['consistency_mean']
        consistency_std = mean_vectors['consistency_std']
        
        print(f"Validation/Test: Using consistency_mean={consistency_mean:.4f}, consistency_std={consistency_std:.4f}")
    
    # Calculate the final feature (Z-score)
    if consistency_std > 0:  # Avoid division by zero
        df['consistency_deviation'] = (df['example_consistency'] - consistency_mean) / consistency_std
    else:
        df['consistency_deviation'] = 0.0  # If std is 0, set all values to 0
    
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

    # --- Enhanced Data Cleaning and Validation ---
    
    # CRITICAL FIX: RENAME 'body' to 'comment_text'
    if 'body' in df.columns:
        df = df.rename(columns={'body': 'comment_text'})
    df.columns = df.columns.str.strip() 
    if 'comment_text' not in df.columns:
        print(f"FATAL ERROR: Text column 'comment_text' not found. Available columns: {list(df.columns)}")
        raise KeyError('comment_text')
    
    # Data quality checks and validation
    print(f"Data validation: {len(df)} rows loaded")
    print(f"Missing values in comment_text: {df['comment_text'].isna().sum()}")
    print(f"Empty comments: {(df['comment_text'].str.len() == 0).sum()}")
    
    # Remove completely empty rows
    initial_rows = len(df)
    df = df[df['comment_text'].str.len() > 0]
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} empty comment rows")
    
    # Validate label column
    if 'rule_violation' in df.columns:
        label_distribution = df['rule_violation'].value_counts()
        print(f"Label distribution: {dict(label_distribution)}")
        print(f"Class balance: {label_distribution[1] / len(df):.3f} positive class")
    
    # FIX: Explicitly set the label column to the one that actually exists
    LABEL_COLUMNS = ['rule_violation'] 
    
    # 2. Calculate Simple Features and Scale 
    df, scaler = calculate_simple_features(df, scaler)

    # 3. Calculate Interaction Features
    df = calculate_interaction_features(df)
    
    # 4. TF-IDF and Mean Vector Calculation (If processing the TRAINING SET)
    if tfidf_model is None:
        print("Fitting TFIDF and calculating mean vectors for the first time...")
        
        # A. Fit TF-IDF with optimized parameters
        tfidf_params = tfidf_params if tfidf_params else {
            'max_features': 8000,           # Increased for better vocabulary coverage
            'stop_words': 'english',        # Remove common words
            'ngram_range': (1, 3),          # Include trigrams for better context
            'min_df': 2,                    # Ignore terms that appear in < 2 documents
            'max_df': 0.95,                 # Ignore terms that appear in > 95% of documents
            'sublinear_tf': True,           # Apply sublinear tf scaling (1 + log(tf))
            'norm': 'l2',                   # L2 normalization for better similarity
            'smooth_idf': True,             # Smooth IDF weights
            'lowercase': True,              # Convert to lowercase
            'strip_accents': 'unicode'      # Remove accents for better matching
        }
        tfidf_model = TfidfVectorizer(**tfidf_params)
        X_tfidf = tfidf_model.fit_transform(df['comment_text']).toarray()
        
        # B. Calculate Mean Vectors
        violation_mask = df['rule_violation'] == 1 
        
        MEAN_VIOLATION_VECTOR = X_tfidf[violation_mask].mean(axis=0)
        MEAN_SAFE_VECTOR = X_tfidf[~violation_mask].mean(axis=0)
        
        # C. Calculate Semantic Difference Vector (Boundary Proximity Feature)
        # Extract positive and negative example texts
        pos_ex1_texts = df['positive_example_1'].astype(str).fillna('').tolist()
        pos_ex2_texts = df['positive_example_2'].astype(str).fillna('').tolist()
        neg_ex1_texts = df['negative_example_1'].astype(str).fillna('').tolist()
        neg_ex2_texts = df['negative_example_2'].astype(str).fillna('').tolist()
        
        # Calculate archetype vectors for positive examples
        ARCHETYPE_VECTOR_1 = calculate_archetype_vector(pos_ex1_texts, tfidf_model)
        ARCHETYPE_VECTOR_2 = calculate_archetype_vector(pos_ex2_texts, tfidf_model)
        
        # Calculate median positive vector (average of the two positive archetypes)
        MEDIAN_POSITIVE_VECTOR = (ARCHETYPE_VECTOR_1 + ARCHETYPE_VECTOR_2) / 2
        
        # Calculate archetype vectors for negative examples
        ARCHETYPE_NEG_VECTOR_1 = calculate_archetype_vector(neg_ex1_texts, tfidf_model)
        ARCHETYPE_NEG_VECTOR_2 = calculate_archetype_vector(neg_ex2_texts, tfidf_model)
        
        # Calculate median negative vector (average of the two negative archetypes)
        MEDIAN_NEGATIVE_VECTOR = (ARCHETYPE_NEG_VECTOR_1 + ARCHETYPE_NEG_VECTOR_2) / 2
        
        # CRITICAL STEP: The Semantic Difference Vector
        SEMANTIC_DIFFERENCE_VECTOR = MEDIAN_POSITIVE_VECTOR - MEDIAN_NEGATIVE_VECTOR
        
        mean_vectors = {
            'violation': MEAN_VIOLATION_VECTOR, 
            'safe': MEAN_SAFE_VECTOR,
            'semantic_difference': SEMANTIC_DIFFERENCE_VECTOR
        }
        
        print(f"Training: Calculated semantic difference vector with shape {SEMANTIC_DIFFERENCE_VECTOR.shape}")
        
    # 5. Calculate Similarity Features (For all datasets)
    # This line now works for both cases because tfidf_model and mean_vectors 
    # are guaranteed to be defined (either passed in or calculated in Step 4)
    df = calculate_similarity_features(df, tfidf_model, mean_vectors)
    
    # 6. Calculate Consistency Features (Global Consistency Feature)
    df = calculate_consistency_features(df, tfidf_model, mean_vectors)
    
    # 7. Final Column Selection (Fixes df_final not defined)
    columns_to_keep = ['comment_text'] + [
        'comment_length', 'comment_char_length', 'exclamation_frequency', 'avg_word_length', 'punctuation_ratio',
        'legal_advice_interaction_feature', 'promo_persuasion_feature', 
        'similarity_to_violation', 'similarity_to_safe', 'consistency_deviation', 'boundary_proximity_score'
    ] + LABEL_COLUMNS 

    # CRITICAL FIX: Define df_final before returning it
    df_final = df[columns_to_keep] 

    print(f"Preprocessing complete. Final features: {list(df_final.columns)}")
    
    return df_final, tfidf_model, mean_vectors, scaler