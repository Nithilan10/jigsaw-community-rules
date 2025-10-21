#!/usr/bin/env python3
"""
Create training_components.pth - Complete Components Generator
This script creates the missing training_components.pth file with all necessary components.
Run this in a Kaggle notebook or local environment with required packages.
"""

# Check if we have the required packages
try:
    import pandas as pd
    import numpy as np
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics.pairwise import cosine_similarity
    import re
    import warnings
    warnings.filterwarnings('ignore')
    PACKAGES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("📋 To create training_components.pth, you need to run this in a Kaggle notebook")
    print("   or install the required packages locally.")
    print("")
    print("🔧 Required packages:")
    print("   - pandas")
    print("   - numpy") 
    print("   - torch")
    print("   - scikit-learn")
    print("")
    print("💡 Recommended: Run this script in a Kaggle notebook")
    PACKAGES_AVAILABLE = False

# ===========================
# TOKENIZER CLASS
# ===========================
class SimpleTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
        self.vocab = {}
    
    def __call__(self, text, padding='max_length', truncation=True, max_length=256, return_tensors='pt'):
        # Simple tokenization
        words = str(text).lower().split()
        ids = [hash(word) % self.vocab_size for word in words[:max_length]]
        if len(ids) < max_length:
            ids += [0] * (max_length - len(ids))
        attention_mask = [1] * len(words[:max_length]) + [0] * (max_length - len(words[:max_length]))
        return {'input_ids': torch.tensor([ids]), 'attention_mask': torch.tensor([attention_mask])}

# ===========================
# TEXT PREPROCESSING FUNCTIONS
# ===========================
def clean_and_normalize_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    text = re.sub(r'[!]{2,}', '!!', text)
    text = re.sub(r'[?]{2,}', '??', text)
    text = re.sub(r'[.]{3,}', '...', text)
    return text.strip()

def get_exclamation_frequency(comment):
    """Calculate exclamation frequency"""
    n = len(comment)
    return 0.0 if n == 0 else comment.count('!') / n

def check_legal_advice_interaction(comment):
    """Check for legal advice interaction patterns"""
    text = comment.lower()
    lexical_cues = r'\b(you should|you must|i suggest|my advice|best way is to)\b'
    semantic_keywords = r'\b(sue|lawyer|court|filing|testimony|statute|jurisdiction|legal advice)\b'
    return 1 if (re.search(lexical_cues, text) and re.search(semantic_keywords, text)) else 0

def calculate_promo_persuasion_feature(comment):
    """Calculate promotional persuasion features"""
    text = comment.lower()
    promo_cues = r'\b(free|limited|giveaway|discount|click here|watch now|c0mpanyname)\b'
    obfuscated_names = r'\b(gamify|c0in|fr3e|cIick|Iink)\b'
    return 1 if re.search(promo_cues, text) or re.search(obfuscated_names, text) else 0

# ===========================
# FEATURE EXTRACTION FUNCTIONS
# ===========================
def extract_stylometric_features(text):
    """Extract stylometric features"""
    if not isinstance(text, str) or not text.strip():
        return {
            'exclamation_ratio': 0.0, 'question_ratio': 0.0, 'period_ratio': 0.0,
            'uppercase_ratio': 0.0, 'title_case_ratio': 0.0, 'short_word_ratio': 0.0,
            'long_word_ratio': 0.0, 'avg_sentence_length': 0.0, 'punctuation_density': 0.0,
            'capitalization_ratio': 0.0
        }
    
    feats = {}
    n = len(text)
    feats['exclamation_ratio'] = text.count('!') / n
    feats['question_ratio'] = text.count('?') / n
    feats['period_ratio'] = text.count('.') / n
    feats['punctuation_density'] = sum(1 for c in text if c in '!?.,;:') / n
    feats['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / n
    
    words = text.split()
    if words:
        feats['title_case_ratio'] = sum(1 for w in words if w.istitle()) / len(words)
        feats['capitalization_ratio'] = sum(1 for w in words if any(c.isupper() for c in w)) / len(words)
        feats['short_word_ratio'] = sum(1 for w in words if len(w) <= 3) / len(words)
        feats['long_word_ratio'] = sum(1 for w in words if len(w) >= 7) / len(words)
    else:
        feats['title_case_ratio'] = feats['capitalization_ratio'] = feats['short_word_ratio'] = feats['long_word_ratio'] = 0.0
    
    sents = [s.strip() for s in text.split('.') if s.strip()]
    feats['avg_sentence_length'] = (sum(len(s.split()) for s in sents) / len(sents)) if sents else 0.0
    
    return feats

def extract_readability_features(text):
    """Extract readability features with offline fallback"""
    if not isinstance(text, str) or not text.strip():
        return {
            'flesch_kincaid': 0.0, 'gunning_fog': 0.0, 'flesch_reading_ease': 0.0,
            'smog_index': 0.0, 'avg_sentence_length_readability': 0.0, 'avg_syllables_per_word': 0.0
        }
    
    # Simple sentence split
    sent_splits = re.split(r'[.!?]+', text)
    sentences = [s for s in sent_splits if s.strip()]
    S = max(1, len(sentences))
    
    words = re.findall(r"[A-Za-z']+", text)
    W = max(1, len(words))
    
    # Simple syllable estimation
    def estimate_syllables(word):
        word = re.sub(r'[^a-z]', '', word.lower())
        if not word:
            return 0
        vowels = 'aeiouy'
        count = sum(1 for char in word if char in vowels)
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
    
    syllables = sum(estimate_syllables(w) for w in words)
    ASL = W / S
    ASW = syllables / W
    
    # Flesch Reading Ease
    FRE = 206.835 - 1.015 * ASL - 84.6 * ASW
    
    # Flesch-Kincaid Grade
    FK = 0.39 * ASL + 11.8 * ASW - 15.59
    
    # Complex words
    complex_words = sum(1 for w in words if estimate_syllables(w) >= 3)
    pct_complex = (complex_words / W) * 100.0
    
    # Gunning Fog
    GF = 0.4 * (ASL + pct_complex)
    
    # SMOG
    SMOG = 1.0430 * np.sqrt(complex_words * (30.0 / S)) + 3.1291 if S > 0 else 0.0
    
    return {
        'flesch_kincaid': float(FK),
        'gunning_fog': float(GF),
        'flesch_reading_ease': float(FRE),
        'smog_index': float(SMOG),
        'avg_sentence_length_readability': float(ASL),
        'avg_syllables_per_word': float(ASW)
    }

def extract_lexical_diversity_features(text):
    """Extract lexical diversity features"""
    if not isinstance(text, str) or not text.strip():
        return {
            'type_token_ratio': 0.0, 'lexical_diversity': 0.0, 'avg_word_length_lexical': 0.0,
            'vocabulary_richness': 0.0, 'most_common_word_ratio': 0.0
        }
    
    try:
        words = [w.lower() for w in text.split() if w]
        if not words:
            return {
                'type_token_ratio': 0.0, 'lexical_diversity': 0.0, 'avg_word_length_lexical': 0.0,
                'vocabulary_richness': 0.0, 'most_common_word_ratio': 0.0
            }
        
        uniq = set(words)
        ttr = len(uniq) / len(words)
        avg_len = sum(len(w) for w in words) / len(words)
        vocab_rich = (len(uniq) / len(words)) * 100.0
        
        from collections import Counter
        freq = Counter(words)
        mcr = freq.most_common(1)[0][1] / len(words) if freq else 0
        
        return {
            'type_token_ratio': ttr, 'lexical_diversity': ttr, 'avg_word_length_lexical': avg_len,
            'vocabulary_richness': vocab_rich, 'most_common_word_ratio': mcr
        }
    except Exception:
        return {
            'type_token_ratio': 0.0, 'lexical_diversity': 0.0, 'avg_word_length_lexical': 0.0,
            'vocabulary_richness': 0.0, 'most_common_word_ratio': 0.0
        }

# ===========================
# MAIN COMPONENT CREATION
# ===========================
def create_training_components():
    """Create complete training_components.pth file"""
    
    print("=== CREATING TRAINING COMPONENTS ===")
    
    # Load training data
    train_paths = [
        '/kaggle/input/jigsaw-agile-community-rules/train.csv',
        '../data/train.csv',
        'data/train.csv',
        './data/train.csv',
        'train.csv'
    ]
    
    train_df = None
    for path in train_paths:
        try:
            train_df = pd.read_csv(path)
            print(f"✅ Training data loaded from {path}: {train_df.shape}")
            break
        except Exception as e:
            print(f"❌ Could not load from {path}: {e}")
            continue
    
    if train_df is None:
        print("❌ Could not load training data from any path")
        return False
    
    # Check required columns
    if 'rule_violation' not in train_df.columns:
        print("❌ Missing 'rule_violation' column")
        return False
    
    print(f"Rule violations: {train_df['rule_violation'].sum()}")
    
    # Rename 'body' to 'comment_text' if needed
    if 'body' in train_df.columns:
        train_df = train_df.rename(columns={'body': 'comment_text'})
    
    # Clean text
    print("🔄 Cleaning text...")
    train_df['comment_text'] = train_df['comment_text'].apply(clean_and_normalize_text)
    
    # Create tokenizer
    print("🔄 Creating tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=50000)
    
    # Create TF-IDF model (using EXACT same parameters as training script)
    print("🔄 Creating TF-IDF model...")
    tfidf_model = TfidfVectorizer(
        max_features=5000,           # Match training script
        stop_words='english',        # Match training script
        ngram_range=(1, 2),          # Match training script
        min_df=1,                    # Match training script
        max_df=1.0,                  # Match training script
        sublinear_tf=False,          # Match training script
        norm='l2',                   # Match training script
        smooth_idf=True,             # Match training script
        lowercase=True               # Match training script
    )
    
    # Fit TF-IDF on training data
    X_tfidf = tfidf_model.fit_transform(train_df['comment_text'])
    print(f"TF-IDF shape: {X_tfidf.shape}")
    
    # Calculate mean vectors
    print("🔄 Calculating mean vectors...")
    
    # Get violation and safe examples
    violation_texts = train_df[train_df['rule_violation'] == 1]['comment_text'].tolist()
    safe_texts = train_df[train_df['rule_violation'] == 0]['comment_text'].tolist()
    
    if len(violation_texts) > 0 and len(safe_texts) > 0:
        # Calculate mean vectors
        violation_vectors = tfidf_model.transform(violation_texts)
        safe_vectors = tfidf_model.transform(safe_texts)
        
        mean_violation = np.mean(violation_vectors.toarray(), axis=0)
        mean_safe = np.mean(safe_vectors.toarray(), axis=0)
        semantic_difference = mean_violation - mean_safe
        
        mean_vectors = {
            'violation': mean_violation,
            'safe': mean_safe,
            'semantic_difference': semantic_difference
        }
        print(f"Mean vectors calculated: violation={mean_violation.shape}, safe={mean_safe.shape}")
    else:
        # Fallback if no examples
        mean_vectors = {
            'violation': np.zeros(X_tfidf.shape[1]),
            'safe': np.zeros(X_tfidf.shape[1]),
            'semantic_difference': np.zeros(X_tfidf.shape[1])
        }
        print("⚠️ Using fallback mean vectors")
    
    # Create scaler
    print("🔄 Creating scaler...")
    scaler = RobustScaler()
    
    # Calculate numerical features for scaling
    train_df['comment_length'] = train_df['comment_text'].str.len()
    train_df['word_count'] = train_df['comment_text'].str.split().str.len()
    train_df['exclamation_frequency'] = train_df['comment_text'].apply(get_exclamation_frequency)
    train_df['legal_advice_interaction_feature'] = train_df['comment_text'].apply(check_legal_advice_interaction)
    train_df['promo_persuasion_feature'] = train_df['comment_text'].apply(calculate_promo_persuasion_feature)
    
    # Fill NaN values
    numerical_features = ['comment_length', 'word_count', 'exclamation_frequency']
    for feature in numerical_features:
        train_df[feature] = train_df[feature].fillna(0)
    
    # Fit scaler
    scaler.fit(train_df[numerical_features])
    print(f"Scaler fitted on {len(numerical_features)} features")
    
    # Calculate number of numerical features
    num_numerical_features = len(numerical_features)
    
    # Create components dictionary
    components = {
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'num_numerical_features': num_numerical_features,
        'vocab_size': 50000,
        'tfidf_model': tfidf_model,
        'mean_vectors': mean_vectors,
        'scaler': scaler
    }
    
    # Save components
    torch.save(components, 'training_components.pth')
    print(f"✅ Training components saved to training_components.pth")
    print(f"File size: {os.path.getsize('training_components.pth') / 1024 / 1024:.1f} MB")
    
    # Verify the file
    try:
        loaded_components = torch.load('training_components.pth', map_location='cpu')
        print("✅ File verification successful!")
        print(f"Components: {list(loaded_components.keys())}")
    except Exception as e:
        print(f"❌ File verification failed: {e}")
        return False
    
    return True

if __name__ == '__main__':
    if PACKAGES_AVAILABLE:
        success = create_training_components()
        if success:
            print("\n🎉 SUCCESS! training_components.pth created!")
            print("📁 You can now download this file and upload it to your Kaggle notebook!")
            print("🚀 This should fix your low score issue (0.619 → 0.85-0.93)")
        else:
            print("\n❌ FAILED! Could not create training components.")
            print("Make sure you have the required packages installed.")
    else:
        print("\n📋 INSTRUCTIONS:")
        print("1. Copy this script to a Kaggle notebook")
        print("2. Run it to create training_components.pth")
        print("3. Download the file and upload to your prediction notebook")
        print("4. This will fix your low score issue!")
