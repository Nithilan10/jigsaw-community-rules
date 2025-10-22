# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # Not needed anymore
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import warnings
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocess import preprocess_data
from custom_model import CustomTransformerModel
from custom_loss import (CustomCostSensitiveLoss, CombinedAdvancedLoss)

# ============================================================================
# COMPARATIVE FEATURE ENGINEERING FOR TRAINING
# ============================================================================

def calculate_comparative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate powerful comparative features between positive and negative examples."""
    print("🚀 Calculating comparative features for training...")
    
    # Group by subreddit and rule to compare examples
    comparative_features = []
    
    for (subreddit, rule), group in df.groupby(['subreddit', 'rule']):
        print(f"📊 Processing {subreddit} - {rule}: {len(group)} examples")
        
        # Get positive and negative examples for this rule
        positive_examples = []
        negative_examples = []
        
        for _, row in group.iterrows():
            # Collect positive examples (violations)
            for col in ['positive_example_1', 'positive_example_2']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    positive_examples.append(str(row[col]).strip())
            
            # Collect negative examples (non-violations)
            for col in ['negative_example_1', 'negative_example_2']:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    negative_examples.append(str(row[col]).strip())
        
        if not positive_examples or not negative_examples:
            print(f"⚠️  Skipping {subreddit} - {rule}: insufficient examples")
            continue
        
        # Calculate comparative features for each comment in this group
        for _, row in group.iterrows():
            comment_text = str(row['comment_text']).strip()
            if not comment_text:
                continue
            
            features = {}
            
            # 1. Similarity to Positive Examples (Violations)
            pos_similarities = []
            for pos_ex in positive_examples:
                if pos_ex != comment_text:  # Don't compare to self
                    similarity = calculate_text_similarity(comment_text, pos_ex)
                    pos_similarities.append(similarity)
            
            features['similarity_to_violations'] = np.mean(pos_similarities) if pos_similarities else 0.0
            features['max_similarity_to_violations'] = np.max(pos_similarities) if pos_similarities else 0.0
            features['min_similarity_to_violations'] = np.min(pos_similarities) if pos_similarities else 0.0
            
            # 2. Similarity to Negative Examples (Non-violations)
            neg_similarities = []
            for neg_ex in negative_examples:
                if neg_ex != comment_text:  # Don't compare to self
                    similarity = calculate_text_similarity(comment_text, neg_ex)
                    neg_similarities.append(similarity)
            
            features['similarity_to_non_violations'] = np.mean(neg_similarities) if neg_similarities else 0.0
            features['max_similarity_to_non_violations'] = np.max(neg_similarities) if neg_similarities else 0.0
            features['min_similarity_to_non_violations'] = np.min(neg_similarities) if neg_similarities else 0.0
            
            # 3. Violation vs Non-violation Similarity Difference
            features['violation_similarity_diff'] = features['similarity_to_violations'] - features['similarity_to_non_violations']
            features['violation_similarity_ratio'] = (
                features['similarity_to_violations'] / (features['similarity_to_non_violations'] + 1e-8)
            )
            
            # 4. Text Length Comparison
            pos_lengths = [len(ex.split()) for ex in positive_examples]
            neg_lengths = [len(ex.split()) for ex in negative_examples]
            
            comment_length = len(comment_text.split())
            features['length_vs_violations'] = comment_length - np.mean(pos_lengths) if pos_lengths else 0
            features['length_vs_non_violations'] = comment_length - np.mean(neg_lengths) if neg_lengths else 0
            features['length_violation_diff'] = features['length_vs_violations'] - features['length_vs_non_violations']
            
            # 5. Complexity Comparison
            comment_complexity = calculate_text_complexity(comment_text)
            pos_complexities = [calculate_text_complexity(ex) for ex in positive_examples]
            neg_complexities = [calculate_text_complexity(ex) for ex in negative_examples]
            
            features['complexity_vs_violations'] = comment_complexity - np.mean(pos_complexities) if pos_complexities else 0
            features['complexity_vs_non_violations'] = comment_complexity - np.mean(neg_complexities) if neg_complexities else 0
            features['complexity_violation_diff'] = features['complexity_vs_violations'] - features['complexity_vs_non_violations']
            
            # 6. Legal Language Comparison
            comment_legal = count_legal_patterns(comment_text)
            pos_legal = [count_legal_patterns(ex) for ex in positive_examples]
            neg_legal = [count_legal_patterns(ex) for ex in negative_examples]
            
            features['legal_vs_violations'] = comment_legal - np.mean(pos_legal) if pos_legal else 0
            features['legal_vs_non_violations'] = comment_legal - np.mean(neg_legal) if neg_legal else 0
            features['legal_violation_diff'] = features['legal_vs_violations'] - features['legal_vs_non_violations']
            
            # 7. Promotional Language Comparison
            comment_promo = count_promotional_patterns(comment_text)
            pos_promo = [count_promotional_patterns(ex) for ex in positive_examples]
            neg_promo = [count_promotional_patterns(ex) for ex in negative_examples]
            
            features['promo_vs_violations'] = comment_promo - np.mean(pos_promo) if pos_promo else 0
            features['promo_vs_non_violations'] = comment_promo - np.mean(neg_promo) if neg_promo else 0
            features['promo_violation_diff'] = features['promo_vs_violations'] - features['promo_vs_non_violations']
            
            # 8. Emotional Intensity Comparison
            comment_emotion = count_emotional_words(comment_text)
            pos_emotion = [count_emotional_words(ex) for ex in positive_examples]
            neg_emotion = [count_emotional_words(ex) for ex in negative_examples]
            
            features['emotion_vs_violations'] = comment_emotion - np.mean(pos_emotion) if pos_emotion else 0
            features['emotion_vs_non_violations'] = comment_emotion - np.mean(neg_emotion) if neg_emotion else 0
            features['emotion_violation_diff'] = features['emotion_vs_violations'] - features['emotion_vs_non_violations']
            
            # 9. Question Pattern Comparison
            comment_questions = count_question_patterns(comment_text)
            pos_questions = [count_question_patterns(ex) for ex in positive_examples]
            neg_questions = [count_question_patterns(ex) for ex in negative_examples]
            
            features['questions_vs_violations'] = comment_questions - np.mean(pos_questions) if pos_questions else 0
            features['questions_vs_non_violations'] = comment_questions - np.mean(neg_questions) if neg_questions else 0
            features['questions_violation_diff'] = features['questions_vs_violations'] - features['questions_vs_non_violations']
            
            # 10. Advanced Similarity Features
            features['similarity_rank_violations'] = np.mean([1 if s > features['similarity_to_non_violations'] else 0 for s in pos_similarities]) if pos_similarities else 0
            features['similarity_rank_non_violations'] = np.mean([1 if s > features['similarity_to_violations'] else 0 for s in neg_similarities]) if neg_similarities else 0
            
            # 11. Rule-Specific Pattern Matching
            rule_keywords = extract_rule_keywords(rule)
            features['rule_keyword_match'] = count_rule_keywords(comment_text, rule_keywords)
            features['rule_keyword_vs_violations'] = features['rule_keyword_match'] - np.mean([count_rule_keywords(ex, rule_keywords) for ex in positive_examples]) if positive_examples else 0
            features['rule_keyword_vs_non_violations'] = features['rule_keyword_match'] - np.mean([count_rule_keywords(ex, rule_keywords) for ex in negative_examples]) if negative_examples else 0
            
            # 12. Subreddit-Specific Features
            subreddit_context = get_subreddit_context(subreddit)
            features['subreddit_context_match'] = count_subreddit_context(comment_text, subreddit_context)
            
            # 13. Advanced Text Patterns
            features['has_imperative'] = count_imperative_patterns(comment_text)
            features['has_conditional'] = count_conditional_patterns(comment_text)
            features['has_negation'] = count_negation_patterns(comment_text)
            
            # 14. Composite Violation Score (Enhanced)
            violation_indicators = [
                'violation_similarity_diff', 'length_violation_diff', 'complexity_violation_diff',
                'legal_violation_diff', 'promo_violation_diff', 'emotion_violation_diff',
                'rule_keyword_vs_violations', 'similarity_rank_violations'
            ]
            available_indicators = [col for col in violation_indicators if col in features]
            features['composite_violation_score'] = np.mean([features[col] for col in available_indicators]) if available_indicators else 0
            
            # 15. Violation Confidence Score
            features['violation_confidence'] = np.std([features[col] for col in available_indicators]) if available_indicators else 0
            
            comparative_features.append(features)
    
    # Convert to DataFrame and merge
    if comparative_features:
        comp_df = pd.DataFrame(comparative_features)
        df = pd.concat([df, comp_df], axis=1)
        print(f"✅ Added {len(comp_df.columns)} comparative features")
    else:
        print("⚠️  No comparative features generated")
    
    return df

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using simple word overlap."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def calculate_text_complexity(text: str) -> float:
    """Calculate text complexity score."""
    words = text.split()
    if not words:
        return 0.0
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in words])
    
    # Sentence count
    sentences = text.count('.') + text.count('!') + text.count('?')
    sentences = max(1, sentences)
    
    # Words per sentence
    words_per_sentence = len(words) / sentences
    
    # Complexity score
    complexity = (avg_word_length * 0.4 + words_per_sentence * 0.6)
    return complexity

def count_legal_patterns(text: str) -> int:
    """Count legal advice patterns."""
    legal_patterns = [
        r'\b(should|must|need to|recommend|suggest|advise|counsel)\b',
        r'\b(lawyer|attorney|legal|court|lawsuit|litigation)\b',
        r'\b(consult|hire|get|seek)\s+(a\s+)?(lawyer|attorney|legal)\b'
    ]
    
    count = 0
    for pattern in legal_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def count_promotional_patterns(text: str) -> int:
    """Count promotional patterns."""
    promo_patterns = [
        r'\b(free|limited|giveaway|discount|click here|watch now)\b',
        r'\b(limited time|act now|don\'t miss|exclusive)\b',
        r'\b(call now|order now|buy now|get it now)\b'
    ]
    
    count = 0
    for pattern in promo_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def count_emotional_words(text: str) -> int:
    """Count emotional words."""
    emotional_words = [
        'amazing', 'incredible', 'fantastic', 'terrible', 'awful', 'horrible',
        'love', 'hate', 'angry', 'furious', 'excited', 'disappointed'
    ]
    
    count = 0
    for word in emotional_words:
        count += text.lower().count(word)
    return count

def count_question_patterns(text: str) -> int:
    """Count question patterns."""
    return text.count('?') + len(re.findall(r'\b(why|how|what|when|where|who)\b', text, re.IGNORECASE))

def extract_rule_keywords(rule: str) -> list:
    """Extract keywords from rule text."""
    # Common rule keywords
    rule_keywords = []
    rule_lower = rule.lower()
    
    if 'advertising' in rule_lower or 'spam' in rule_lower:
        rule_keywords.extend(['advertising', 'spam', 'promotional', 'marketing', 'sell', 'buy', 'discount', 'free'])
    if 'legal' in rule_lower or 'advice' in rule_lower:
        rule_keywords.extend(['legal', 'advice', 'lawyer', 'attorney', 'court', 'lawsuit'])
    if 'harassment' in rule_lower or 'abuse' in rule_lower:
        rule_keywords.extend(['harassment', 'abuse', 'threat', 'insult', 'offensive'])
    if 'personal' in rule_lower or 'private' in rule_lower:
        rule_keywords.extend(['personal', 'private', 'information', 'contact', 'phone', 'email'])
    
    return rule_keywords

def count_rule_keywords(text: str, keywords: list) -> int:
    """Count rule-specific keywords in text."""
    count = 0
    text_lower = text.lower()
    for keyword in keywords:
        count += text_lower.count(keyword)
    return count

def get_subreddit_context(subreddit: str) -> list:
    """Get context-specific keywords for subreddit."""
    subreddit_contexts = {
        'legaladvice': ['legal', 'law', 'court', 'attorney', 'lawsuit', 'rights'],
        'personalfinance': ['money', 'investment', 'budget', 'debt', 'credit', 'loan'],
        'relationships': ['relationship', 'partner', 'marriage', 'dating', 'family'],
        'technology': ['tech', 'software', 'computer', 'programming', 'code'],
        'fitness': ['exercise', 'workout', 'gym', 'fitness', 'health', 'diet']
    }
    return subreddit_contexts.get(subreddit.lower(), [])

def count_subreddit_context(text: str, context_keywords: list) -> int:
    """Count subreddit-specific context keywords."""
    count = 0
    text_lower = text.lower()
    for keyword in context_keywords:
        count += text_lower.count(keyword)
    return count

def count_imperative_patterns(text: str) -> int:
    """Count imperative patterns (commands, instructions)."""
    imperative_patterns = [
        r'\b(you should|you must|you need to|you have to|you ought to)\b',
        r'\b(do this|try this|use this|get this|buy this)\b',
        r'\b(don\'t|never|always|make sure|be sure)\b'
    ]
    
    count = 0
    for pattern in imperative_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def count_conditional_patterns(text: str) -> int:
    """Count conditional patterns."""
    conditional_patterns = [
        r'\b(if|when|unless|provided that|in case)\b',
        r'\b(would|could|should|might|may)\b'
    ]
    
    count = 0
    for pattern in conditional_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

def count_negation_patterns(text: str) -> int:
    """Count negation patterns."""
    negation_patterns = [
        r'\b(no|not|never|none|nothing|nowhere|nobody)\b',
        r'\b(doesn\'t|don\'t|won\'t|can\'t|shouldn\'t|wouldn\'t)\b'
    ]
    
    count = 0
    for pattern in negation_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count

# Simple Tokenizer Class (no internet required)
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.vocab_size = vocab_size
        self.unk_token_id = 1
        self.pad_token_id = 0
        self.cls_token_id = 2
        self.sep_token_id = 3
        
    def build_vocab(self, texts):
        """Build vocabulary from training texts"""
        word_counts = Counter()
        for text in texts:
            words = str(text).lower().split()
            word_counts.update(words)
        
        # Add most common words to vocab
        for i, (word, count) in enumerate(word_counts.most_common(self.vocab_size - 4)):
            self.vocab[word] = i + 4
        
        print(f"Built vocabulary with {len(self.vocab)} words")
    
    def tokenize(self, text):
        """Simple word tokenization"""
        words = str(text).lower().split()
        return words
    
    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.unk_token_id))
        return ids
    
    def __call__(self, text, padding='max_length', truncation=True, max_length=256, return_tensors='pt'):
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Truncate if needed
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length-1] + [self.sep_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if needed
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': torch.tensor([input_ids]),
            'attention_mask': torch.tensor([attention_mask])
        }

# --- 0. Configuration and Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_MODEL_NAME = 'bert-base-uncased'  # Not used anymore but kept for compatibility
TRAIN_FILE_PATH = '../data/train.csv'
TEST_FILE_PATH = '../data/test.csv'
NUM_RULES = 1
BATCH_SIZE = 16  # Increased since no BERT
LEARNING_RATE = 1e-3  # Higher learning rate for simpler model
NUM_EPOCHS = 10
MAX_SEQ_LENGTH = 256
VALIDATION_SPLIT_RATIO = 0.15
RANDOM_SEED = 42
VOCAB_SIZE = 50000

# Enhanced Training Parameters
USE_LEARNING_RATE_SCHEDULING = True
USE_EARLY_STOPPING = True
USE_GRADIENT_CLIPPING = True
USE_MIXED_PRECISION = False
EARLY_STOPPING_PATIENCE = 3
GRADIENT_CLIP_NORM = 1.0
WARMUP_RATIO = 0.1

# Class Imbalance Handling
USE_CLASS_WEIGHTING = True
USE_FOCAL_LOSS = False  # DISABLED - can cause NaN with unstable predictions
USE_SMOTE = False

# Advanced Loss Function Parameters
USE_ADVANCED_LOSS = False  # DISABLED - use simpler loss first
USE_LABEL_SMOOTHING = False
USE_MIXUP_AUGMENTATION = False
USE_ADVANCED_REGULARIZATION = False

# --- Performance Configuration ---
ENABLE_SPACY_FEATURES = False

# LightGBM Configuration
USE_LIGHTGBM = True
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': RANDOM_SEED
}
LIGHTGBM_NUM_ROUNDS = 1000
LIGHTGBM_EARLY_STOPPING_ROUNDS = 50

# --- 1. Enhanced Training Utilities ---

class EarlyStopping:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_auc = 0.0
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_auc: float) -> bool:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                return True
            return False

def check_for_nan(tensor, name=""):
    """Check for NaN/Inf in tensors and print debug info"""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}: shape {tensor.shape}")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}: shape {tensor.shape}")
        return True
    return False

def safe_loss_computation(logits, labels, numerical_features, criterion, use_advanced_loss=False):
    """Safely compute loss with extensive NaN checking"""
    
    # Check inputs for NaN/Inf
    if check_for_nan(logits, "logits"):
        print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
    
    if check_for_nan(labels, "labels"):
        print(f"Labels unique values: {torch.unique(labels)}")
    
    if check_for_nan(numerical_features, "numerical_features"):
        print(f"Features NaN count: {torch.isnan(numerical_features).sum().item()}")
    
    try:
        if use_advanced_loss:
            loss = criterion(logits, labels)
        else:
            loss = criterion(logits, labels, numerical_features)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"❌ NaN/Inf loss detected: {loss.item()}")
            print(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"Labels range: [{labels.min().item():.4f}, {labels.max().item():.4f}]")
            # Return a small but valid loss to continue training
            return torch.tensor(1.0, requires_grad=True, device=logits.device)
        
        return loss
        
    except Exception as e:
        print(f"❌ Error in loss computation: {e}")
        # Return safe fallback loss
        return torch.tensor(1.0, requires_grad=True, device=logits.device)

# --- 2. Custom Dataset Class ---

class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df['comment_text'].values
        
        # Get all numerical features dynamically
        numerical_cols = [col for col in df.columns 
                         if col not in ['comment_text', 'rule_violation', 'subreddit', 'rule'] 
                         and str(df.dtypes[col]) in ['int64', 'float64']]
        
        # Convert to numpy first for NaN checking
        numerical_array = df[numerical_cols].values
        
        # Replace any NaN/inf in features with 0
        numerical_array = np.nan_to_num(numerical_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.numerical = torch.tensor(numerical_array, dtype=torch.float32)
        
        # Handle labels
        if 'rule_violation' in df.columns:
            labels_array = df['rule_violation'].values
            # Ensure labels are valid
            labels_array = np.nan_to_num(labels_array, nan=0.0)
            self.labels = torch.tensor(labels_array, dtype=torch.float32).unsqueeze(1)
        else:
            rule_cols = df.filter(regex='rule_').values
            rule_cols = np.nan_to_num(rule_cols, nan=0.0)
            self.labels = torch.tensor(rule_cols, dtype=torch.float32)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Dataset created: {len(self.texts)} samples, {len(numerical_cols)} features")
        print(f"Labels distribution: {torch.unique(self.labels, return_counts=True)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if pd.isna(text) or text is None:
            text = ""
        else:
            text = str(text)
        
        encoding = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'numerical_features': self.numerical[idx],
            'labels': self.labels[idx]
        }

# --- 3. Evaluation Function ---

def evaluate_model(model, dataloader, device=DEVICE):
    """Calculates AUC on the validation set."""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)

            logits = model(input_ids, attention_mask, numerical_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(batch['labels'].cpu().numpy())
    
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    try:
        auc_score = roc_auc_score(labels, probs)
        return auc_score, probs, 0.5
    except ValueError:
        return 0.5, probs, 0.5

# --- 4. Main Training Function ---

def train_model():
    
    # --- Data Loading and Splitting ---
    print("--- 1. Data Loading and Splitting ---")
    
    try:
        print(f"Loading data from: {TRAIN_FILE_PATH}")
        full_train_df = pd.read_csv(TRAIN_FILE_PATH)
        print(f"Data loaded successfully. Shape: {full_train_df.shape}")
        
        # Basic data validation
        print(f"Label column stats:")
        if 'rule_violation' in full_train_df.columns:
            print(f"Rule violation distribution: {full_train_df['rule_violation'].value_counts()}")
            print(f"Rule violation NaN count: {full_train_df['rule_violation'].isna().sum()}")
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return 
    
    # Split data
    if 'rule_violation' in full_train_df.columns:
        stratify_col = full_train_df['rule_violation']
    else:
        stratify_col = None

    train_df_raw, validation_df_raw = train_test_split(
        full_train_df,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_col 
    )

    print(f"Dataset split: Train={len(train_df_raw)} samples, Validation={len(validation_df_raw)} samples")

    # --- Preprocessing ---
    print("\nProcessing TRAINING data...")
    train_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
        file_path=None, 
        df_to_process=train_df_raw,
        enable_spacy=ENABLE_SPACY_FEATURES
    )
    
    # Add powerful comparative features for training
    print("\n🚀 Adding comparative features for training...")
    train_df_processed = calculate_comparative_features(train_df_processed)
    
    print("Processing VALIDATION data...")
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None, 
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler,
        enable_spacy=ENABLE_SPACY_FEATURES
    )
    
    # Add comparative features for validation (using training examples as reference)
    print("\n🚀 Adding comparative features for validation...")
    validation_df_processed = calculate_comparative_features(validation_df_processed)
    
    # Validate processed data
    print(f"Processed train shape: {train_df_processed.shape}")
    print(f"Processed validation shape: {validation_df_processed.shape}")
    
    # Check for NaN in processed data
    if train_df_processed.isna().any().any():
        print("⚠️ NaN values detected in processed training data")
        train_df_processed = train_df_processed.fillna(0)
    
    if validation_df_processed.isna().any().any():
        print("⚠️ NaN values detected in processed validation data")
        validation_df_processed = validation_df_processed.fillna(0)

    # --- Train LightGBM Model ---
    if USE_LIGHTGBM:
        try:
            import lightgbm as lgb
            
            numerical_cols = [col for col in train_df_processed.columns 
                             if col not in ['comment_text', 'rule_violation', 'subreddit', 'rule'] 
                             and str(train_df_processed.dtypes[col]) in ['int64', 'float64']]
            
            X_train = train_df_processed[numerical_cols].fillna(0).values
            y_train = train_df_processed['rule_violation'].fillna(0).values
            X_val = validation_df_processed[numerical_cols].fillna(0).values
            y_val = validation_df_processed['rule_violation'].fillna(0).values
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            print("Training LightGBM model...")
            lightgbm_model = lgb.train(
                LIGHTGBM_PARAMS,
                train_data,
                num_boost_round=LIGHTGBM_NUM_ROUNDS,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(LIGHTGBM_EARLY_STOPPING_ROUNDS)]
            )
            
            val_preds = lightgbm_model.predict(X_val)
            lightgbm_auc = roc_auc_score(y_val, val_preds)
            print(f"LightGBM Validation AUC: {lightgbm_auc:.4f}")
            
        except Exception as e:
            print(f"LightGBM training failed: {e}")
            lightgbm_auc = 0.0

    # --- Setup DataLoader ---
    print("\nSetting up tokenizer and datasets...")
    try:
        # Build vocabulary from training texts
        tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
        tokenizer.build_vocab(train_df_processed['comment_text'].values)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        return
    
    train_dataset = CustomDataset(train_df_processed, tokenizer, MAX_SEQ_LENGTH)
    validation_dataset = CustomDataset(validation_df_processed, tokenizer, MAX_SEQ_LENGTH)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model and Loss Initialization ---
    print("\n--- 2. Model and Loss Initialization ---")
    
    # Count numerical features
    numerical_cols = [col for col in train_df_processed.columns 
                     if col not in ['comment_text', 'rule_violation', 'subreddit', 'rule'] 
                     and str(train_df_processed.dtypes[col]) in ['int64', 'float64']]
    num_numerical_features = len(numerical_cols)
    
    print(f"DEBUG: Found {num_numerical_features} numerical features")
    print(f"DEBUG: Numerical columns: {numerical_cols[:10]}...")  # Show first 10
    print(f"DEBUG: All columns: {list(train_df_processed.columns)}")
    
    # Initialize model with new architecture
    model = CustomTransformerModel(
        transformer_name=TRANSFORMER_MODEL_NAME,  # Not used but kept for compatibility
        num_numerical_features=num_numerical_features,
        num_rules=NUM_RULES,
        vocab_size=VOCAB_SIZE
    ).to(DEVICE)
    
    # Use simpler loss function for stability
    criterion = CustomCostSensitiveLoss(
        rule_weights={0: 1.0},  # Simpler weights
        feature_weights={}       # No feature weights for now
    )

    # Conservative optimizer settings
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,  # Very low learning rate
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # --- Training Loop ---
    print(f"\n--- 3. Starting Training ---")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Learning rate: {LEARNING_RATE}")
    
    best_auc = 0.0
    nan_count = 0
    max_nan_batches = 10  # Stop if too many NaN batches
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0
        valid_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move data to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Check for NaN in input data
            if (check_for_nan(input_ids, "input_ids") or 
                check_for_nan(attention_mask, "attention_mask") or
                check_for_nan(numerical_features, "numerical_features") or
                check_for_nan(labels, "labels")):
                nan_count += 1
                print(f"⚠️ Skipping batch {batch_idx} due to NaN in inputs")
                if nan_count >= max_nan_batches:
                    print("❌ Too many NaN batches, stopping training")
                    return
                continue

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            
            # Safe loss computation
            loss = safe_loss_computation(
                logits, labels, numerical_features, 
                criterion, USE_ADVANCED_LOSS
            )
            
            # Skip if loss is still problematic
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                print(f"⚠️ Skipping batch {batch_idx} due to NaN loss")
                if nan_count >= max_nan_batches:
                    print("❌ Too many NaN losses, stopping training")
                    return
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            
            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None and check_for_nan(param.grad, f"grad_{name}"):
                    print(f"⚠️ NaN gradients in {name}, skipping update")
                    optimizer.zero_grad()
                    continue
            
            optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            if batch_idx > len(train_dataloader) * 2:
                print(f"⚠️ Breaking loop safety check")
                break

        if valid_batches == 0:
            print("❌ No valid batches in epoch, stopping training")
            break
            
        avg_loss = total_loss / valid_batches
        
        # Validation
        validation_auc, _, _ = evaluate_model(model, validation_dataloader, DEVICE)
        
        print(f"\n--- Epoch {epoch+1} Complete ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Validation AUC: {validation_auc:.4f}")
        print(f"NaN batches skipped: {nan_count}")
        
        # Save best model
        if validation_auc > best_auc and not np.isnan(validation_auc):
            torch.save(model.state_dict(), 'best_model.pth')
            # Also save training components
            torch.save({
                'tokenizer_vocab': tokenizer.vocab,
                'tokenizer_vocab_size': tokenizer.vocab_size,
                'num_numerical_features': num_numerical_features,
                'vocab_size': VOCAB_SIZE,
                'tfidf_model': tfidf_model,
                'mean_vectors': mean_vectors,
                'scaler': scaler
            }, 'training_components.pth')
            print(f"✅ New best model saved! AUC: {validation_auc:.4f}")
            best_auc = validation_auc
    
    print(f"\nTraining complete. Best validation AUC: {best_auc:.4f}")

# --- Execute Script ---
if __name__ == '__main__':
    train_model()