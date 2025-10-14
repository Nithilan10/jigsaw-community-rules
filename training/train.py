# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple
from torch.cuda.amp import autocast, GradScaler
from collections import Counter

# Import custom modules
# NOTE: The import below assumes your data_preprocessing file is named 'preprocess.py'
from preprocess import preprocess_data
from custom_model import CustomTransformerModel
from custom_loss import (CustomCostSensitiveLoss, SGDAOptimizer, OGDAOptimizer, RobustLoss,
                        LabelSmoothingLoss, MixupLoss, AdvancedRegularizationLoss, CombinedAdvancedLoss)

# --- 0. Configuration and Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_MODEL_NAME = 'bert-base-uncased'  # BERT-base (best performer in your ensemble)
TRAIN_FILE_PATH = '../data/train.csv' # Use the actual file name
TEST_FILE_PATH = '../data/test.csv'   # Test file for final evaluation
NUM_RULES = 1  # IMPORTANT: Set this to your actual number of policy columns
BATCH_SIZE = 8  # Balanced for DeBERTa-v3 (larger model needs more data per batch)
LEARNING_RATE = 1e-5  # Base learning rate (will be scheduled)
NUM_EPOCHS = 6  # More epochs with early stopping
MAX_SEQ_LENGTH = 256
VALIDATION_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# Enhanced Training Parameters
USE_LEARNING_RATE_SCHEDULING = True  # Enable LR scheduling
USE_EARLY_STOPPING = True           # Enable early stopping
USE_GRADIENT_CLIPPING = True        # Enable gradient clipping
USE_MIXED_PRECISION = False         # Enable mixed precision training
EARLY_STOPPING_PATIENCE = 3         # Stop if no improvement for 3 epochs
GRADIENT_CLIP_NORM = 1.0           # Max gradient norm
WARMUP_RATIO = 0.1                 # 10% of training for warmup

# Class Imbalance Handling
USE_CLASS_WEIGHTING = True         # Enable class weighting
USE_FOCAL_LOSS = True              # Enable focal loss for class imbalance
USE_SMOTE = False                  # Enable SMOTE (synthetic minority oversampling)
FOCAL_LOSS_ALPHA = 0.25            # Focal loss alpha parameter
FOCAL_LOSS_GAMMA = 2.0             # Focal loss gamma parameter

# Advanced Loss Function Parameters
USE_ADVANCED_LOSS = True           # Enable advanced loss functions
USE_LABEL_SMOOTHING = True         # Enable label smoothing
USE_MIXUP_AUGMENTATION = False     # Enable mixup data augmentation
USE_ADVANCED_REGULARIZATION = True # Enable advanced regularization

# --- Performance Configuration ---
ENABLE_SPACY_FEATURES = False      # Enable spaCy POS/dependency features (very slow)

# Advanced Loss Configuration
LABEL_SMOOTHING_FACTOR = 0.1       # Label smoothing factor
MIXUP_ALPHA = 0.2                  # Mixup alpha parameter
WEIGHT_DECAY = 1e-4                # Weight decay coefficient
GRADIENT_PENALTY_WEIGHT = 0.1      # Gradient penalty weight

# Removed ensemble and SPD - focusing on single model + features
HYPERPARAMETER_TUNING = False
ENSEMBLE_MODE = False
SPD_MODE = False

# This list must exactly match the order of features created in data_preprocessing.py
NUMERICAL_FEATURES = [
    'exclamation_frequency', 
    'legal_advice_interaction_feature', 'promo_persuasion_feature', 
    'similarity_to_violation', 'similarity_to_safe', 'consistency_deviation', 'boundary_proximity_score',
    # Context-aware stylometric features (30 features: 10 base features × 3 comparison types)
    'exclamation_ratio_violation_vs_safe_diff', 'exclamation_ratio_violation_vs_safe_ratio', 'exclamation_ratio_violation_zscore',
    'question_ratio_violation_vs_safe_diff', 'question_ratio_violation_vs_safe_ratio', 'question_ratio_violation_zscore',
    'period_ratio_violation_vs_safe_diff', 'period_ratio_violation_vs_safe_ratio', 'period_ratio_violation_zscore',
    'uppercase_ratio_violation_vs_safe_diff', 'uppercase_ratio_violation_vs_safe_ratio', 'uppercase_ratio_violation_zscore',
    'title_case_ratio_violation_vs_safe_diff', 'title_case_ratio_violation_vs_safe_ratio', 'title_case_ratio_violation_zscore',
    'short_word_ratio_violation_vs_safe_diff', 'short_word_ratio_violation_vs_safe_ratio', 'short_word_ratio_violation_zscore',
    'long_word_ratio_violation_vs_safe_diff', 'long_word_ratio_violation_vs_safe_ratio', 'long_word_ratio_violation_zscore',
    'avg_sentence_length_violation_vs_safe_diff', 'avg_sentence_length_violation_vs_safe_ratio', 'avg_sentence_length_violation_zscore',
    'punctuation_density_violation_vs_safe_diff', 'punctuation_density_violation_vs_safe_ratio', 'punctuation_density_violation_zscore',
    'capitalization_ratio_violation_vs_safe_diff', 'capitalization_ratio_violation_vs_safe_ratio', 'capitalization_ratio_violation_zscore',
    # Advanced text features (filtered for high discrimination)
    # High-value POS features (8 features)
    'pos_adj_ratio', 'pos_adv_ratio', 'pos_aux_ratio', 'pos_conj_ratio', 'pos_intj_ratio', 'pos_pron_ratio', 'pos_propn_ratio', 'pos_verb_ratio',
    # Dependency features (4 features)
    'has_imperative', 'has_conditional', 'has_negation', 'has_auxiliary',
    # Readability features (4 features)
    'flesch_kincaid', 'gunning_fog', 'flesch_reading_ease', 'smog_index',
    # Lexical diversity features (4 features)
    'type_token_ratio', 'lexical_diversity', 'vocabulary_richness', 'most_common_word_ratio',
    # Domain-specific features (24 features)
    # Legal/Brand recognition features (10 features)
    'legal_terms_count', 'legal_terms_density', 'brand_mentions_count', 'brand_mentions_density',
    'lawsuit_patterns_count', 'has_lawsuit_patterns', 'legal_references_count', 'has_legal_references',
    'legal_advice_indicators', 'has_legal_advice',
    # Sentiment features (6 features)
    'positive_sentiment_count', 'negative_sentiment_count', 'positive_sentiment_ratio', 'negative_sentiment_ratio',
    'sentiment_polarity', 'emotional_intensity',
    # Formality features (7 features)
    'formal_words_count', 'informal_words_count', 'contractions_count', 'formal_words_ratio',
    'informal_words_ratio', 'contractions_ratio', 'formality_score',
    # Question pattern features (6 features)
    'question_marks_count', 'has_questions', 'question_words_count', 'question_words_ratio',
    'rhetorical_questions_count', 'has_rhetorical_questions',
    # Specificity features (17 features)
    'email_count', 'phone_count', 'url_count', 'contact_info_count', 'specific_action_count',
    'specific_number_count', 'specific_location_count', 'generic_phrase_count', 'specific_phrase_count',
    'contact_info_density', 'specific_action_density', 'specific_number_density', 'specific_location_density',
    'generic_phrase_density', 'specific_phrase_density', 'overall_specificity_score',
    # Advanced text processing features (48 features)
    # Advanced TF-IDF features (12 features)
    'standard_tfidf_sum', 'standard_tfidf_mean', 'standard_tfidf_max', 'standard_tfidf_std',
    'sublinear_tfidf_sum', 'sublinear_tfidf_mean', 'sublinear_tfidf_max', 'sublinear_tfidf_std',
    'bm25_sum', 'bm25_mean', 'bm25_max', 'bm25_std',
    # Word embedding features (18 features)
    'avg_word_length', 'max_word_length', 'min_word_length', 'word_length_std',
    'char_count', 'char_count_no_spaces', 'digit_count', 'alpha_count', 'special_char_count',
    'unique_words', 'total_words', 'word_diversity', 'most_frequent_word_count',
    'bigram_count', 'trigram_count', 'unique_bigrams', 'unique_trigrams',
    # Text augmentation features (8 features)
    'synonym_pattern_count', 'translation_pattern_count', 'repeated_words_count', 'frequent_words_count',
    'sentence_count', 'avg_sentence_length', 'complex_word_count', 'complex_word_ratio',
    # BERT sentence features (11 features)
    'sentence_count', 'avg_sentence_length', 'max_sentence_length', 'min_sentence_length', 'sentence_length_std',
    'paragraph_count', 'avg_paragraph_length', 'max_paragraph_length', 'min_paragraph_length', 'paragraph_length_std',
    'bigram_diversity',
    # Rule-specific comparison features (12 features)
    'rule_pos_similarity', 'rule_neg_similarity', 'rule_similarity_diff', 'rule_similarity_ratio',
    'rule_diff_alignment', 'rule_pos_consistency', 'rule_neg_consistency', 'rule_consistency_diff',
    'rule_pos_var_alignment', 'rule_neg_var_alignment', 'rule_var_alignment_diff', 'rule_violation_score'
    # Note: Feature selection & engineering features are dynamically generated and will be added automatically
]

# Dynamic feature count will be calculated at runtime
NUM_NUMERICAL_FEATURES = len(NUMERICAL_FEATURES)  # Base count, will be updated dynamically

# Custom Loss Weights (Reduced to prevent instability)
RULE_WEIGHTS = {
    0: 2.0,  # Reduced from 3.0 to prevent extreme losses
} 
FEATURE_WEIGHTS = {
    'legal_advice_interaction_feature': 2.0,  # Reduced from 8.0
    'promo_persuasion_feature': 2.0,          # Reduced from 9.0
    'comment_length_short': 1.5               # Reduced from 5.0
}

# --- 1. Enhanced Training Utilities ---

class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_auc = 0.0
        self.wait = 0
        self.stopped_epoch = 0
        
    def __call__(self, val_auc: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_auc: Current validation AUC
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.wait = 0
            return False  # Continue training
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.wait
                return True  # Stop training
            return False  # Continue training

def calculate_class_weights(labels: np.ndarray) -> dict:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Dictionary of class weights
    """
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Calculate weights inversely proportional to class frequency
    weights = {}
    for class_id, count in class_counts.items():
        weights[class_id] = total_samples / (num_classes * count)
    
    return weights

def apply_smote_oversampling(X_text: list, X_numerical: np.ndarray, y: np.ndarray) -> tuple:
    """
    Apply SMOTE oversampling to handle class imbalance.
    
    Args:
        X_text: List of text samples
        X_numerical: Numerical features
        y: Labels
        
    Returns:
        Oversampled data
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        # Combine numerical features for SMOTE
        # Note: SMOTE works on numerical features, not text
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
        X_numerical_resampled, y_resampled = smote.fit_resample(X_numerical, y)
        
        # For text, we'll duplicate samples (not ideal but necessary)
        # In practice, you'd want more sophisticated text augmentation
        X_text_resampled = []
        for i, label in enumerate(y_resampled):
            if i < len(X_text):
                X_text_resampled.append(X_text[i])
            else:
                # Find a sample with the same label to duplicate
                same_label_indices = np.where(y == label)[0]
                if len(same_label_indices) > 0:
                    idx = np.random.choice(same_label_indices)
                    X_text_resampled.append(X_text[idx])
                else:
                    X_text_resampled.append(X_text[0])  # Fallback
        
        print(f"SMOTE applied: {len(y)} -> {len(y_resampled)} samples")
        return X_text_resampled, X_numerical_resampled, y_resampled
        
    except ImportError:
        print("SMOTE not available. Install imbalanced-learn: pip install imbalanced-learn")
        return X_text, X_numerical, y

# --- 2. Custom Dataset Class ---

class CustomDataset(Dataset):
    """Handles tokenizing text and extracting numerical features and labels."""
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 256):
        self.texts = df['comment_text'].values
        
        # Get all numerical features dynamically
        numerical_cols = [col for col in df.columns 
                         if col not in ['comment_text', 'rule_violation', 'subreddit', 'rule'] 
                         and str(df.dtypes[col]) in ['int64', 'float64']]
        
        self.numerical = torch.tensor(df[numerical_cols].values, dtype=torch.float32)
        
        # Handle the actual label column 'rule_violation' from the CSV
        if 'rule_violation' in df.columns:
            # Convert labels to float32 for compatibility with loss functions
            self.labels = torch.tensor(df['rule_violation'].values, dtype=torch.float32).unsqueeze(1)
        else:
            # Fallback to rule_ columns if they exist
            self.labels = torch.tensor(df.filter(regex='rule_').values, dtype=torch.float32)
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Ensure text is a string and handle NaN/None values
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

# --- 3. Evaluation Function (For AUC Metric) ---

def find_optimal_threshold(y_true, y_scores):
    """Find the optimal threshold for the given true labels and predicted scores."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]

def evaluate_model(model, dataloader, device=DEVICE, find_optimal_threshold_flag=False) -> Tuple[float, np.ndarray, float]:
    """Calculates Column-Averaged AUC and individual AUCs on the validation set."""
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(batch['labels'].cpu().numpy())
    
    # Concatenate all results
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    optimal_threshold = 0.5
    if find_optimal_threshold_flag:
        optimal_threshold = find_optimal_threshold(labels, probs)
        print(f"Optimal threshold: {optimal_threshold}")
    
    auc_scores = []
    for i in range(NUM_RULES):
        try:
            # roc_auc_score requires at least one positive and one negative sample
            score = roc_auc_score(labels[:, i], probs[:, i])
            auc_scores.append(score)
        except ValueError:
            # Handle cases where a rule column is constant (very rare or no positive samples)
            auc_scores.append(0.5) 

    column_averaged_auc = np.mean(auc_scores)
    return column_averaged_auc, np.array(auc_scores), optimal_threshold

def predict_with_optimal_threshold(model, dataloader, device=DEVICE, threshold=None):
    """
    Make predictions using optimal threshold instead of default 0.5
    """
    model.eval()
    all_predictions = []
    
    # Use provided threshold or model's stored threshold
    if threshold is None:
        threshold = getattr(model, 'optimal_threshold', 0.5)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions = (probs > threshold).astype(int)
            
            all_predictions.append(predictions)
    
    return np.concatenate(all_predictions)

# --- 4. Main Training Function ---

def train_model():
    
    # --- Data Loading and Splitting ---
    print("--- 1. Data Loading and Splitting ---")
    
    try:
        print(f"Loading data from: {TRAIN_FILE_PATH}")
        full_train_df = pd.read_csv(TRAIN_FILE_PATH)
        print(f"Data loaded successfully. Shape: {full_train_df.shape}")
        print(f"Columns: {list(full_train_df.columns)}")
    except FileNotFoundError:
        print(f"FATAL ERROR: '{TRAIN_FILE_PATH}' not found. Please check file path.")
        return
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return 
    
    # Identify label columns for stratification
    if 'rule_violation' in full_train_df.columns:
        label_columns = ['rule_violation']
        stratify_col = full_train_df['rule_violation']
    else:
        label_columns = full_train_df.filter(regex='rule_').columns.tolist()
        if not label_columns:
            print("FATAL ERROR: No label columns found in train.csv.")
            return
        # Create a column for stratification (sum of all rule violations)
        stratify_col = full_train_df[label_columns].sum(axis=1) 

    # Split the full training data into model training and validation sets
    train_df_raw, validation_df_raw = train_test_split(
        full_train_df,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_col 
    )

    print(f"Dataset split: Train={len(train_df_raw)} samples, Validation={len(validation_df_raw)} samples")

    # --- Preprocessing ---
    # NOTE: You MUST update your preprocess.py to accept the df_to_process argument (as detailed previously)

    # A. Process the TRAINING set (Fits all scalers/models: TFIDF, mean vectors, MinMaxScaler)
    print("\nProcessing TRAINING data (Fitting models)...")
    # We pass df_to_process=train_df_raw and file_path=None to signal using the DataFrame
    train_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
        file_path=None, 
        df_to_process=train_df_raw,
        enable_spacy=ENABLE_SPACY_FEATURES
    )
    
    # Update NUM_NUMERICAL_FEATURES based on actual processed data
    global NUM_NUMERICAL_FEATURES
    actual_numerical_features = [col for col in train_df_processed.columns 
                                if col not in ['comment_text', 'rule_violation', 'subreddit', 'rule'] 
                                and str(train_df_processed.dtypes[col]) in ['int64', 'float64']]
    NUM_NUMERICAL_FEATURES = len(actual_numerical_features)
    print(f"Updated NUM_NUMERICAL_FEATURES to {NUM_NUMERICAL_FEATURES} based on processed data")

    # B. Process the VALIDATION set (Uses fitted models to TRANSFORM only)
    print("Processing VALIDATION data (Transforming only)...")
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None, 
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler,
        enable_spacy=ENABLE_SPACY_FEATURES
    )
    
    # Check for empty dataframes after preprocessing (in case of a processing error)
    if train_df_processed.empty or validation_df_processed.empty:
        print("FATAL ERROR: DataFrames are empty after preprocessing. Check your 'preprocess.py'.")
        return

    # --- Setup DataLoader ---
    print("\nSetting up tokenizer and datasets...")
    try:
        print(f"Loading tokenizer: {TRANSFORMER_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"ERROR loading tokenizer: {e}")
        print("This might be a network issue. Please check your internet connection.")
        return
    
    print("Creating training dataset...")
    train_dataset = CustomDataset(train_df_processed, tokenizer, MAX_SEQ_LENGTH)
    print(f"Training dataset created with {len(train_dataset)} samples")
    
    print("Creating training dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Training dataloader created with {len(train_dataloader)} batches")
    
    print("Creating validation dataset...")
    validation_dataset = CustomDataset(validation_df_processed, tokenizer, MAX_SEQ_LENGTH)
    print(f"Validation dataset created with {len(validation_dataset)} samples")
    
    print("Creating validation dataloader...")
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Validation dataloader created with {len(validation_dataloader)} batches")

    # --- Model and Loss Initialization ---
    print("\n--- 2. Model and Loss Initialization ---")
    print(f"Initializing model on device: {DEVICE}")
    
    # Clear GPU cache if using CUDA
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory before model creation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    model = CustomTransformerModel(
        transformer_name=TRANSFORMER_MODEL_NAME, 
        num_numerical_features=NUM_NUMERICAL_FEATURES, 
        num_rules=1
    ).to(DEVICE)
    
    if DEVICE.type == 'cuda':
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # Initialize loss function based on configuration
    if USE_ADVANCED_LOSS:
        print("Using Combined Advanced Loss Function")
        criterion = CombinedAdvancedLoss(
            focal_alpha=FOCAL_LOSS_ALPHA,
            focal_gamma=FOCAL_LOSS_GAMMA,
            label_smoothing=LABEL_SMOOTHING_FACTOR,
            weight_decay=WEIGHT_DECAY,
            use_mixup=USE_MIXUP_AUGMENTATION,
            mixup_alpha=MIXUP_ALPHA
        )
    else:
        print("Using Custom Cost Sensitive Loss Function")
        criterion = CustomCostSensitiveLoss(
            rule_weights=RULE_WEIGHTS, 
            feature_weights=FEATURE_WEIGHTS
        )

    # Enhanced optimizer with better parameters
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=True
    )

    # Learning rate scheduler
    scheduler = None
    if USE_LEARNING_RATE_SCHEDULING:
        total_steps = len(train_dataloader) * NUM_EPOCHS
        warmup_steps = int(total_steps * WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        print(f"Learning rate scheduling enabled: {warmup_steps} warmup steps")

    # Mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    if USE_MIXED_PRECISION:
        print("Mixed precision training enabled")

    # Early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE) if USE_EARLY_STOPPING else None
    if USE_EARLY_STOPPING:
        print(f"Early stopping enabled with patience: {EARLY_STOPPING_PATIENCE}")

    # --- Training Loop ---
    print(f"\n--- 3. Starting Enhanced Training on {DEVICE} ---")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of batches per epoch: {len(train_dataloader)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient clipping: {'Enabled' if USE_GRADIENT_CLIPPING else 'Disabled'}")
    
    best_auc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nStarting Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch data to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Ensure labels are float32 for loss computation
            labels = labels.float()

            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask, numerical_features)
                    # Calculate loss based on loss function type
                    if USE_ADVANCED_LOSS:
                        loss = criterion(logits, labels, model=model)
                    else:
                        loss = criterion(logits, labels, numerical_features)
                
                # Backward pass with mixed precision
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if USE_GRADIENT_CLIPPING:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                logits = model(input_ids, attention_mask, numerical_features)
                try:
                        # Calculate loss based on loss function type
                    if USE_ADVANCED_LOSS:
                        loss = criterion(logits, labels, model=model)
                    else:
                loss = criterion(logits, labels, numerical_features)
            except Exception as e:
                print(f"ERROR in loss calculation at batch {batch_idx}: {e}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}, Features shape: {numerical_features.shape}")
                raise e
            
                # Standard backward pass
            loss.backward()
                
                # Gradient clipping
                if USE_GRADIENT_CLIPPING:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
                
                # Optimizer step
                optimizer.step()
            
            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # More frequent updates
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            # Add a safety check to prevent infinite loops
            if batch_idx > len(train_dataloader) * 2:  # Safety check
                print(f"WARNING: Batch index {batch_idx} exceeds expected range. Breaking loop.")
                break

        avg_loss = total_loss / len(train_dataloader)
        
        # --- Validation and AUC Check ---
        validation_auc, auc_per_rule, optimal_threshold = evaluate_model(model, validation_dataloader, DEVICE, find_optimal_threshold_flag=True)
        
        # Store optimal threshold in model for later use
        model.optimal_threshold = optimal_threshold

        print(f"\n--- Epoch {epoch+1} Complete ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Validation Column-Averaged AUC: {validation_auc:.4f}")
        print(f"AUC Per Rule: {np.round(auc_per_rule, 4)}") # Display per-rule performance
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        
        # Save best model for error analysis
        if validation_auc > best_auc:
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! AUC: {validation_auc:.4f}")
            best_auc = validation_auc
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(validation_auc):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best AUC achieved: {early_stopping.best_auc:.4f}")
                break
    
    print(f"\nTraining complete. Best validation AUC: {best_auc:.4f}")

# --- Execute Script ---
if __name__ == '__main__':
    train_model()
