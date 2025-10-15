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

# Suppress warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocess import preprocess_data
from custom_model import CustomTransformerModel
from custom_loss import (CustomCostSensitiveLoss, CombinedAdvancedLoss)

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
    
    print("Processing VALIDATION data...")
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None, 
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler,
        enable_spacy=ENABLE_SPACY_FEATURES
    )
    
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
                'vocab_size': VOCAB_SIZE
            }, 'training_components.pth')
            print(f"✅ New best model saved! AUC: {validation_auc:.4f}")
            best_auc = validation_auc
    
    print(f"\nTraining complete. Best validation AUC: {best_auc:.4f}")

# --- Execute Script ---
if __name__ == '__main__':
    train_model()