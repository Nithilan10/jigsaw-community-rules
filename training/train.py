# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Dict, Any, Tuple

# Import custom modules
# NOTE: The import below assumes your data_preprocessing file is named 'preprocess.py'
from preprocess import preprocess_data
from custom_model import CustomTransformerModel
from custom_loss import CustomCostSensitiveLoss

# --- 0. Configuration and Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORMER_MODEL_NAME = 'bert-base-uncased'
TRAIN_FILE_PATH = '../data/train.csv' # Use the actual file name
NUM_RULES = 1  # IMPORTANT: Set this to your actual number of policy columns
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 256
VALIDATION_SPLIT_RATIO = 0.15
RANDOM_SEED = 42

# This list must exactly match the order of features created in data_preprocessing.py
NUMERICAL_FEATURES = [
    'comment_length', 'exclamation_frequency', 
    'legal_advice_interaction_feature', 'promo_persuasion_feature', 
    'similarity_to_violation', 'similarity_to_safe'
]
NUM_NUMERICAL_FEATURES = len(NUMERICAL_FEATURES)

# Custom Loss Weights (The cost-sensitive prioritization logic)
RULE_WEIGHTS = {
    2: 7.0,  # Example: Rule 3 (index 2) is highest priority
    0: 3.0,  # Example: Rule 1 (index 0) is medium priority
} 
FEATURE_WEIGHTS = {
    'legal_advice_interaction_feature': 8.0,
    'promo_persuasion_feature': 9.0,
    'comment_length_short': 5.0 
}

# --- 1. Custom Dataset Class (No changes needed here) ---

class ToxicityDataset(Dataset):
    """Handles tokenizing text and extracting numerical features and labels."""
    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer):
        self.texts = df['comment_text'].values
        self.numerical = torch.tensor(df[NUMERICAL_FEATURES].values, dtype=torch.float32)
        # Handle the actual label column 'rule_violation' from the CSV
        if 'rule_violation' in df.columns:
            self.labels = torch.tensor(df['rule_violation'].values, dtype=torch.long).unsqueeze(1)
        else:
            # Fallback to rule_ columns if they exist
            self.labels = torch.tensor(df.filter(regex='rule_').values, dtype=torch.long)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_SEQ_LENGTH,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'numerical_features': self.numerical[idx],
            'labels': self.labels[idx]
        }


# --- 2. Evaluation Function (For AUC Metric) ---

def evaluate_model(model, dataloader, device=DEVICE) -> Tuple[float, np.ndarray]:
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
    return column_averaged_auc, np.array(auc_scores)


# --- 3. Main Training Function ---

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
        df_to_process=train_df_raw
    )

    # B. Process the VALIDATION set (Uses fitted models to TRANSFORM only)
    print("Processing VALIDATION data (Transforming only)...")
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None, 
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler
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
    train_dataset = ToxicityDataset(train_df_processed, tokenizer)
    print(f"Training dataset created with {len(train_dataset)} samples")
    
    print("Creating training dataloader...")
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    print(f"Training dataloader created with {len(train_dataloader)} batches")
    
    print("Creating validation dataset...")
    validation_dataset = ToxicityDataset(validation_df_processed, tokenizer)
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
        num_rules=NUM_RULES
    ).to(DEVICE)
    
    if DEVICE.type == 'cuda':
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    criterion = CustomCostSensitiveLoss(
        rule_weights=RULE_WEIGHTS, 
        feature_weights=FEATURE_WEIGHTS
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    print(f"\n--- 3. Starting Training on {DEVICE} ---")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Number of batches per epoch: {len(train_dataloader)}")
    print(f"Batch size: {BATCH_SIZE}")
    
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

            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            
            # Calculate the Custom Weighted Loss
            try:
                loss = criterion(logits, labels, numerical_features)
            except Exception as e:
                print(f"ERROR in loss calculation at batch {batch_idx}: {e}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}, Features shape: {numerical_features.shape}")
                raise e
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:  # More frequent updates
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
            
            # Add a safety check to prevent infinite loops
            if batch_idx > len(train_dataloader) * 2:  # Safety check
                print(f"WARNING: Batch index {batch_idx} exceeds expected range. Breaking loop.")
                break

        avg_loss = total_loss / len(train_dataloader)
        
        # --- Validation and AUC Check ---
        validation_auc, auc_per_rule = evaluate_model(model, validation_dataloader, DEVICE)

        print(f"\n--- Epoch {epoch+1} Complete ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Validation Column-Averaged AUC: {validation_auc:.4f}")
        print(f"AUC Per Rule: {np.round(auc_per_rule, 4)}") # Display per-rule performance
        
        # NOTE: At this point, you would typically save the model if validation_auc improved.

# --- Execute Script ---
if __name__ == '__main__':
    train_model()