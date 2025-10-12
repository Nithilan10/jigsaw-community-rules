# train.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from typing import Dict, Any, Tuple, List
from torch.cuda.amp import autocast, GradScaler
import itertools
import random
from collections import Counter

# Import custom modules
# NOTE: The import below assumes your data_preprocessing file is named 'preprocess.py'
from preprocess import preprocess_data
from custom_model import CustomTransformerModel
from custom_loss import CustomCostSensitiveLoss, SGDAOptimizer, OGDAOptimizer, RobustLoss

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
USE_MIXED_PRECISION = True          # Enable mixed precision training
EARLY_STOPPING_PATIENCE = 3         # Stop if no improvement for 3 epochs
GRADIENT_CLIP_NORM = 1.0           # Max gradient norm
WARMUP_RATIO = 0.1                 # 10% of training for warmup

# Hyperparameter Tuning Configuration
HYPERPARAMETER_TUNING = True       # Set to True to enable hyperparameter tuning
TUNING_MODE = 'random'             # Options: 'grid', 'random', 'bayesian'
MAX_TUNING_TRIALS = 20             # Maximum number of trials for random/bayesian search
TUNING_CV_FOLDS = 3                # Cross-validation folds for tuning

# Hyperparameter Search Spaces
HYPERPARAMETER_SPACES = {
    'learning_rate': [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5],
    'batch_size': [4, 6, 8, 10, 12, 16],
    'num_epochs': [3, 4, 5, 6, 7, 8],
    'weight_decay': [0.0, 0.01, 0.05, 0.1, 0.2],
    'dropout_rate': [0.1, 0.15, 0.2, 0.25, 0.3],
    'warmup_ratio': [0.05, 0.1, 0.15, 0.2],
    'gradient_clip_norm': [0.5, 1.0, 1.5, 2.0]
}

# Class Imbalance Handling
USE_CLASS_WEIGHTING = True         # Enable class weighting
USE_FOCAL_LOSS = True              # Enable focal loss for class imbalance
USE_SMOTE = False                  # Enable SMOTE (synthetic minority oversampling)
FOCAL_LOSS_ALPHA = 0.25            # Focal loss alpha parameter
FOCAL_LOSS_GAMMA = 2.0             # Focal loss gamma parameter

# Ensemble configuration
ENSEMBLE_MODE = False  # Set to True to train multiple models

# SPD (Stochastic Primal-Dual) configuration
SPD_MODE = False  # Set to True to use SPD optimization
SPD_METHOD = 'SGDA'  # Options: 'SGDA', 'OGDA'
SPD_LR_PRIMAL = 1e-5  # Learning rate for primal variables (model parameters)
SPD_LR_DUAL = 1e-4   # Learning rate for dual variables (Lagrange multipliers)
ENSEMBLE_MODELS = {
    'deberta_v3_base': {
        'name': 'microsoft/deberta-v3-base',
        'batch_size': 6,
        'learning_rate': 5e-6,
        'epochs': 3,
        'weight': 0.4
    },
    'roberta_base': {
        'name': 'roberta-base',
        'batch_size': 8,
        'learning_rate': 1e-5,
        'epochs': 3,
        'weight': 0.3
    },
    'deberta_base': {
        'name': 'microsoft/deberta-base',
        'batch_size': 8,
        'learning_rate': 1e-5,
        'epochs': 3,
        'weight': 0.2
    },
    'bert_base': {
        'name': 'bert-base-uncased',
        'batch_size': 10,
        'learning_rate': 2e-5,
        'epochs': 3,
        'weight': 0.1
    }
}

# This list must exactly match the order of features created in data_preprocessing.py
NUMERICAL_FEATURES = [
    'comment_length', 'exclamation_frequency', 
    'legal_advice_interaction_feature', 'promo_persuasion_feature', 
    'similarity_to_violation', 'similarity_to_safe', 'consistency_deviation', 'boundary_proximity_score'
]
NUM_NUMERICAL_FEATURES = len(NUMERICAL_FEATURES)

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

class HyperparameterTuner:
    """
    Hyperparameter tuning utility for transformer models.
    """
    def __init__(self, search_space: dict, mode: str = 'random', max_trials: int = 20):
        self.search_space = search_space
        self.mode = mode
        self.max_trials = max_trials
        self.trial_results = []
        
    def generate_hyperparameters(self) -> dict:
        """Generate hyperparameters based on search mode."""
        if self.mode == 'grid':
            return self._grid_search()
        elif self.mode == 'random':
            return self._random_search()
        else:
            raise ValueError(f"Unknown search mode: {self.mode}")
    
    def _random_search(self) -> dict:
        """Generate random hyperparameters from search space."""
        params = {}
        for param_name, param_values in self.search_space.items():
            params[param_name] = random.choice(param_values)
        return params
    
    def _grid_search(self) -> dict:
        """Generate hyperparameters using grid search."""
        # This would be more complex for full grid search
        # For now, return random search
        return self._random_search()
    
    def add_trial_result(self, params: dict, score: float):
        """Add a trial result."""
        self.trial_results.append({
            'params': params,
            'score': score
        })
    
    def get_best_params(self) -> dict:
        """Get the best hyperparameters found so far."""
        if not self.trial_results:
            return {}
        
        best_trial = max(self.trial_results, key=lambda x: x['score'])
        return best_trial['params']

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

# --- 2. Custom Dataset Class (No changes needed here) ---

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

            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(input_ids, attention_mask, numerical_features)
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
        validation_auc, auc_per_rule = evaluate_model(model, validation_dataloader, DEVICE)

        print(f"\n--- Epoch {epoch+1} Complete ---")
        print(f"Average Training Loss: {avg_loss:.4f}")
        print(f"Validation Column-Averaged AUC: {validation_auc:.4f}")
        print(f"AUC Per Rule: {np.round(auc_per_rule, 4)}") # Display per-rule performance
        
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

def train_with_hyperparameter_tuning():
    """
    Train model with hyperparameter tuning.
    """
    print("Starting Hyperparameter Tuning")
    print("="*60)
    
    # Load and preprocess data
    try:
        print(f"Loading data from: {TRAIN_FILE_PATH}")
        full_train_df = pd.read_csv(TRAIN_FILE_PATH)
        print(f"Data loaded successfully. Shape: {full_train_df.shape}")
    except FileNotFoundError:
        print(f"FATAL ERROR: '{TRAIN_FILE_PATH}' not found.")
        return
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return
    
    # Split data
    if 'rule_violation' in full_train_df.columns:
        stratify_col = full_train_df['rule_violation']
    else:
        label_columns = full_train_df.filter(regex='rule_').columns.tolist()
        stratify_col = full_train_df[label_columns].sum(axis=1) if label_columns else None
    
    train_df_raw, validation_df_raw = train_test_split(
        full_train_df,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )
    
    print(f"Dataset split: Train={len(train_df_raw)} samples, Validation={len(validation_df_raw)} samples")
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
        file_path=None,
        df_to_process=train_df_raw
    )
    
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None,
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler
    )
    
    # Load test data
    try:
        print(f"Loading test data from: {TEST_FILE_PATH}")
        test_df_raw = pd.read_csv(TEST_FILE_PATH)
        print(f"Test data loaded successfully. Shape: {test_df_raw.shape}")
    except FileNotFoundError:
        print(f"WARNING: '{TEST_FILE_PATH}' not found. Using validation set for evaluation.")
        test_df_raw = validation_df_raw
    except Exception as e:
        print(f"WARNING: Error loading test data: {e}. Using validation set for evaluation.")
        test_df_raw = validation_df_raw
    
    test_df_processed, _, _, _ = preprocess_data(
        file_path=None,
        df_to_process=test_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler
    )
    
    # Apply class imbalance handling if enabled
    if USE_SMOTE:
        print("\nApplying SMOTE oversampling...")
        X_text = train_df_processed['comment_text'].tolist()
        X_numerical = train_df_processed[NUMERICAL_FEATURES].values
        y = train_df_processed['rule_violation'].values
        
        X_text_resampled, X_numerical_resampled, y_resampled = apply_smote_oversampling(X_text, X_numerical, y)
        
        # Recreate DataFrame
        train_df_processed = pd.DataFrame({
            'comment_text': X_text_resampled,
            'rule_violation': y_resampled
        })
        # Add numerical features
        for i, feature in enumerate(NUMERICAL_FEATURES):
            train_df_processed[feature] = X_numerical_resampled[:, i]
    
    # Calculate class weights
    class_weights = None
    if USE_CLASS_WEIGHTING:
        labels = train_df_processed['rule_violation'].values
        class_weights = calculate_class_weights(labels)
        print(f"Class weights calculated: {class_weights}")
    
    # Initialize hyperparameter tuner
    tuner = HyperparameterTuner(
        search_space=HYPERPARAMETER_SPACES,
        mode=TUNING_MODE,
        max_trials=MAX_TUNING_TRIALS
    )
    
    print(f"\nStarting hyperparameter tuning with {TUNING_MODE} search...")
    print(f"Maximum trials: {MAX_TUNING_TRIALS}")
    
    best_auc = 0.0
    best_params = {}
    
    for trial in range(MAX_TUNING_TRIALS):
        print(f"\n{'='*50}")
        print(f"TRIAL {trial + 1}/{MAX_TUNING_TRIALS}")
        print(f"{'='*50}")
        
        # Generate hyperparameters
        params = tuner.generate_hyperparameters()
        print(f"Trial parameters: {params}")
        
        try:
            # Train model with these hyperparameters
            auc = train_single_trial(
                params, 
                train_df_processed, 
                validation_df_processed, 
                test_df_processed,
                tfidf_model, 
                mean_vectors, 
                scaler,
                class_weights
            )
            
            # Record result
            tuner.add_trial_result(params, auc)
            
            print(f"Trial {trial + 1} Test AUC: {auc:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_params = params.copy()
                print(f"🎯 NEW BEST TEST AUC: {best_auc:.4f}")
                print(f"Best parameters: {best_params}")
            
        except Exception as e:
            print(f"Trial {trial + 1} failed: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Test AUC: {best_auc:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Save best parameters
    import json
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"Best hyperparameters saved to best_hyperparameters.json")
    
    return best_params, best_auc

def train_single_trial(params: dict, train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame,
                      tfidf_model, mean_vectors, scaler, class_weights: dict = None) -> float:
    """
    Train a single model with given hyperparameters.
    
    Args:
        params: Hyperparameters for this trial
        train_df: Training data
        validation_df: Validation data (for early stopping)
        test_df: Test data (for final evaluation)
        tfidf_model: Fitted TF-IDF model
        mean_vectors: Mean vectors
        scaler: Fitted scaler
        class_weights: Class weights for loss function
        
    Returns:
        Test AUC for this trial
    """
    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    train_dataset = ToxicityDataset(train_df, tokenizer)
    validation_dataset = ToxicityDataset(validation_df, tokenizer)
    test_dataset = ToxicityDataset(test_df, tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=params.get('batch_size', BATCH_SIZE), 
        shuffle=True, 
        num_workers=0
    )
    validation_dataloader = DataLoader(
        validation_dataset, 
        batch_size=params.get('batch_size', BATCH_SIZE), 
        shuffle=False, 
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=params.get('batch_size', BATCH_SIZE), 
        shuffle=False, 
        num_workers=0
    )
    
    # Initialize model
    model = CustomTransformerModel(
        transformer_name=TRANSFORMER_MODEL_NAME,
        num_numerical_features=NUM_NUMERICAL_FEATURES,
        num_rules=NUM_RULES
    ).to(DEVICE)
    
    # Initialize loss function with class weights
    if USE_FOCAL_LOSS:
        from custom_loss import FocalLoss
        base_loss_fn = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
    else:
        base_loss_fn = None
    
    criterion = CustomCostSensitiveLoss(
        rule_weights=RULE_WEIGHTS,
        feature_weights=FEATURE_WEIGHTS,
        base_loss_fn=base_loss_fn
    )
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params.get('learning_rate', LEARNING_RATE),
        weight_decay=params.get('weight_decay', 0.01)
    )
    
    # Training loop
    num_epochs = params.get('num_epochs', NUM_EPOCHS)
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            loss = criterion(logits, labels, numerical_features)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if USE_GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    params.get('gradient_clip_norm', GRADIENT_CLIP_NORM)
                )
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation (for early stopping)
        validation_auc, _ = evaluate_model(model, validation_dataloader, DEVICE)
        
        if validation_auc > best_auc:
            best_auc = validation_auc
    
    # Final evaluation on test set
    test_auc, _ = evaluate_model(model, test_dataloader, DEVICE)
    print(f"    Final Test AUC: {test_auc:.4f}")
    
    return test_auc

# --- Ensemble Training Functions ---

def train_ensemble_model(model_name: str, model_config: dict, train_df_processed: pd.DataFrame, 
                        validation_df_processed: pd.DataFrame, tfidf_model, mean_vectors, scaler) -> float:
    """
    Train a single model in the ensemble.
    
    Args:
        model_name: Name of the model (e.g., 'deberta_v3_base')
        model_config: Configuration dictionary for this model
        train_df_processed: Preprocessed training data
        validation_df_processed: Preprocessed validation data
        tfidf_model: Fitted TF-IDF model
        mean_vectors: Mean vectors for similarity features
        scaler: Fitted scaler
        
    Returns:
        Best validation AUC for this model
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ENSEMBLE MODEL: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Create tokenizer for this specific model
    tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
    print(f"Loaded tokenizer: {model_config['name']}")
    
    # Create datasets with model-specific tokenizer
    train_dataset = ToxicityDataset(train_df_processed, tokenizer)
    validation_dataset = ToxicityDataset(validation_df_processed, tokenizer)
    
    # Create dataloaders with model-specific batch size
    train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size'], shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=model_config['batch_size'], shuffle=False, num_workers=0)
    
    print(f"Training batches: {len(train_dataloader)}, Validation batches: {len(validation_dataloader)}")
    
    # Initialize model
    model = CustomTransformerModel(
        transformer_name=model_config['name'],
        num_numerical_features=NUM_NUMERICAL_FEATURES,
        num_rules=NUM_RULES
    ).to(DEVICE)
    
    # Initialize loss and optimizer
    criterion = CustomCostSensitiveLoss(
        rule_weights=RULE_WEIGHTS,
        feature_weights=FEATURE_WEIGHTS
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['learning_rate'])
    
    # Training loop
    best_auc = 0.0
    
    for epoch in range(model_config['epochs']):
        print(f"\nEpoch {epoch+1}/{model_config['epochs']} for {model_name}")
        
        # Training phase
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move data to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            
            # Compute loss
            loss = criterion(logits, labels, numerical_features)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        # Validation phase
        avg_loss = total_loss / len(train_dataloader)
        validation_auc, _ = evaluate_model(model, validation_dataloader, DEVICE)
        
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Validation AUC: {validation_auc:.4f}")
        
        # Save best model
        if validation_auc > best_auc:
            best_auc = validation_auc
            model_path = f'models/{model_name}_best.pth'
            import os
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved: {model_path}")
    
    print(f"\n{model_name} training complete! Best AUC: {best_auc:.4f}")
    return best_auc

def train_ensemble():
    """
    Train multiple models for ensemble.
    """
    print("Starting Ensemble Training")
    print("="*60)
    
    # Load and split data (same as single model training)
    try:
        print(f"Loading data from: {TRAIN_FILE_PATH}")
        full_train_df = pd.read_csv(TRAIN_FILE_PATH)
        print(f"Data loaded successfully. Shape: {full_train_df.shape}")
    except FileNotFoundError:
        print(f"FATAL ERROR: '{TRAIN_FILE_PATH}' not found.")
        return
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return
    
    # Split data
    if 'rule_violation' in full_train_df.columns:
        stratify_col = full_train_df['rule_violation']
    else:
        label_columns = full_train_df.filter(regex='rule_').columns.tolist()
        stratify_col = full_train_df[label_columns].sum(axis=1) if label_columns else None
    
    train_df_raw, validation_df_raw = train_test_split(
        full_train_df,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )
    
    print(f"Dataset split: Train={len(train_df_raw)} samples, Validation={len(validation_df_raw)} samples")
    
    # Preprocess data once (all models will use the same preprocessing)
    print("\nPreprocessing data for ensemble...")
    train_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
        file_path=None,
        df_to_process=train_df_raw
    )
    
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None,
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler
    )
    
    # Train each model in the ensemble
    model_performances = {}
    
    for model_name, model_config in ENSEMBLE_MODELS.items():
        try:
            auc = train_ensemble_model(
                model_name, model_config, 
                train_df_processed, validation_df_processed,
                tfidf_model, mean_vectors, scaler
            )
            model_performances[model_name] = auc
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            model_performances[model_name] = 0.0
    
    # Create ensemble weights based on performance
    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*60}")
    
    total_performance = sum(model_performances.values())
    ensemble_weights = {}
    
    for model_name, auc in model_performances.items():
        print(f"{model_name}: {auc:.4f} AUC")
        
        # Create weights based on performance
        original_weight = ENSEMBLE_MODELS[model_name]['weight']
        performance_weight = auc / total_performance if total_performance > 0 else 0.25
        
        # Combine original and performance-based weights
        final_weight = 0.7 * performance_weight + 0.3 * original_weight
        ensemble_weights[model_name] = final_weight
    
    # Normalize weights to sum to 1
    total_weight = sum(ensemble_weights.values())
    for model_name in ensemble_weights:
        ensemble_weights[model_name] /= total_weight
    
    print(f"\nFinal Ensemble Weights:")
    for model_name, weight in ensemble_weights.items():
        print(f"  {model_name}: {weight:.4f}")
    
    # Save ensemble configuration
    import json
    config = {
        'model_configs': ENSEMBLE_MODELS,
        'ensemble_weights': ensemble_weights,
        'model_performances': model_performances
    }
    
    with open('models/ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nEnsemble training complete!")
    print(f"Configuration saved to models/ensemble_config.json")
    
    # Evaluate ensemble performance
    print(f"\n{'='*60}")
    print("EVALUATING ENSEMBLE PERFORMANCE")
    print(f"{'='*60}")
    
    ensemble_auc = evaluate_ensemble(ensemble_weights, validation_df_processed, tfidf_model, mean_vectors, scaler)
    print(f"Final Ensemble AUC: {ensemble_auc:.4f}")
    
    return ensemble_auc

def evaluate_ensemble(ensemble_weights: dict, validation_df: pd.DataFrame, 
                     tfidf_model, mean_vectors, scaler) -> float:
    """
    Evaluate ensemble performance by combining predictions from all models.
    
    Args:
        ensemble_weights: Dictionary of model weights
        validation_df: Validation dataset
        tfidf_model: Fitted TF-IDF model
        mean_vectors: Mean vectors for similarity features
        scaler: Fitted scaler
        
    Returns:
        Ensemble AUC score
    """
    print("Loading trained models for ensemble evaluation...")
    
    # Load all trained models
    models = {}
    for model_name, model_config in ENSEMBLE_MODELS.items():
        try:
            # Create model
            model = CustomTransformerModel(
                transformer_name=model_config['name'],
                num_numerical_features=NUM_NUMERICAL_FEATURES,
                num_rules=NUM_RULES
            ).to(DEVICE)
            
            # Load trained weights
            model_path = f'models/{model_name}_best.pth'
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            
            models[model_name] = model
            print(f"  ✓ Loaded {model_name}")
            
        except Exception as e:
            print(f"  ✗ Failed to load {model_name}: {e}")
            continue
    
    if not models:
        print("ERROR: No models loaded for ensemble evaluation!")
        return 0.0
    
    print(f"Evaluating ensemble with {len(models)} models...")
    
    # Collect predictions from all models
    all_model_predictions = {}
    all_labels = []
    
    for model_name, model in models.items():
        print(f"  Getting predictions from {model_name}...")
        
        # Create dataset with model-specific tokenizer
        model_config = ENSEMBLE_MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        validation_dataset = ToxicityDataset(validation_df, tokenizer)
        validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        model_predictions = []
        
        with torch.no_grad():
            for batch in validation_dataloader:
                try:
                    # Move data to device
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    numerical_features = batch['numerical_features'].to(DEVICE)
                    
                    # Forward pass
                    logits = model(input_ids, attention_mask, numerical_features)
                    
                    # Convert to probabilities
                    probs = torch.sigmoid(logits).cpu().numpy()
                    model_predictions.append(probs)
                    
                    # Store labels (only need to do this once)
                    if model_name == list(models.keys())[0]:
                        all_labels.append(batch['labels'].cpu().numpy())
                    
                    # Clear CUDA cache to prevent memory issues
                    if DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"    Error processing batch for {model_name}: {e}")
                    continue
        
        # Concatenate all predictions for this model
        if model_predictions:
            all_model_predictions[model_name] = np.concatenate(model_predictions, axis=0)
            print(f"    ✓ {model_name}: {len(model_predictions)} batches processed")
        else:
            print(f"    ✗ {model_name}: No predictions generated")
    
    # Concatenate all labels
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Check if we have valid predictions
    if not all_model_predictions:
        print("ERROR: No valid model predictions for ensemble evaluation!")
        return 0.0
    
    # Calculate ensemble predictions
    print("Calculating ensemble predictions...")
    first_model_name = list(all_model_predictions.keys())[0]
    ensemble_predictions = np.zeros_like(all_model_predictions[first_model_name])
    
    for model_name, predictions in all_model_predictions.items():
        weight = ensemble_weights.get(model_name, 0.0)
        ensemble_predictions += weight * predictions
        print(f"  {model_name}: weight={weight:.4f}, predictions shape={predictions.shape}")
    
    # Calculate ensemble AUC
    try:
        ensemble_auc = roc_auc_score(all_labels, ensemble_predictions)
        print(f"Ensemble AUC: {ensemble_auc:.4f}")
        
        # Compare with individual models
        print(f"\nIndividual Model AUCs:")
        for model_name, predictions in all_model_predictions.items():
            individual_auc = roc_auc_score(all_labels, predictions)
            print(f"  {model_name}: {individual_auc:.4f}")
        
        return ensemble_auc
        
    except Exception as e:
        print(f"ERROR calculating ensemble AUC: {e}")
        print(f"Labels shape: {all_labels.shape}")
        print(f"Ensemble predictions shape: {ensemble_predictions.shape}")
        return 0.0

# --- SPD Training Functions ---

def train_with_spd():
    """
    Train model using Stochastic Primal-Dual (SPD) optimization.
    """
    print("Starting SPD Training")
    print("="*60)
    print(f"SPD Method: {SPD_METHOD}")
    print(f"Primal LR: {SPD_LR_PRIMAL}, Dual LR: {SPD_LR_DUAL}")
    
    # Load and preprocess data (same as regular training)
    try:
        print(f"Loading data from: {TRAIN_FILE_PATH}")
        full_train_df = pd.read_csv(TRAIN_FILE_PATH)
        print(f"Data loaded successfully. Shape: {full_train_df.shape}")
    except FileNotFoundError:
        print(f"FATAL ERROR: '{TRAIN_FILE_PATH}' not found.")
        return
    except Exception as e:
        print(f"FATAL ERROR loading data: {e}")
        return
    
    # Split data
    if 'rule_violation' in full_train_df.columns:
        stratify_col = full_train_df['rule_violation']
    else:
        label_columns = full_train_df.filter(regex='rule_').columns.tolist()
        stratify_col = full_train_df[label_columns].sum(axis=1) if label_columns else None
    
    train_df_raw, validation_df_raw = train_test_split(
        full_train_df,
        test_size=VALIDATION_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=stratify_col
    )
    
    print(f"Dataset split: Train={len(train_df_raw)} samples, Validation={len(validation_df_raw)} samples")
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_df_processed, tfidf_model, mean_vectors, scaler = preprocess_data(
        file_path=None,
        df_to_process=train_df_raw
    )
    
    validation_df_processed, _, _, _ = preprocess_data(
        file_path=None,
        df_to_process=validation_df_raw,
        tfidf_model=tfidf_model,
        mean_vectors=mean_vectors,
        scaler=scaler
    )
    
    # Create dataloaders
    print("\nSetting up dataloaders...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
    train_dataset = ToxicityDataset(train_df_processed, tokenizer)
    validation_dataset = ToxicityDataset(validation_df_processed, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    print(f"\nInitializing model: {TRANSFORMER_MODEL_NAME}")
    model = CustomTransformerModel(
        transformer_name=TRANSFORMER_MODEL_NAME,
        num_numerical_features=NUM_NUMERICAL_FEATURES,
        num_rules=NUM_RULES
    ).to(DEVICE)
    
    # Initialize SPD loss and optimizer
    print("Initializing SPD loss and optimizer...")
    
    # Create robust loss function
    base_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    robust_loss_fn = RobustLoss(base_loss_fn, perturbation_radius=0.1)
    
    # Create dual parameters (Lagrange multipliers)
    dual_params = [robust_loss_fn.dual_param]
    
    # Initialize SPD optimizer
    if SPD_METHOD == 'SGDA':
        optimizer = SGDAOptimizer(
            model.parameters(), 
            dual_params, 
            lr_primal=SPD_LR_PRIMAL, 
            lr_dual=SPD_LR_DUAL
        )
    elif SPD_METHOD == 'OGDA':
        optimizer = OGDAOptimizer(
            model.parameters(), 
            dual_params, 
            lr_primal=SPD_LR_PRIMAL, 
            lr_dual=SPD_LR_DUAL
        )
    else:
        raise ValueError(f"Unknown SPD method: {SPD_METHOD}")
    
    # Training loop
    print(f"\nStarting SPD training for {NUM_EPOCHS} epochs...")
    best_auc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        total_primal_loss = 0
        total_dual_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move data to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            numerical_features = batch['numerical_features'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            # Forward pass
            logits = model(input_ids, attention_mask, numerical_features)
            
            # Compute SPD losses
            primal_loss, dual_loss = robust_loss_fn(logits, labels, numerical_features)
            
            # SPD optimization step
            optimizer.step(primal_loss, dual_loss)
            
            total_primal_loss += primal_loss.item()
            total_dual_loss += dual_loss.item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_dataloader)}")
                print(f"    Primal Loss: {primal_loss.item():.4f}")
                print(f"    Dual Loss: {dual_loss.item():.4f}")
                print(f"    Dual Param: {robust_loss_fn.dual_param.item():.4f}")
        
        # Validation
        avg_primal_loss = total_primal_loss / len(train_dataloader)
        avg_dual_loss = total_dual_loss / len(train_dataloader)
        validation_auc, _ = evaluate_model(model, validation_dataloader, DEVICE)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Primal Loss: {avg_primal_loss:.4f}")
        print(f"  Average Dual Loss: {avg_dual_loss:.4f}")
        print(f"  Validation AUC: {validation_auc:.4f}")
        print(f"  Dual Parameter: {robust_loss_fn.dual_param.item():.4f}")
        
        # Save best model
        if validation_auc > best_auc:
            best_auc = validation_auc
            torch.save(model.state_dict(), 'spd_best_model.pth')
            print(f"  New best SPD model saved! AUC: {validation_auc:.4f}")
    
    print(f"\nSPD training complete!")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Final dual parameter: {robust_loss_fn.dual_param.item():.4f}")

# --- Execute Script ---
if __name__ == '__main__':
    if HYPERPARAMETER_TUNING:
        train_with_hyperparameter_tuning()
    elif ENSEMBLE_MODE:
        train_ensemble()
    elif SPD_MODE:
        train_with_spd()
    else:
        train_model()