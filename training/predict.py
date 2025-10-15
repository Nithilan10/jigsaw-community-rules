# predict.py - Updated prediction script for no-BERT model

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from preprocess import preprocess_data
from custom_model import CustomTransformerModel

# Simple Tokenizer Class (same as in training)
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 50000):
        self.vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.vocab_size = vocab_size
        self.unk_token_id = 1
        self.pad_token_id = 0
        self.cls_token_id = 2
        self.sep_token_id = 3
        
    def set_vocab(self, vocab):
        """Set vocabulary from saved training data"""
        self.vocab = vocab
        print(f"Loaded vocabulary with {len(self.vocab)} words")
    
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

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df['comment_text'].values
        
        # Get all numerical features dynamically
        numerical_cols = [col for col in df.columns 
                         if col not in ['comment_text', 'rule_violation', 'subreddit', 'rule'] 
                         and str(df.dtypes[col]) in ['int64', 'float64']]
        
        # Convert to numpy and handle NaN
        numerical_array = df[numerical_cols].values
        numerical_array = np.nan_to_num(numerical_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.features = torch.tensor(numerical_array, dtype=torch.float32)
        
        # Handle labels if they exist
        if 'rule_violation' in df.columns:
            labels_array = df['rule_violation'].values
            labels_array = np.nan_to_num(labels_array, nan=0.0)
            self.labels = torch.tensor(labels_array, dtype=torch.float32).unsqueeze(1)
        else:
            self.labels = torch.zeros(len(self.features), 1, dtype=torch.float32)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Feature dataset created: {len(self.features)} samples, {len(numerical_cols)} features")

    def __len__(self):
        return len(self.features)

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
            'numerical_features': self.features[idx],
            'labels': self.labels[idx]
        }

def main():
    print("=== NO-BERT PREDICTION SCRIPT ===")
    
    # Load test data
    test_df = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/test.csv')
    print(f"Test data loaded: {test_df.shape}")
    print(f"Test data columns: {list(test_df.columns)}")
    
    # Load training components
    try:
        components = torch.load('/kaggle/input/reddit_jigsaw/pytorch/bert-base-uncased/1/training_components.pth', map_location=torch.device('cpu'))
        print("Training components loaded successfully!")
    except Exception as e:
        print(f"Error loading training components: {e}")
        print("Using default values...")
        components = {
            'tokenizer_vocab': {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3},
            'tokenizer_vocab_size': 50000,
            'num_numerical_features': 153,
            'vocab_size': 50000
        }
    
    # Setup tokenizer
    tokenizer = SimpleTokenizer(vocab_size=components['vocab_size'])
    tokenizer.set_vocab(components['tokenizer_vocab'])
    
    # Preprocess test data
    print("\nPreprocessing test data...")
    test_df_processed, _, _, _ = preprocess_data(
        df_to_process=test_df,
        enable_spacy=False
    )
    
    print(f"Preprocessed test data shape: {test_df_processed.shape}")
    
    # Create model
    model = CustomTransformerModel(
        transformer_name='bert-base-uncased',  # Not used but kept for compatibility
        num_numerical_features=components['num_numerical_features'],
        num_rules=1,
        vocab_size=components['vocab_size']
    )
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load('/kaggle/input/reddit_jigsaw/pytorch/bert-base-uncased/1/best_model.pth', map_location=torch.device('cpu')))
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using untrained model...")
    
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = CustomDataset(test_df_processed, tokenizer, max_length=256)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Make predictions
    all_predictions = []
    print("\nMaking predictions...")
    
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            numerical_features = batch['numerical_features']
            
            logits = model(input_ids, attention_mask, numerical_features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_predictions.extend(probs.flatten())
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'row_id': test_df['row_id'],
        'rule_violation': all_predictions
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission file created with {len(submission)} predictions!")
    print(f"Prediction range: {min(all_predictions):.4f} to {max(all_predictions):.4f}")
    
    # Display first few predictions
    print("\nFirst 10 predictions:")
    print(submission.head(10))
    
    return submission

if __name__ == '__main__':
    submission = main()
