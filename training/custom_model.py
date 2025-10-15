# custom_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_name: str, num_numerical_features: int, num_rules: int, vocab_size: int = 50000):
        super().__init__()
        
        # 1. Text Embedding Layer (replaces BERT)
        self.text_embedding = nn.Embedding(vocab_size, 256, padding_idx=0)
        
        # 2. Text Processing Layers
        self.text_lstm = nn.LSTM(256, 128, batch_first=True, bidirectional=True)
        self.text_attention = nn.Linear(256, 1)
        
        # 3. Combined feature processing
        self.text_feature_size = 128  # From LSTM output
        self.num_numerical_features = num_numerical_features
        combined_feature_size = self.text_feature_size + num_numerical_features
        
        print(f"Model architecture: text_features={self.text_feature_size}, numerical_features={num_numerical_features}, combined={combined_feature_size}")
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_rules)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized text IDs
            attention_mask: Mask for padding tokens
            numerical_features: Engineered numerical features
        """
        
        # 1. Text embedding
        text_emb = self.text_embedding(input_ids)
        
        # 2. Apply attention mask
        text_emb = text_emb * attention_mask.unsqueeze(-1).float()
        
        # 3. LSTM processing
        lstm_out, (hidden, cell) = self.text_lstm(text_emb)
        
        # 4. Attention mechanism
        attention_weights = F.softmax(self.text_attention(lstm_out), dim=1)
        text_features = (lstm_out * attention_weights).sum(dim=1)
        
        # 5. Apply dropout
        text_features = self.dropout(text_features)
        
        # Debug: Check shapes
        if text_features.shape[0] == 1:  # Only print for first batch
            print(f"Debug - text_features shape: {text_features.shape}")
            print(f"Debug - numerical_features shape: {numerical_features.shape}")
            print(f"Debug - Expected combined shape: {self.text_feature_size + self.num_numerical_features}")
        
        # 6. Combine with numerical features
        combined_features = torch.cat((text_features, numerical_features.float()), dim=1)
        
        # Debug: Check final shape
        if text_features.shape[0] == 1:  # Only print for first batch
            print(f"Debug - combined_features shape: {combined_features.shape}")
        
        # 7. Final classification
        logits = self.classifier(combined_features)
        
        return logits