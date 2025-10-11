# custom_model.py

import torch
import torch.nn as nn
from transformers import AutoModel

class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_name: str, num_numerical_features: int, num_rules: int):
        super().__init__()
        
        # 1. Transformer Backbone (e.g., 'bert-base-uncased')
        # This generates the semantic representation (embeddings) of the text
        self.transformer = AutoModel.from_pretrained(transformer_name)
        
        # Get the size of the Transformer's output embedding (e.g., 768 for BERT-base)
        hidden_size = self.transformer.config.hidden_size
        
        # 2. Merger Layer: The total size of the combined features
        # Transformer Output (e.g., 768) + Numerical Features (e.g., 6)
        combined_feature_size = hidden_size + num_numerical_features
        
        # 3. Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 4. Final Classification Head
        # This layer takes the combined features and maps them to the number of rules (your predictions)
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_size, 512),  # Intermediate layer
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_rules) # Final output layer (one prediction/logit per rule)
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized text IDs from the Transformer.
            attention_mask: Mask for padding tokens.
            numerical_features: Your 6 engineered numerical features (scaled).
        """
        
        # Pass text through the Transformer
        transformer_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # The output of the Transformer's [CLS] token (used for classification)
        # Handle different transformer architectures (BERT vs DeBERTa)
        if hasattr(transformer_output, 'pooler_output') and transformer_output.pooler_output is not None:
            # BERT and some other models have pooler_output
            cls_output = transformer_output.pooler_output
        else:
            # DeBERTa and other models use last_hidden_state with [CLS] token
            cls_output = transformer_output.last_hidden_state[:, 0, :]  # [CLS] token (first token)
        
        cls_output = self.dropout(cls_output)
        
        # Combine Text Embeddings and Numerical Features
        combined_features = torch.cat((cls_output, numerical_features.float()), dim=1)
        
        # Pass combined features through the classification head
        logits = self.classifier(combined_features)
        
        # Logits are returned (pre-sigmoid) for use with BCEWithLogitsLoss
        return logits