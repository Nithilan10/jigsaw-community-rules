# custom_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class ResidualBlock(nn.Module):
    """
    Residual block with skip connections for better gradient flow.
    """
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.dropout(out)
        return self.layer_norm(out + residual)  # Residual connection with layer norm

class AttentionFeatureFusion(nn.Module):
    """
    Attention mechanism for fusing text and numerical features.
    """
    def __init__(self, text_dim: int, num_dim: int, attention_dim: int = 256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, attention_dim)
        self.num_proj = nn.Linear(num_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=8, batch_first=True)
        self.layer_norm = nn.LayerNorm(attention_dim)
        
    def forward(self, text_features, numerical_features):
        # Project features to same dimension
        text_proj = self.text_proj(text_features)
        num_proj = self.num_proj(numerical_features)
        
        # Create sequence for attention (text + numerical as sequence)
        sequence = torch.stack([text_proj, num_proj], dim=1)  # [batch, 2, attention_dim]
        
        # Apply self-attention
        attended, attention_weights = self.attention(sequence, sequence, sequence)
        
        # Layer normalization
        attended = self.layer_norm(attended)
        
        # Return fused representation (mean of attended features)
        fused = attended.mean(dim=1)  # [batch, attention_dim]
        return fused, attention_weights

class EnhancedClassifier(nn.Module):
    """
    Enhanced classifier with deeper network, residual connections, and attention.
    """
    def __init__(self, text_dim: int, num_dim: int, num_rules: int):
        super().__init__()
        
        # 1. Attention-based feature fusion
        self.feature_fusion = AttentionFeatureFusion(text_dim, num_dim, attention_dim=256)
        
        # 2. Deeper network with residual connections
        self.residual_block1 = ResidualBlock(256, 512, dropout_rate=0.2)
        self.residual_block2 = ResidualBlock(256, 512, dropout_rate=0.15)
        
        # 3. Final classification layers with graduated dropout
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),           # 256 → 512
            nn.ReLU(),
            nn.Dropout(0.2),              # Higher dropout early
            nn.Linear(512, 256),           # 512 → 256
            nn.ReLU(),
            nn.Dropout(0.15),             # Medium dropout
            nn.Linear(256, 128),           # 256 → 128
            nn.ReLU(),
            nn.Dropout(0.1),              # Lower dropout near output
            nn.Linear(128, num_rules)      # 128 → 1
        )
        
    def forward(self, text_features, numerical_features):
        # Fuse features using attention
        fused_features, attention_weights = self.feature_fusion(text_features, numerical_features)
        
        # Apply residual blocks
        out = self.residual_block1(fused_features)
        out = self.residual_block2(out)
        
        # Final classification
        logits = self.classifier(out)
        return logits, attention_weights

class CustomTransformerModel(nn.Module):
    def __init__(self, transformer_name: str, num_numerical_features: int, num_rules: int):
        super().__init__()
        
        # 1. Transformer Backbone (e.g., 'bert-base-uncased')
        # This generates the semantic representation (embeddings) of the text
        self.transformer = AutoModel.from_pretrained(transformer_name)
        
        # Get the size of the Transformer's output embedding (e.g., 768 for BERT-base)
        hidden_size = self.transformer.config.hidden_size
        
        # 2. Enhanced dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 3. Enhanced Classification Head with all improvements
        self.classifier = EnhancedClassifier(
            text_dim=hidden_size,
            num_dim=num_numerical_features,
            num_rules=num_rules
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        """
        Forward pass through the enhanced model.
        
        Args:
            input_ids: Tokenized text IDs from the Transformer.
            attention_mask: Mask for padding tokens.
            numerical_features: Your engineered numerical features (scaled).
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
            text_features = transformer_output.pooler_output
        else:
            # DeBERTa and other models use last_hidden_state with [CLS] token
            text_features = transformer_output.last_hidden_state[:, 0, :]  # [CLS] token (first token)
        
        # Apply dropout to text features
        text_features = self.dropout(text_features)
        
        # Pass through enhanced classifier with attention fusion and residual connections
        logits, attention_weights = self.classifier(text_features, numerical_features.float())
        
        # Logits are returned (pre-sigmoid) for use with BCEWithLogitsLoss
        return logits