# custom_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class LightGBMModel:
    """LightGBM model optimized for tabular data with engineered features"""
    
    def __init__(self, num_rules: int = 1):
        self.num_rules = num_rules
        self.models = {}
        self.feature_importance = {}
        
        # Optimized LightGBM parameters for tabular data
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'max_depth': 6,
            'min_split_gain': 0.0
        }
        
        print("🚀 LightGBM Model initialized with optimized parameters for tabular data")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train LightGBM model with cross-validation"""
        print(f"📊 Training LightGBM on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # Train model
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Get feature importance
        self.feature_importance = dict(zip(
            range(X_train.shape[1]), 
            self.model.feature_importance(importance_type='gain')
        ))
        
        print(f"✅ LightGBM training completed. Best iteration: {self.model.best_iteration}")
        return self
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self, 'model'):
            return self.model.predict(X, num_iteration=self.model.best_iteration)
        else:
            raise ValueError("Model not trained yet. Call fit() first.")
    
    def predict(self, X):
        """Get binary predictions"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance"""
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': [self.feature_importance.get(i, 0) for i in range(len(feature_names))]
        }).sort_values('importance', ascending=False)
        
        return importance_df

class XGBoostModel:
    """XGBoost model as alternative to LightGBM"""
    
    def __init__(self, num_rules: int = 1):
        self.num_rules = num_rules
        
        # Optimized XGBoost parameters
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'early_stopping_rounds': 50,
            'verbosity': 0
        }
        
        print("🚀 XGBoost Model initialized with optimized parameters")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost model"""
        print(f"📊 Training XGBoost on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Create XGBoost model
        self.model = xgb.XGBClassifier(**self.xgb_params)
        
        # Train with early stopping
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print(f"✅ XGBoost training completed")
        return self
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self, 'model'):
            return self.model.predict_proba(X)[:, 1]
        else:
            raise ValueError("Model not trained yet. Call fit() first.")
    
    def predict(self, X):
        """Get binary predictions"""
        if hasattr(self, 'model'):
            return self.model.predict(X)
        else:
            raise ValueError("Model not trained yet. Call fit() first.")
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance"""
        if hasattr(self, 'model'):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            raise ValueError("Model not trained yet. Call fit() first.")

class EnsembleModel:
    """Ensemble of LightGBM and XGBoost for maximum performance"""
    
    def __init__(self, num_rules: int = 1):
        self.num_rules = num_rules
        self.lgb_model = LightGBMModel(num_rules)
        self.xgb_model = XGBoostModel(num_rules)
        self.weights = [0.6, 0.4]  # LightGBM gets more weight (typically better for tabular)
        
        print("🚀 Ensemble Model (LightGBM + XGBoost) initialized")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Train both models"""
        print("🔄 Training ensemble models...")
        
        # Train LightGBM
        self.lgb_model.fit(X_train, y_train, X_val, y_val)
        
        # Train XGBoost
        self.xgb_model.fit(X_train, y_train, X_val, y_val)
        
        # Optimize weights based on validation performance
        if X_val is not None and y_val is not None:
            lgb_pred = self.lgb_model.predict_proba(X_val)
            xgb_pred = self.xgb_model.predict_proba(X_val)
            
            # Simple weight optimization
            lgb_auc = roc_auc_score(y_val, lgb_pred)
            xgb_auc = roc_auc_score(y_val, xgb_pred)
            
            total_auc = lgb_auc + xgb_auc
            if total_auc > 0:
                self.weights = [lgb_auc / total_auc, xgb_auc / total_auc]
            
            print(f"📊 Model weights optimized: LightGBM={self.weights[0]:.3f}, XGBoost={self.weights[1]:.3f}")
        
        print("✅ Ensemble training completed")
        return self
    
    def predict_proba(self, X):
        """Get ensemble prediction probabilities"""
        lgb_pred = self.lgb_model.predict_proba(X)
        xgb_pred = self.xgb_model.predict_proba(X)
        
        # Weighted average
        ensemble_pred = self.weights[0] * lgb_pred + self.weights[1] * xgb_pred
        return ensemble_pred
    
    def predict(self, X):
        """Get ensemble binary predictions"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

# Keep the original PyTorch model for compatibility
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
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 5. Create classifier dynamically in forward pass
        self.num_rules = num_rules
        self.classifier = None
        
        print(f"Model architecture: text_features={self.text_feature_size}, numerical_features={num_numerical_features}")
        
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
        
        # 6. Combine with numerical features
        combined_features = torch.cat((text_features, numerical_features.float()), dim=1)
        
        # 7. Create classifier dynamically if needed
        actual_combined_size = combined_features.shape[1]
        if self.classifier is None or self.classifier[0].in_features != actual_combined_size:
            print(f"Creating classifier for {actual_combined_size} features")
            self.classifier = nn.Sequential(
                nn.Linear(actual_combined_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, self.num_rules)
            ).to(combined_features.device)
            
            # Initialize the new classifier weights
            for module in self.classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        
        # 8. Final classification
        logits = self.classifier(combined_features)
        
        return logits