# custom_loss.py

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import Dict, Any

# NOTE: The NUM_RULES constant is removed here. The logic now relies on the 
# configuration being passed via rule_weights, which should only use index 0.

class CustomCostSensitiveLoss(_Loss):
    """
    A custom loss function that applies dynamic, cost-sensitive weights 
    to maximize Column-Averaged AUC by prioritizing hard/costly samples.
    
    It is now configured to handle a single label (index 0) based on your data.
    """
    def __init__(self, rule_weights: Dict[int, float], feature_weights: Dict[str, float], base_loss_fn=None):
        super().__init__()
        
        # Base loss function (Binary Cross Entropy with Logits is standard for multi-label)
        self.base_loss_fn = base_loss_fn if base_loss_fn else nn.BCEWithLogitsLoss(reduction='none')
        
        # Rule weight lookup table (Should only contain key 0 for single-label)
        self.rule_weights = rule_weights
        
        # Feature weight lookup table
        self.feature_weights = feature_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, numerical_features: torch.Tensor) -> torch.Tensor:
        """
        Calculates the weighted loss for the batch.
        
        Args:
            logits: Model predictions before sigmoid (shape: [Batch, 1])
            labels: True binary labels (shape: [Batch, 1])
            numerical_features: Pre-engineered features (shape: [Batch, Num_Features])
        """
        
        # Debug: Check for NaN or infinite values that could cause hanging
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN or Inf values detected in logits")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print("WARNING: NaN or Inf values detected in labels")
            labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(numerical_features).any() or torch.isinf(numerical_features).any():
            print("WARNING: NaN or Inf values detected in numerical_features")
            numerical_features = torch.nan_to_num(numerical_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 1. Start with the standard loss for every sample and every rule
        # Output shape: [Batch, 1]
        element_wise_loss = self.base_loss_fn(logits, labels.float())
        
        # 2. Determine the Rule-Based Weight (Policy Cost)
        rule_weight_tensor = torch.ones_like(labels).float() 
        
        # CRITICAL FIX: Loop/logic changed to only target index 0. 
        # The loop must ensure it only tries to access indices that exist (i.e., index 0).
        if 0 in self.rule_weights:
            weight = self.rule_weights[0]
            
            # Apply the base rule weight only to the samples that violate this rule (index 0)
            rule_weight_tensor[:, 0] = torch.where(
                labels[:, 0] == 1, 
                torch.tensor(weight, device=labels.device), 
                rule_weight_tensor[:, 0]
            )
            
        # 3. Determine the Feature-Based Boosts (High-Confidence FN Boost)
        
        # CRITICAL FIX: Initialize feature_boost to zero (solves UnboundLocalError)
        feature_boost = torch.zeros(logits.shape[0], device=logits.device)
        
        # Get feature indices: [0]=length, [2]=legal_advise, [3]=promo
        
        # A. Boost for Legal Advice Interaction
        if self.feature_weights.get('legal_advice_interaction_feature'):
            # numerical_features[:, 2] is 'legal_advice_interaction_feature'
            boost = numerical_features[:, 2] * self.feature_weights['legal_advice_interaction_feature']
            feature_boost += boost 

        # B. Boost for Spam/Promo Interaction
        if self.feature_weights.get('promo_persuasion_feature'):
            # numerical_features[:, 3] is 'promo_persuasion_feature'
            boost = numerical_features[:, 3] * self.feature_weights['promo_persuasion_feature']
            feature_boost += boost

        # C. Boost for Very Short Comments (Difficult Samples)
        if self.feature_weights.get('comment_length_short'):
            # numerical_features[:, 0] is scaled 'comment_length'
            # Check if length is in the scaled range [0, 0.1]
            is_short = (numerical_features[:, 0] <= 0.1).float()
            boost = is_short * self.feature_weights['comment_length_short']
            feature_boost += boost
            
        # Add a small base weight (1.0) and the calculated feature boost
        final_weight = (1.0 + feature_boost).unsqueeze(1).expand_as(element_wise_loss) 
        
        # 4. Final Weighted Loss Calculation
        # The total loss is: Rule_Weight * Final_Weight * Base_Loss
        weighted_loss = element_wise_loss * rule_weight_tensor * final_weight
        
        # 5. Clip extreme values to prevent numerical instability
        weighted_loss = torch.clamp(weighted_loss, min=-10.0, max=10.0)
        
        # Return the mean loss across the entire batch
        return weighted_loss.mean()