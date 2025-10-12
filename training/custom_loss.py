# custom_loss.py

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import Dict, Any
import torch.nn.functional as F

# NOTE: The NUM_RULES constant is removed here. The logic now relies on the 
# configuration being passed via rule_weights, which should only use index 0.

# --- Focal Loss for Class Imbalance ---

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    where p_t is the predicted probability for the true class.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            
        Returns:
            Focal loss value
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate p_t (probability for true class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Calculate binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- Stochastic Primal-Dual (SPD) Optimizers ---

class SGDAOptimizer:
    """
    Stochastic Gradient Descent-Ascent (SGDA) Optimizer for Primal-Dual optimization.
    
    This implements the basic SPD method where we alternate between:
    - Primal step: minimize the main loss function
    - Dual step: maximize the dual objective (minimize negative dual)
    """
    
    def __init__(self, model_params, dual_params, lr_primal=1e-5, lr_dual=1e-4):
        """
        Initialize SGDA optimizer.
        
        Args:
            model_params: Model parameters (primal variables)
            dual_params: Dual parameters (Lagrange multipliers)
            lr_primal: Learning rate for primal updates
            lr_dual: Learning rate for dual updates
        """
        # Primal optimizer (for model parameters)
        self.primal_optimizer = torch.optim.AdamW(model_params, lr=lr_primal)
        
        # Dual optimizer (for dual parameters)
        self.dual_optimizer = torch.optim.AdamW(dual_params, lr=lr_dual)
        
        print(f"SGDA Optimizer initialized:")
        print(f"  - Primal LR: {lr_primal}")
        print(f"  - Dual LR: {lr_dual}")
    
    def step(self, primal_loss, dual_loss):
        """
        Perform one step of SGDA optimization.
        
        Args:
            primal_loss: Loss to minimize (main objective)
            dual_loss: Loss to maximize (dual objective)
        """
        # Primal step: minimize main loss
        self.primal_optimizer.zero_grad()
        primal_loss.backward(retain_graph=True)  # retain_graph for dual step
        self.primal_optimizer.step()
        
        # Dual step: maximize dual objective (minimize negative)
        self.dual_optimizer.zero_grad()
        (-dual_loss).backward()  # Maximize by minimizing negative
        self.dual_optimizer.step()

class OGDAOptimizer:
    """
    Optimistic Gradient Descent-Ascent (OGDA) Optimizer.
    
    This is an improved version of SGDA that uses "optimistic" updates
    with momentum to achieve better convergence properties.
    """
    
    def __init__(self, model_params, dual_params, lr_primal=1e-5, lr_dual=1e-4, momentum=0.9):
        """
        Initialize OGDA optimizer.
        
        Args:
            model_params: Model parameters (primal variables)
            dual_params: Dual parameters (Lagrange multipliers)
            lr_primal: Learning rate for primal updates
            lr_dual: Learning rate for dual updates
            momentum: Momentum factor for optimistic updates
        """
        # Primal optimizer with momentum
        self.primal_optimizer = torch.optim.AdamW(model_params, lr=lr_primal, betas=(momentum, 0.999))
        
        # Dual optimizer with momentum
        self.dual_optimizer = torch.optim.AdamW(dual_params, lr=lr_dual, betas=(momentum, 0.999))
        
        # Store previous gradients for optimistic updates
        self.prev_primal_grad = None
        self.prev_dual_grad = None
        
        print(f"OGDA Optimizer initialized:")
        print(f"  - Primal LR: {lr_primal}")
        print(f"  - Dual LR: {lr_dual}")
        print(f"  - Momentum: {momentum}")
    
    def step(self, primal_loss, dual_loss):
        """
        Perform one step of OGDA optimization.
        
        Args:
            primal_loss: Loss to minimize (main objective)
            dual_loss: Loss to maximize (dual objective)
        """
        # Compute current gradients
        self.primal_optimizer.zero_grad()
        primal_loss.backward(retain_graph=True)
        current_primal_grad = [p.grad.clone() for p in self.primal_optimizer.param_groups[0]['params']]
        
        # Optimistic primal update (if we have previous gradient)
        if self.prev_primal_grad is not None:
            for param, prev_grad in zip(self.primal_optimizer.param_groups[0]['params'], self.prev_primal_grad):
                if param.grad is not None:
                    param.grad = 2 * param.grad - prev_grad  # Optimistic update
        
        self.primal_optimizer.step()
        
        # Compute current dual gradients
        self.dual_optimizer.zero_grad()
        (-dual_loss).backward()
        current_dual_grad = [p.grad.clone() for p in self.dual_optimizer.param_groups[0]['params']]
        
        # Optimistic dual update (if we have previous gradient)
        if self.prev_dual_grad is not None:
            for param, prev_grad in zip(self.dual_optimizer.param_groups[0]['params'], self.prev_dual_grad):
                if param.grad is not None:
                    param.grad = 2 * param.grad - prev_grad  # Optimistic update
        
        self.dual_optimizer.step()
        
        # Store gradients for next iteration
        self.prev_primal_grad = current_primal_grad
        self.prev_dual_grad = current_dual_grad

class RobustLoss(nn.Module):
    """
    Robust loss function that can be used with SPD methods.
    
    This implements a min-max formulation where we minimize the worst-case loss
    over some perturbation set (e.g., adversarial examples).
    """
    
    def __init__(self, base_loss_fn, perturbation_radius=0.1):
        """
        Initialize robust loss.
        
        Args:
            base_loss_fn: Base loss function (e.g., BCEWithLogitsLoss)
            perturbation_radius: Radius of perturbation set
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.perturbation_radius = perturbation_radius
        
        # Dual parameter for the robust optimization
        self.dual_param = nn.Parameter(torch.tensor(0.0))
        
        print(f"Robust Loss initialized with perturbation radius: {perturbation_radius}")
    
    def forward(self, logits, labels, numerical_features=None):
        """
        Compute robust loss.
        
        Args:
            logits: Model predictions
            labels: True labels
            numerical_features: Additional features (optional)
            
        Returns:
            Tuple of (primal_loss, dual_loss) for SPD optimization
        """
        # Base loss (primal objective)
        base_loss = self.base_loss_fn(logits, labels.float())
        
        # Robust loss: min_θ max_δ L(θ, x + δ, y)
        # For simplicity, we use a quadratic penalty for perturbations
        perturbation_penalty = self.dual_param * (torch.norm(logits) ** 2 - self.perturbation_radius ** 2)
        
        # Primal loss: base loss + penalty
        primal_loss = base_loss + perturbation_penalty
        
        # Dual loss: negative of the penalty (to maximize)
        dual_loss = -perturbation_penalty
        
        return primal_loss, dual_loss

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