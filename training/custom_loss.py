# custom_loss.py

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing import Dict, Any
import torch.nn.functional as F

# NOTE: The NUM_RULES constant is removed here. The logic now relies on the 
# configuration being passed via rule_weights, which should only use index 0.

# --- Custom Cost Sensitive Loss ---

class CustomCostSensitiveLoss(nn.Module):
    """
    Custom Cost Sensitive Loss that applies rule-based and feature-based weighting.
    """
    
    def __init__(self, rule_weights: Dict[str, float] = None, feature_weights: Dict[str, float] = None):
        """
        Initialize Custom Cost Sensitive Loss.
        
        Args:
            rule_weights: Dictionary of rule weights
            feature_weights: Dictionary of feature weights
        """
        super().__init__()
        self.rule_weights = rule_weights or {}
        self.feature_weights = feature_weights or {}
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, numerical_features: torch.Tensor = None, model: nn.Module = None) -> torch.Tensor:
        """
        Forward pass of Custom Cost Sensitive Loss.
        
        Args:
            logits: Model predictions (logits)
            labels: True labels
            numerical_features: Numerical features (optional)
            model: Model for regularization (optional, not used)
            
        Returns:
            Weighted loss value
        """
        # Ensure logits and labels have the same shape
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
            
        # Calculate base binary cross entropy loss
        base_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
        
        # Apply rule-based weighting (simplified - just use a constant weight for now)
        rule_weight = 1.0
        if self.rule_weights:
            rule_weight = list(self.rule_weights.values())[0] if self.rule_weights else 1.0
            
        # Apply feature-based weighting (simplified - just use a constant weight for now)
        feature_weight = 1.0
        if self.feature_weights:
            feature_weight = list(self.feature_weights.values())[0] if self.feature_weights else 1.0
            
        # Combine weights
        total_weight = rule_weight * feature_weight
        
        # Apply weighting
        weighted_loss = base_loss * total_weight
        
        # Clip loss to prevent numerical instability
        weighted_loss = torch.clamp(weighted_loss, min=-10.0, max=10.0)
        
        return weighted_loss.mean()

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
        
        # Ensure labels have the right shape [Batch, 1]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        
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
        # Ensure feature_boost has the same shape as element_wise_loss
        if feature_boost.dim() == 1:
            feature_boost = feature_boost.unsqueeze(1)
        final_weight = (1.0 + feature_boost) 
        
        # 4. Final Weighted Loss Calculation
        # The total loss is: Rule_Weight * Final_Weight * Base_Loss
        weighted_loss = element_wise_loss * rule_weight_tensor * final_weight
        
        # 5. Clip extreme values to prevent numerical instability
        weighted_loss = torch.clamp(weighted_loss, min=-10.0, max=10.0)
        
        # Return the mean loss across the entire batch
        return weighted_loss.mean()

# --- Advanced Loss Functions ---

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization and better generalization.
    
    Instead of hard labels (0 or 1), uses soft labels with smoothing factor.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            smoothing: Smoothing factor (0.0 = no smoothing, 1.0 = uniform distribution)
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Label Smoothing Loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels (0 or 1)
            
        Returns:
            Label smoothing loss value
        """
        # Ensure targets are float
        targets = targets.float()
        
        # Apply label smoothing
        # For binary classification: smooth labels become [smoothing/2, 1-smoothing/2]
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing / 2
        
        # Calculate binary cross entropy with smoothed labels
        loss = F.binary_cross_entropy_with_logits(inputs, smooth_targets, reduction='none')
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class MixupLoss(nn.Module):
    """
    Mixup Loss for data augmentation and regularization.
    
    Mixup creates virtual training examples by mixing pairs of examples and their labels.
    """
    
    def __init__(self, alpha: float = 0.2, reduction: str = 'mean'):
        """
        Initialize Mixup Loss.
        
        Args:
            alpha: Beta distribution parameter for mixup
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, 
                lam: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Mixup Loss.
        
        Args:
            inputs: Model predictions (logits)
            targets_a: First set of labels
            targets_b: Second set of labels
            lam: Mixing ratio (lambda)
            
        Returns:
            Mixup loss value
        """
        # Calculate loss for both mixed targets
        loss_a = F.binary_cross_entropy_with_logits(inputs, targets_a.float(), reduction='none')
        loss_b = F.binary_cross_entropy_with_logits(inputs, targets_b.float(), reduction='none')
        
        # Mix the losses according to lambda
        mixed_loss = lam * loss_a + (1 - lam) * loss_b
        
        # Apply reduction
        if self.reduction == 'mean':
            return mixed_loss.mean()
        elif self.reduction == 'sum':
            return mixed_loss.sum()
        else:
            return mixed_loss

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple:
    """
    Apply mixup data augmentation.
    
    Args:
        x: Input features
        y: Labels
        alpha: Beta distribution parameter
        
    Returns:
        Tuple of (mixed_x, y_a, y_b, lam)
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

class AdvancedRegularizationLoss(nn.Module):
    """
    Advanced regularization techniques including:
    - Dropout regularization
    - Weight decay
    - Spectral normalization
    - Gradient penalty
    """
    
    def __init__(self, base_loss_fn, weight_decay: float = 1e-4, 
                 gradient_penalty_weight: float = 0.1, spectral_norm: bool = False):
        """
        Initialize Advanced Regularization Loss.
        
        Args:
            base_loss_fn: Base loss function
            weight_decay: Weight decay coefficient
            gradient_penalty_weight: Gradient penalty weight
            spectral_norm: Whether to apply spectral normalization
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.weight_decay = weight_decay
        self.gradient_penalty_weight = gradient_penalty_weight
        self.spectral_norm = spectral_norm
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                model: nn.Module = None) -> torch.Tensor:
        """
        Forward pass with advanced regularization.
        
        Args:
            inputs: Model predictions
            targets: True labels
            model: Model for regularization (optional)
            
        Returns:
            Regularized loss value
        """
        # Base loss
        base_loss = self.base_loss_fn(inputs, targets.float())
        
        # Weight decay regularization
        weight_decay_loss = 0.0
        if model is not None:
            for param in model.parameters():
                weight_decay_loss += torch.norm(param, p=2) ** 2
            weight_decay_loss = self.weight_decay * weight_decay_loss
        
        # Gradient penalty (simplified version)
        gradient_penalty = 0.0
        if self.gradient_penalty_weight > 0 and model is not None:
            # Calculate gradient penalty
            inputs.requires_grad_(True)
            grad_outputs = torch.ones_like(inputs)
            gradients = torch.autograd.grad(
                outputs=inputs,
                inputs=model.parameters(),
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )
            
            if gradients:
                gradient_norm = torch.norm(gradients[0], p=2)
                gradient_penalty = self.gradient_penalty_weight * (gradient_norm - 1) ** 2
        
        # Total loss
        total_loss = base_loss + weight_decay_loss + gradient_penalty
        
        return total_loss

class CombinedAdvancedLoss(nn.Module):
    """
    Combined advanced loss function that integrates:
    - Focal Loss for class imbalance
    - Label Smoothing for regularization
    - Advanced regularization techniques
    """
    
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1, weight_decay: float = 1e-4,
                 use_mixup: bool = False, mixup_alpha: float = 0.2):
        """
        Initialize Combined Advanced Loss.
        
        Args:
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
            weight_decay: Weight decay coefficient
            use_mixup: Whether to use mixup augmentation
            mixup_alpha: Mixup alpha parameter
        """
        super().__init__()
        
        # Initialize component losses
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing_loss = LabelSmoothingLoss(smoothing=label_smoothing)
        self.mixup_loss = MixupLoss(alpha=mixup_alpha) if use_mixup else None
        self.advanced_reg_loss = AdvancedRegularizationLoss(
            base_loss_fn=nn.BCEWithLogitsLoss(),
            weight_decay=weight_decay
        )
        
        # Configuration
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        
        print(f"Combined Advanced Loss initialized:")
        print(f"  - Focal Loss: α={focal_alpha}, γ={focal_gamma}")
        print(f"  - Label Smoothing: {label_smoothing}")
        print(f"  - Weight Decay: {weight_decay}")
        print(f"  - Mixup: {'Enabled' if use_mixup else 'Disabled'}")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                model: nn.Module = None, use_mixup: bool = None) -> torch.Tensor:
        """
        Forward pass of combined advanced loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: True labels
            model: Model for regularization (optional)
            use_mixup: Override mixup usage (optional)
            
        Returns:
            Combined loss value
        """
        # Determine if we should use mixup
        apply_mixup = (use_mixup if use_mixup is not None else self.use_mixup) and self.mixup_loss is not None
        
        if apply_mixup:
            # Apply mixup data augmentation
            mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, self.mixup_alpha)
            
            # Calculate mixup loss
            loss = self.mixup_loss(mixed_inputs, targets_a, targets_b, lam)
        else:
            # Use focal loss with label smoothing
            focal_loss = self.focal_loss(inputs, targets)
            smooth_loss = self.label_smoothing_loss(inputs, targets)
            
            # Combine focal and label smoothing losses
            loss = 0.7 * focal_loss + 0.3 * smooth_loss
        
        # Add advanced regularization
        if model is not None:
            reg_loss = self.advanced_reg_loss(inputs, targets, model)
            loss = loss + 0.1 * reg_loss
        
        return loss