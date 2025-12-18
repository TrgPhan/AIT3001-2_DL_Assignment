"""
Distillation loss functions for latent reasoning transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LatentDistillationLoss(nn.Module):
    """
    Multi-layer hidden state distillation loss.
    
    Aligns student hidden states with teacher hidden states across
    multiple transformer layers to transfer latent reasoning patterns.
    """
    
    def __init__(
        self,
        layers: Optional[list] = None,
        loss_type: str = "mse",
        temperature: float = 1.0
    ):
        """
        Args:
            layers: List of layer indices to compute loss on (default: [8-12])
            loss_type: Type of loss - "mse", "cosine", or "kl"
            temperature: Temperature for softmax (used in KL divergence)
        """
        super().__init__()
        self.layers = layers if layers is not None else list(range(8, 13))
        self.loss_type = loss_type
        self.temperature = temperature
        
    def forward(
        self,
        student_hiddens: Dict[int, torch.Tensor],
        teacher_hiddens: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute distillation loss between student and teacher hidden states.
        
        Args:
            student_hiddens: Dict mapping layer_idx -> hidden states [B, L, D]
            teacher_hiddens: Dict mapping layer_idx -> hidden states [B, L, D]
            
        Returns:
            Scalar loss value
        """
        total_loss = 0.0
        num_layers = 0
        
        for layer_idx in self.layers:
            if layer_idx not in student_hiddens or layer_idx not in teacher_hiddens:
                continue
                
            student_h = student_hiddens[layer_idx]
            teacher_h = teacher_hiddens[layer_idx]
            
            # Ensure same shape
            if student_h.shape != teacher_h.shape:
                raise ValueError(
                    f"Shape mismatch at layer {layer_idx}: "
                    f"student {student_h.shape} vs teacher {teacher_h.shape}"
                )
            
            # Compute layer-wise loss
            if self.loss_type == "mse":
                layer_loss = F.mse_loss(student_h, teacher_h)
            elif self.loss_type == "cosine":
                # Cosine embedding loss (1 - cosine_similarity)
                layer_loss = 1 - F.cosine_similarity(
                    student_h.view(-1, student_h.size(-1)),
                    teacher_h.view(-1, teacher_h.size(-1))
                ).mean()
            elif self.loss_type == "kl":
                # KL divergence on normalized hidden states
                student_prob = F.softmax(student_h / self.temperature, dim=-1)
                teacher_prob = F.softmax(teacher_h / self.temperature, dim=-1)
                layer_loss = F.kl_div(
                    student_prob.log(),
                    teacher_prob,
                    reduction='batchmean'
                )
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            total_loss += layer_loss
            num_layers += 1
        
        # Average across layers
        if num_layers == 0:
            return torch.tensor(0.0, device=student_h.device)
        
        return total_loss / num_layers


class CombinedDistillationLoss(nn.Module):
    """
    Combined loss: latent distillation + task loss.
    
    L_total = α * L_distill + (1-α) * L_task
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        layers: Optional[list] = None,
        distill_loss_type: str = "mse",
        temperature: float = 1.0
    ):
        """
        Args:
            alpha: Weight for distillation loss (0 to 1)
            layers: Layers to compute distillation loss on
            distill_loss_type: Type of distillation loss
            temperature: Temperature for softmax
        """
        super().__init__()
        self.alpha = alpha
        self.distill_loss = LatentDistillationLoss(
            layers=layers,
            loss_type=distill_loss_type,
            temperature=temperature
        )
        
    def forward(
        self,
        student_hiddens: Dict[int, torch.Tensor],
        teacher_hiddens: Dict[int, torch.Tensor],
        task_loss: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            student_hiddens: Student hidden states
            teacher_hiddens: Teacher hidden states
            task_loss: Cross-entropy loss on final outputs
            
        Returns:
            (total_loss, distill_loss, task_loss) tuple
        """
        distill_loss = self.distill_loss(student_hiddens, teacher_hiddens)
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss
        
        return total_loss, distill_loss, task_loss


def compute_cosine_similarity(
    student_hiddens: Dict[int, torch.Tensor],
    teacher_hiddens: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """
    Compute cosine similarity between student and teacher at each layer.
    
    Used for monitoring/validation, not for training loss.
    
    Returns:
        Dictionary mapping layer index to average cosine similarity
    """
    similarities = {}
    
    for layer_idx in student_hiddens.keys():
        if layer_idx not in teacher_hiddens:
            continue
            
        student_h = student_hiddens[layer_idx]
        teacher_h = teacher_hiddens[layer_idx]
        
        # Flatten to [batch*seq_len, hidden_dim]
        student_flat = student_h.view(-1, student_h.size(-1))
        teacher_flat = teacher_h.view(-1, teacher_h.size(-1))
        
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=1)
        similarities[layer_idx] = cos_sim.mean().item()
    
    return similarities


def compute_layer_wise_mse(
    student_hiddens: Dict[int, torch.Tensor],
    teacher_hiddens: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """
    Compute MSE loss at each layer separately.
    
    Useful for analyzing which layers align well.
    
    Returns:
        Dictionary mapping layer index to MSE loss
    """
    layer_losses = {}
    
    for layer_idx in student_hiddens.keys():
        if layer_idx not in teacher_hiddens:
            continue
            
        student_h = student_hiddens[layer_idx]
        teacher_h = teacher_hiddens[layer_idx]
        
        mse = F.mse_loss(student_h, teacher_h).item()
        layer_losses[layer_idx] = mse
    
    return layer_losses


class TokenLevelDistillationLoss(nn.Module):
    """
    Standard token-level distillation (for comparison baseline).
    
    Matches output logits distribution using KL divergence.
    """
    
    def __init__(self, temperature: float = 2.0):
        """
        Args:
            temperature: Softening temperature for logits
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between student and teacher output distributions.
        
        Args:
            student_logits: Student model logits [B, L, V]
            teacher_logits: Teacher model logits [B, L, V]
            
        Returns:
            KL divergence loss
        """
        # Soften probabilities with temperature
        student_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_prob = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(
            student_prob,
            teacher_prob,
            reduction='batchmean'
        )
        
        # Scale by temperature^2 (standard practice)
        loss = loss * (self.temperature ** 2)
        
        return loss


def print_loss_info(
    total_loss: float,
    distill_loss: float,
    task_loss: float,
    similarities: Optional[Dict[int, float]] = None,
    step: int = 0
):
    """Pretty print loss information during training."""
    print(f"\n{'='*70}")
    print(f"Step {step} - Loss Summary:")
    print(f"{'='*70}")
    print(f"Total Loss:        {total_loss:.4f}")
    print(f"Distillation Loss: {distill_loss:.4f}")
    print(f"Task Loss:         {task_loss:.4f}")
    
    if similarities:
        print(f"\nLayer-wise Cosine Similarity:")
        for layer_idx, sim in sorted(similarities.items()):
            print(f"  Layer {layer_idx:2d}: {sim:.4f}")
    
    print(f"{'='*70}\n")
