"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, List


def create_lora_model(
    base_model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    task_type: TaskType = TaskType.CAUSAL_LM
) -> nn.Module:
    """
    Apply LoRA adapters to a base model.
    
    Args:
        base_model: Base transformer model
        r: LoRA rank (dimensionality of low-rank matrices)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA (e.g., ["q_proj", "v_proj"])
        task_type: Type of task (CAUSAL_LM for language modeling)
        
    Returns:
        Model with LoRA adapters attached
        
    LoRA Details:
        - Injects trainable rank decomposition matrices into existing weights
        - W' = W + BA, where B ∈ R^(d×r), A ∈ R^(r×d)
        - Only B and A are trained; W remains frozen
        - Reduces trainable parameters from d² to 2rd
    """
    if target_modules is None:
        # Default: Apply to query and value projections
        target_modules = ["q_proj", "v_proj"]
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=task_type,
        inference_mode=False
    )
    
    # Apply LoRA to model
    peft_model = get_peft_model(base_model, lora_config)
    
    print(f"\n{'='*70}")
    print("LoRA Configuration:")
    print(f"{'='*70}")
    print(f"Rank (r):             {r}")
    print(f"Alpha:                {lora_alpha}")
    print(f"Dropout:              {lora_dropout}")
    print(f"Target modules:       {', '.join(target_modules)}")
    print(f"Scaling factor:       {lora_alpha / r}")
    print(f"{'='*70}\n")
    
    # Print trainable parameters
    peft_model.print_trainable_parameters()
    
    return peft_model


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA adapter weights into base model for inference.
    
    After training, this combines W + BA into a single matrix,
    eliminating the overhead of adapter computation during inference.
    
    Args:
        model: PEFT model with LoRA adapters
        
    Returns:
        Model with merged weights
    """
    if hasattr(model, 'merge_and_unload'):
        print("Merging LoRA weights into base model...")
        merged_model = model.merge_and_unload()
        print("Weights merged successfully")
        return merged_model
    else:
        print("WARNING: Model does not have LoRA adapters to merge")
        return model


def save_lora_adapters(model: nn.Module, save_path: str):
    """
    Save only LoRA adapter weights (not the full model).
    
    This saves only the trainable parameters (B and A matrices),
    drastically reducing checkpoint size.
    
    Args:
        model: PEFT model with LoRA adapters
        save_path: Directory to save adapters
    """
    if hasattr(model, 'save_pretrained'):
        print(f"Saving LoRA adapters to: {save_path}")
        model.save_pretrained(save_path)
        print("Adapters saved successfully")
    else:
        print("⚠️ Model does not support save_pretrained")


def load_lora_adapters(base_model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Load LoRA adapters into a base model.
    
    Args:
        base_model: Base model to load adapters into
        adapter_path: Path to saved adapters
        
    Returns:
        Model with loaded adapters
    """
    from peft import PeftModel
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    print("Adapters loaded successfully")
    
    return peft_model


class LoRALayer(nn.Module):
    """
    Custom LoRA layer implementation for educational purposes.
    
    This is a simplified version showing how LoRA works.
    In practice, use the PEFT library's implementation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(lora_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns ΔW·x = B·A·x
        
        This is added to the original weight matrix output:
        output = W·x + ΔW·x
        """
        # x: [batch, seq_len, in_features]
        # A: [r, in_features]
        # B: [out_features, r]
        
        # x @ A^T: [batch, seq_len, r]
        result = self.dropout(x) @ self.lora_A.T
        
        # result @ B^T: [batch, seq_len, out_features]
        result = result @ self.lora_B.T
        
        # Scale by alpha/r
        result = result * self.scaling
        
        return result


def calculate_lora_params(d: int, r: int, num_layers: int, num_modules: int = 2) -> int:
    """
    Calculate total LoRA parameters.
    
    Args:
        d: Model hidden dimension
        r: LoRA rank
        num_layers: Number of transformer layers
        num_modules: Number of modules per layer (e.g., 2 for Q and V)
        
    Returns:
        Total number of trainable LoRA parameters
        
    Example:
        For Llama-2-7B (d=4096, 32 layers, Q+V projections):
        params = 2 × 16 × 4096 × 32 × 2 = 8,388,608 ≈ 8.4M
    """
    params_per_module = 2 * r * d  # A and B matrices
    total_params = params_per_module * num_layers * num_modules
    
    print(f"\nLoRA Parameter Calculation:")
    print(f"  Hidden dim (d):        {d}")
    print(f"  Rank (r):              {r}")
    print(f"  Layers:                {num_layers}")
    print(f"  Modules per layer:     {num_modules}")
    print(f"  Params per module:     {params_per_module:,}")
    print(f"  Total LoRA params:     {total_params:,}")
    
    return total_params
