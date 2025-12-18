"""
Model definitions for Teacher and Student models.
Handles loading pretrained models and extracting hidden states.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Tuple


class TeacherModel(nn.Module):
    """Teacher model with latent reasoning capabilities (COCONUT-style)."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load model with 4-bit quantization for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()  # Teacher is frozen
        
    def extract_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states from specified layers.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            layers: List of layer indices to extract (default: last 5 layers)
            
        Returns:
            Dictionary mapping layer index to hidden states [batch_size, seq_len, hidden_dim]
        """
        if layers is None:
            layers = list(range(8, 13))  # Layers 8-12 (last 5 layers)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
        hidden_states = {}
        for layer_idx in layers:
            # Apply layer normalization for consistent scale
            hidden = outputs.hidden_states[layer_idx]
            hidden_states[layer_idx] = nn.functional.layer_norm(
                hidden,
                normalized_shape=(hidden.size(-1),)
            )
            
        return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Forward pass with hidden state extraction.
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden_states: Dictionary of hidden states from specified layers
        """
        hidden_states = self.extract_hidden_states(input_ids, attention_mask, layers)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            
        return outputs.logits, hidden_states


class StudentModel(nn.Module):
    """Student model with LoRA adapters for efficient fine-tuning."""
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = ["q_proj", "v_proj"]
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Store LoRA config
        self.lora_config = {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules
        }
        
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layers: Optional[List[int]] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Get hidden states from student model (same interface as teacher).
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            layers: List of layer indices to extract
            
        Returns:
            Dictionary mapping layer index to hidden states
        """
        if layers is None:
            layers = list(range(8, 13))  # Default: layers 8-12
            
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = {}
        for layer_idx in layers:
            # Apply layer normalization
            hidden = outputs.hidden_states[layer_idx]
            hidden_states[layer_idx] = nn.functional.layer_norm(
                hidden,
                normalized_shape=(hidden.size(-1),)
            )
            
        return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        layers: Optional[List[int]] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with hidden state extraction.
        
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            hidden_states: Dictionary of hidden states from specified layers
            loss: Task loss if labels provided
        """
        hidden_states = self.get_hidden_states(input_ids, attention_mask, layers)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs.logits, hidden_states, outputs.loss if labels is not None else None
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            "trainable": trainable_params,
            "total": total_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }


def print_model_info(model: nn.Module, name: str = "Model"):
    """Print detailed model information."""
    if hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        print(f"\n{'='*70}")
        print(f"{name} Information:")
        print(f"{'='*70}")
        print(f"Total parameters:     {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")
        print(f"Trainable %:          {params['trainable_percentage']:.4f}%")
        print(f"{'='*70}\n")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"{name}: {total:,} parameters")
