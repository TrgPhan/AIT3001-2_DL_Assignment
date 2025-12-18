"""
Utility functions for data loading, preprocessing, and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import re


class GSM8KDataset(Dataset):
    """GSM8K dataset for mathematical reasoning."""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            split: Dataset split ("train" or "test")
            tokenizer: Tokenizer for text encoding
            max_length: Maximum sequence length
            num_samples: Number of samples to load (None = all)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load GSM8K dataset
        print(f"Loading GSM8K {split} split...")
        dataset = load_dataset("gsm8k", "main", split=split)
        
        if num_samples is not None:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        self.data = dataset
        print(f"Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format question and answer
        question = item["question"]
        answer = item["answer"]
        
        # Extract numeric answer
        numeric_answer = self.extract_numeric_answer(answer)
        
        # Create prompt
        prompt = f"Question: {question}\nAnswer:"
        full_text = f"{prompt} {numeric_answer}"
        
        # Tokenize
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Create labels (mask prompt tokens)
            labels = encoding["input_ids"].clone()
            prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
            labels[:, :prompt_length] = -100  # Ignore prompt in loss
            
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
                "question": question,
                "answer": numeric_answer
            }
        else:
            return {
                "question": question,
                "answer": numeric_answer
            }
    
    @staticmethod
    def extract_numeric_answer(answer_text: str) -> str:
        """Extract numeric answer from GSM8K answer format."""
        # GSM8K answers end with "#### [number]"
        match = re.search(r'#### (.+)', answer_text)
        if match:
            return match.group(1).strip()
        return answer_text.strip()


def create_dataloaders(
    tokenizer: AutoTokenizer,
    train_batch_size: int = 4,
    eval_batch_size: int = 8,
    max_length: int = 512,
    num_train_samples: Optional[int] = None,
    num_eval_samples: Optional[int] = None,
    num_workers: int = 4
) -> tuple:
    """
    Create train and validation dataloaders for GSM8K.
    
    Returns:
        (train_dataloader, val_dataloader)
    """
    print(f"\n{'='*70}")
    print("Creating Dataloaders")
    print(f"{'='*70}")
    
    # Create datasets
    train_dataset = GSM8KDataset(
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_train_samples
    )
    
    val_dataset = GSM8KDataset(
        split="test",
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_eval_samples
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_dataloader)}")
    print(f"Val batches:   {len(val_dataloader)}")
    print(f"{'='*70}\n")
    
    return train_dataloader, val_dataloader


def evaluate_accuracy(
    model,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model accuracy on GSM8K.
    
    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    correct = 0
    total = 0
    
    print("\nEvaluating accuracy...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and total >= max_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            true_answers = batch["answer"]
            
            # Generate predictions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Check correctness
            for pred, true_ans in zip(predictions, true_answers):
                pred_answer = extract_answer_from_generation(pred)
                if pred_answer == true_ans:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def extract_answer_from_generation(text: str) -> str:
    """Extract numeric answer from generated text."""
    # Look for "Answer:" followed by number
    match = re.search(r'Answer:\s*([0-9,.]+)', text)
    if match:
        return match.group(1).strip()
    
    # Fallback: find last number in text
    numbers = re.findall(r'[0-9,.]+', text)
    if numbers:
        return numbers[-1].strip()
    
    return ""


def print_sample_predictions(
    model,
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    num_samples: int = 5,
    device: str = "cuda"
):
    """Print sample predictions for inspection."""
    model.eval()
    
    print(f"\n{'='*70}")
    print("Sample Predictions")
    print(f"{'='*70}\n")
    
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"][:num_samples].to(device)
    attention_mask = batch["attention_mask"][:num_samples].to(device)
    questions = batch["question"][:num_samples]
    true_answers = batch["answer"][:num_samples]
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False
        )
    
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    for i, (q, pred, true_ans) in enumerate(zip(questions, predictions, true_answers)):
        print(f"Example {i+1}:")
        print(f"Question: {q}")
        print(f"True Answer: {true_ans}")
        print(f"Prediction: {pred}")
        print(f"{'─'*70}\n")


def count_parameters(model) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"\n GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved\n")
