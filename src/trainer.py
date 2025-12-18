"""
Training loop for LRD-PEFT (Latent Reasoning Distillation with PEFT).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Optional, Dict, List
import os
import json
from datetime import datetime

from .distillation import CombinedDistillationLoss, compute_cosine_similarity


class LRDPEFTTrainer:
    """Trainer for Latent Reasoning Distillation with Parameter-Efficient Fine-Tuning."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        alpha: float = 0.1,
        warmup_ratio: float = 0.1,
        output_dir: str = "./output",
        logging_steps: int = 50,
        eval_steps: int = 375,
        save_steps: int = 375,
        device: str = "cuda",
        distill_layers: Optional[List[int]] = None
    ):
        """
        Args:
            teacher_model: Teacher model (frozen)
            student_model: Student model with LoRA adapters
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            alpha: Weight for distillation loss (0 to 1)
            warmup_ratio: Ratio of warmup steps to total steps
            output_dir: Directory to save checkpoints and logs
            logging_steps: Log metrics every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            device: Device to train on
            distill_layers: Layers to compute distillation loss on
        """
        self.teacher = teacher_model
        self.student = student_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.device = device
        
        if distill_layers is None:
            distill_layers = list(range(8, 13))  # Default: layers 8-12
        self.distill_layers = distill_layers
        
        # Setup
        os.makedirs(output_dir, exist_ok=True)
        self.teacher.eval()  # Teacher is always in eval mode
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.loss_fn = CombinedDistillationLoss(
            alpha=alpha,
            layers=distill_layers,
            distill_loss_type="mse"
        )
        
        # Metrics tracking
        self.global_step = 0
        self.metrics = {
            "train_loss": [],
            "train_distill_loss": [],
            "train_task_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
    def train(self):
        """Main training loop."""
        print(f"\n{'='*70}")
        print("🚀 Starting Training")
        print(f"{'='*70}")
        print(f"Epochs:            {self.num_epochs}")
        print(f"Steps per epoch:   {len(self.train_dataloader)}")
        print(f"Total steps:       {len(self.train_dataloader) * self.num_epochs}")
        print(f"Learning rate:     {self.learning_rate}")
        print(f"Alpha (distill):   {self.alpha}")
        print(f"Distill layers:    {self.distill_layers}")
        print(f"Output dir:        {self.output_dir}")
        print(f"{'='*70}\n")
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'='*70}")
            
            self.train_epoch(epoch)
            
            # Evaluate at end of epoch
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                print(f"\nValidation - Loss: {val_metrics['loss']:.4f}")
                
        print(f"\n{'='*70}")
        print("Training Complete")
        print(f"{'='*70}\n")
        
        # Save final model
        self.save_checkpoint("final_model")
        
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.student.train()
        
        epoch_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_task_loss = 0.0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            leave=True
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass - Teacher (no gradients)
            with torch.no_grad():
                teacher_logits, teacher_hiddens = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layers=self.distill_layers
                )
            
            # Forward pass - Student
            student_logits, student_hiddens, task_loss = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                layers=self.distill_layers
            )
            
            # Compute combined loss
            total_loss, distill_loss, task_loss = self.loss_fn(
                student_hiddens=student_hiddens,
                teacher_hiddens=teacher_hiddens,
                task_loss=task_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_distill_loss += distill_loss.item()
            epoch_task_loss += task_loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": total_loss.item(),
                "distill": distill_loss.item(),
                "task": task_loss.item(),
                "lr": self.scheduler.get_last_lr()[0]
            })
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                self.log_metrics(
                    total_loss.item(),
                    distill_loss.item(),
                    task_loss.item()
                )
            
            # Evaluation
            if self.val_dataloader is not None and self.global_step % self.eval_steps == 0:
                val_metrics = self.evaluate()
                print(f"\n[Step {self.global_step}] Val Loss: {val_metrics['loss']:.4f}")
                self.student.train()
            
            # Save checkpoint
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(f"checkpoint-{self.global_step}")
        
        # Log epoch summary
        avg_loss = epoch_loss / len(self.train_dataloader)
        avg_distill = epoch_distill_loss / len(self.train_dataloader)
        avg_task = epoch_task_loss / len(self.train_dataloader)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Loss:        {avg_loss:.4f}")
        print(f"  Avg Distill:     {avg_distill:.4f}")
        print(f"  Avg Task:        {avg_task:.4f}")
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.student.eval()
        
        total_loss = 0.0
        total_distill_loss = 0.0
        total_task_loss = 0.0
        all_similarities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Teacher forward
                teacher_logits, teacher_hiddens = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    layers=self.distill_layers
                )
                
                # Student forward
                student_logits, student_hiddens, task_loss = self.student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    layers=self.distill_layers
                )
                
                # Compute losses
                loss, distill_loss, task_loss = self.loss_fn(
                    student_hiddens=student_hiddens,
                    teacher_hiddens=teacher_hiddens,
                    task_loss=task_loss
                )
                
                total_loss += loss.item()
                total_distill_loss += distill_loss.item()
                total_task_loss += task_loss.item()
                
                # Compute similarities
                similarities = compute_cosine_similarity(student_hiddens, teacher_hiddens)
                all_similarities.append(similarities)
        
        # Average metrics
        num_batches = len(self.val_dataloader)
        metrics = {
            "loss": total_loss / num_batches,
            "distill_loss": total_distill_loss / num_batches,
            "task_loss": total_task_loss / num_batches,
            "similarities": self._average_similarities(all_similarities)
        }
        
        return metrics
    
    def _average_similarities(self, all_sims: List[Dict[int, float]]) -> Dict[int, float]:
        """Average similarity scores across batches."""
        if not all_sims:
            return {}
        
        avg_sims = {}
        layers = all_sims[0].keys()
        
        for layer in layers:
            avg_sims[layer] = sum(s[layer] for s in all_sims) / len(all_sims)
        
        return avg_sims
    
    def log_metrics(self, total_loss: float, distill_loss: float, task_loss: float):
        """Log training metrics."""
        self.metrics["train_loss"].append(total_loss)
        self.metrics["train_distill_loss"].append(distill_loss)
        self.metrics["train_task_loss"].append(task_loss)
        self.metrics["learning_rate"].append(self.scheduler.get_last_lr()[0])
        
        # Save metrics to file
        metrics_file = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save student model (with LoRA adapters)
        if hasattr(self.student, 'save_pretrained'):
            self.student.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.student.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": self.metrics
        }
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        print(f"Checkpoint saved: {checkpoint_dir}")
