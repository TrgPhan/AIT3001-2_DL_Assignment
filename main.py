"""
Main script for LRD-PEFT: Latent Reasoning Distillation with Parameter-Efficient Fine-Tuning

Usage:
    python main.py --epochs 3 --alpha 0.1 --lr 1e-4
"""

import argparse
import torch
import os
from datetime import datetime

# Import custom modules
from src.model import TeacherModel, StudentModel, print_model_info
from src.lora import create_lora_model
from src.trainer import LRDPEFTTrainer
from src.utils import create_dataloaders, evaluate_accuracy, print_gpu_memory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LRD-PEFT: Latent Reasoning Distillation with PEFT"
    )
    
    # Model arguments
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Teacher model name or path"
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Student model name or path"
    )
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Eval batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Distillation weight")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    
    # Distillation arguments
    parser.add_argument(
        "--distill_layers",
        type=int,
        nargs="+",
        default=[8, 9, 10, 11, 12],
        help="Layers to distill from"
    )
    
    # Data arguments
    parser.add_argument(
        "--num_train_samples",
        type=int,
        default=2000,
        help="Number of training samples (None=all)"
    )
    parser.add_argument(
        "--num_eval_samples",
        type=int,
        default=500,
        help="Number of eval samples (None=all)"
    )
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=375, help="Eval every N steps")
    parser.add_argument("--save_steps", type=int, default=375, help="Save every N steps")
    parser.add_argument("--output_dir", type=str, default="./distill_output", help="Output directory")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, don't train")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Main training pipeline."""
    
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Print configuration
    print("\n" + "="*70)
    print("LRD-PEFT: Latent Reasoning Distillation with PEFT")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    for arg, value in vars(args).items():
        print(f"  {arg:20s}: {value}")
    print("="*70 + "\n")
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print_gpu_memory()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ========================================
    # Step 1: Load Teacher Model
    # ========================================
    print("\n" + "="*70)
    print("Loading Teacher Model")
    print("="*70)
    teacher = TeacherModel(
        model_name=args.teacher_model,
        device=device,
        load_in_4bit=True
    )
    print("Teacher model loaded successfully")
    print_model_info(teacher, "Teacher Model")
    
    # ========================================
    # Step 2: Load Student Model
    # ========================================
    print("\n" + "="*70)
    print("Loading Student Model")
    print("="*70)
    student = StudentModel(
        model_name=args.student_model,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    print("Student model loaded successfully")
    
    # ========================================
    # Step 3: Apply LoRA
    # ========================================
    print("\n" + "="*70)
    print("Applying LoRA Adapters")
    print("="*70)
    student.model = create_lora_model(
        base_model=student.model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    print("LoRA adapters applied successfully")
    print_model_info(student, "Student Model (with LoRA)")
    
    # ========================================
    # Step 4: Prepare Data
    # ========================================
    print("\n" + "="*70)
    print("Preparing Data")
    print("="*70)
    train_dataloader, val_dataloader = create_dataloaders(
        tokenizer=student.tokenizer,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        num_train_samples=args.num_train_samples,
        num_eval_samples=args.num_eval_samples,
        num_workers=args.num_workers
    )
    
    # ========================================
    # Step 5: Training or Evaluation
    # ========================================
    if args.eval_only:
        print("\n" + "="*70)
        print("Evaluation Mode")
        print("="*70)
        
        # Load checkpoint if provided
        if args.checkpoint:
            print(f"Loading checkpoint: {args.checkpoint}")
            student.model.load_state_dict(torch.load(args.checkpoint))
        
        # Evaluate
        metrics = evaluate_accuracy(
            model=student.model,
            dataloader=val_dataloader,
            tokenizer=student.tokenizer,
            device=device
        )
        
        print(f"\n{'='*70}")
        print("Evaluation Results:")
        print(f"{'='*70}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Correct:  {metrics['correct']}/{metrics['total']}")
        print(f"{'='*70}\n")
        
    else:
        # ========================================
        # Step 6: Train
        # ========================================
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        trainer = LRDPEFTTrainer(
            teacher_model=teacher,
            student_model=student,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            alpha=args.alpha,
            warmup_ratio=args.warmup_ratio,
            output_dir=args.output_dir,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            device=device,
            distill_layers=args.distill_layers
        )
        
        # Train
        trainer.train()
        
        # ========================================
        # Step 7: Final Evaluation
        # ========================================
        print("\n" + "="*70)
        print("Final Evaluation")
        print("="*70)
        
        metrics = evaluate_accuracy(
            model=student.model,
            dataloader=val_dataloader,
            tokenizer=student.tokenizer,
            device=device
        )
        
        print(f"\n{'='*70}")
        print("Final Results:")
        print(f"{'='*70}")
        print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Correct:  {metrics['correct']}/{metrics['total']}")
        print(f"{'='*70}\n")
    
    print("\n" + "="*70)
    print("All Done!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
