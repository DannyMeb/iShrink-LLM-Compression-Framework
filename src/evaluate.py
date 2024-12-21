# Evaluate.py

import os
import sys
import torch
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from peft import PeftModel, PeftConfig

# Setup project paths
PROJECT_ROOT = Path("/home/daniel.gebre/Thesis/LLM-Compression")
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import ModelLoader
from src.metrics import MetricsTracker
from src.data import create_mmlu_dataloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Load configuration file"""
    config_path = PROJECT_ROOT / config_path
    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_hf_token():
    """Get HuggingFace token from environment variable or prompt user"""
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Please enter your HuggingFace token: ").strip()
        if not token:
            raise ValueError("HuggingFace token is required")
    return token

def get_model_choice():
    """Prompt user for model choice"""
    while True:
        choice = input("\nWhich model would you like to evaluate?\n"
                      "0. Initial model\n"
                      "1. Final pruned model\n"
                      "2. Sparsified model\n"
                      "3. Finetuned model\n"
                      "Enter choice (0, 1, 2, or 3): ").strip()
        if choice in ['0', '1', '2', '3']:
            return int(choice)
        print("Invalid choice. Please enter 0, 1, 2, or 3.")

def get_finetuned_model_choice():
    """Prompt user to choose between finetuned pruned or sparsified model"""
    while True:
        choice = input("\nWhich type of finetuned model?\n1. Finetuned pruned model\n2. Finetuned sparsified model\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")

def get_sparsity_level():
    """Prompt user for sparsity level"""
    sparsity_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    while True:
        print("\nAvailable sparsity levels:", ", ".join(f"{x}%" for x in sparsity_levels))
        choice = input("Enter sparsity percentage: ").strip().replace('%', '')
        try:
            level = int(choice)
            if level in sparsity_levels:
                return level
            print("Invalid sparsity level. Please choose from the available levels.")
        except ValueError:
            print("Please enter a valid number.")

def get_model_paths(config):
    """Get model paths based on user choice"""
    model_choice = get_model_choice()
    save_dir = PROJECT_ROOT / config['system']['save_dir']
    models_dir = PROJECT_ROOT / "models/Llama-3.2-1B-Instruct"
    
    if model_choice == 0:
        model_path = models_dir
        adapter_path = None
        model_type = "initial"
    elif model_choice == 1:
        model_path = save_dir / 'final_model'
        adapter_path = None
        model_type = "pruned"
    elif model_choice == 2:
        sparsity = get_sparsity_level()
        model_path_1 = save_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
        model_path_2 = save_dir / f"sparsified_model_at_{sparsity}%"
        
        if model_path_1.exists():
            model_path = model_path_1
        elif model_path_2.exists():
            model_path = model_path_2
        else:
            available_models = []
            for path in [save_dir, save_dir / 'sparsified_models']:
                if path.exists():
                    available_models.extend(p.name for p in path.iterdir() if p.is_dir() and 'sparsified_model_at_' in p.name)
            
            if available_models:
                logger.error(f"Available sparsified models: {', '.join(available_models)}")
            raise FileNotFoundError(f"Sparsified model at {sparsity}% not found")
        
        adapter_path = None
        model_type = f"sparsified_{sparsity}%"
    else:  # model_choice == 3
        finetuned_choice = get_finetuned_model_choice()
        if finetuned_choice == 1:
            # For finetuned pruned model
            model_path = save_dir / 'final_model'
            adapter_path = save_dir / 'finetuned_models' / 'finetuned_pruned_model'
            model_type = "finetuned_pruned"
        else:
            # For finetuned sparsified model
            sparsity = get_sparsity_level()
            model_path = save_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
            adapter_path = save_dir / 'finetuned_models' / f"sparsified_model_at_{sparsity}%_finetuned"
            model_type = f"finetuned_sparsified_{sparsity}%"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Base model not found at {model_path}")
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found at {adapter_path}")
            
    logger.info(f"Using model from: {model_path}")
    if adapter_path:
        logger.info(f"Using LoRA adapter from: {adapter_path}")
    
    return model_path, adapter_path, model_type

def load_model(model_path: Path, adapter_path: Path | None, config: dict, hf_token: str):
    """Load model with optional LoRA adapter"""
    try:
        # Load base model
        logger.info(f"Loading base model from: {model_path}")
        model_loader = ModelLoader(
            config={**config['model'], 'local_path': str(model_path)},
            hf_token=hf_token
        )
        base_model, tokenizer = model_loader.load()
        
        # If adapter path is provided, load and apply LoRA
        if adapter_path:
            logger.info(f"Loading and applying LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                is_trainable=False
            )
            # Get the underlying model for metrics calculation
            model.model = base_model.model
            logger.info("Successfully applied LoRA adapter")
        else:
            model = base_model
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def save_evaluation_results(model_path: Path, adapter_path: Path | None, results: dict):
    """Save evaluation results in the model's directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = f'evaluation_results.json'
    
    # Save in appropriate directory (adapter directory for finetuned models)
    save_path = adapter_path if adapter_path else model_path
    eval_path = save_path / 'evaluations'
    eval_path.mkdir(exist_ok=True)
    
    result_file = eval_path / eval_filename
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nEvaluation results saved to: {result_file}")

def verify_model(config_path: str = "config/config.yaml"):
    """Load and verify model performance"""
    try:
        # Load configuration
        config = load_config(config_path)
        device = torch.device(config['model']['device'])
        hf_token = get_hf_token()
        
        # Get model paths
        model_path, adapter_path, model_type = get_model_paths(config)
        
        # Load model configs if available
        model_config = None
        sparsification_stats = None
        
        if (model_path / "config.json").exists():
            with open(model_path / "config.json") as f:
                model_config = json.load(f)
        
        if (model_path / "sparsification_stats.json").exists():
            with open(model_path / "sparsification_stats.json") as f:
                sparsification_stats = json.load(f)
        
        # Load model and tokenizer
        model, tokenizer = load_model(model_path, adapter_path, config, hf_token)
        
        # Create evaluation dataloader
        logger.info("Creating evaluation dataloader...")
        eval_dataloader, _ = create_mmlu_dataloader(
            tokenizer=tokenizer,
            config=config,
            split="validation"
        )
        
        # Setup metrics tracker
        save_dir = PROJECT_ROOT / config['system']['save_dir']
        metrics_tracker = MetricsTracker(
            save_dir=save_dir,
            device=device,
            tokenizer=tokenizer,
            config=config,
            use_wandb=False
        )
        
        torch.cuda.empty_cache()
        
        # Evaluate model
        logger.info("\nEvaluating model...")
        metrics = metrics_tracker.evaluate_model(model, tokenizer, verbose=True)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info(f"MODEL VERIFICATION RESULTS ({model_type.upper()})")
        logger.info("="*50)
        
        logger.info("\n=== Basic Performance ===")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Latency: {metrics.latency:.2f} ms")
        logger.info(f"Throughput: {metrics.throughput:.2f} samples/second")
        logger.info(f"GPU Allocated: {metrics.gpu_memory_mb:.2f} mb")

        
        logger.info("\n=== Model Size ===")
        logger.info(f"Total Parameters: {metrics.parameter_count:,}")
        logger.info(f"Active Parameters: {metrics.active_parameter_count:,}")
        logger.info(f"Overall Sparsity: {metrics.sparsity * 100:.2f}%")
        
        logger.info("\n=== Component Analysis ===")
        logger.info(f"Attention Parameters: {metrics.attention_params:,}")
        logger.info(f"Attention Nonzero: {metrics.attention_nonzero:,}")
        logger.info(f"Attention Sparsity: {metrics.attention_sparsity * 100:.2f}%")
        logger.info(f"MLP Parameters: {metrics.mlp_params:,}")
        logger.info(f"MLP Nonzero: {metrics.mlp_nonzero:,}")
        logger.info(f"MLP Sparsity: {metrics.mlp_sparsity * 100:.2f}%")
        
        # Prepare results dictionary
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'model_path': str(model_path),
            'adapter_path': str(adapter_path) if adapter_path else None,
            'model_config': model_config,
            'sparsification_stats': sparsification_stats,
            'metrics': {
                'accuracy': metrics.accuracy,
                'latency_ms': metrics.latency,
                'throughput': metrics.throughput,
                'parameter_count': metrics.parameter_count,
                'active_parameters': metrics.active_parameter_count,
                'sparsity': metrics.sparsity,
                'compute': {
                    'flops': metrics.flops,
                    'macs': metrics.macs,
                    'memory_footprint': metrics.memory_footprint,
                    'activation_memory_mb': metrics.activation_memory_mb
                },
                'attention': {
                    'params': metrics.attention_params,
                    'nonzero': metrics.attention_nonzero,
                    'sparsity': metrics.attention_sparsity
                },
                'mlp': {
                    'params': metrics.mlp_params,
                    'nonzero': metrics.mlp_nonzero,
                    'sparsity': metrics.mlp_sparsity
                },
                'compute_metrics': vars(metrics.compute_metrics),
                'cost_metrics': vars(metrics.cost_metrics)
            }
        }
        
        # Save evaluation results
        save_evaluation_results(model_path, adapter_path, results)
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    verify_model()