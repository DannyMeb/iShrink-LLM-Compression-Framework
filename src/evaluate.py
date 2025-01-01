#Evaluate.py

import os
import sys
import torch
import argparse
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from peft import PeftModel, PeftConfig

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Please enter your HuggingFace token: ").strip()
        if not token:
            raise ValueError("HuggingFace token is required")
    return token

def get_model_choice():
    while True:
        choice = input("\nWhich model to evaluate?\n"
                      "0. Initial model\n"
                      "1. Pruned model\n"
                      "2. Sparsified model\n"
                      "3. Finetuned model\n"
                      "Enter choice (0-3): ").strip()
        if choice in ['0', '1', '2', '3']:
            return int(choice)
        print("Invalid choice. Enter 0-3.")

def get_finetuned_model_choice():
    while True:
        choice = input("\nWhich finetuned model?\n1. Pruned\n2. Sparsified\nEnter (1-2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid choice. Enter 1-2.")

def get_sparsity_level():
    levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    while True:
        print("\nAvailable sparsity levels:", ", ".join(f"{x}%" for x in levels))
        choice = input("Enter sparsity %: ").strip().replace('%', '')
        try:
            level = int(choice)
            if level in levels:
                return level
            print(f"Choose from: {levels}")
        except ValueError:
            print("Enter valid number.")

def get_model_paths(config):
    model_choice = get_model_choice()
    save_dir = PROJECT_ROOT / config['system']['save_dir']
    model_name = config['model']['name'].split('/')[-1]
    model_base_dir = save_dir / model_name
    
    if model_choice == 0:
        model_path = PROJECT_ROOT / "models" / model_name
        adapter_path = None
        model_type = "initial"
        
    elif model_choice == 1:
        model_path = model_base_dir / 'pruned_model'
        adapter_path = None
        model_type = "pruned"
        
    elif model_choice == 2:
        sparsity = get_sparsity_level()
        model_path = model_base_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
        
        if not model_path.exists():
            available = [p.name for p in (model_base_dir/'sparsified_models').iterdir() 
                        if p.is_dir() and 'sparsified_model_at_' in p.name]
            if available:
                logger.error(f"Available models: {', '.join(available)}")
            raise FileNotFoundError(f"No model at {sparsity}% sparsity")
        
        adapter_path = None
        model_type = f"sparsified_{sparsity}%"
        
    else:  # Finetuned
        finetuned_choice = get_finetuned_model_choice()
        if finetuned_choice == 1:
            model_path = model_base_dir / 'pruned_model'
            adapter_path = model_base_dir / 'finetuned_models' / 'finetuned_pruned_model'
            model_type = "finetuned_pruned"
        else:
            sparsity = get_sparsity_level()
            model_path = model_base_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
            adapter_path = model_base_dir / 'finetuned_models' / f"sparsified_model_at_{sparsity}%_finetuned"
            model_type = f"finetuned_sparsified_{sparsity}%"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Base model not found: {model_path}")
        if not adapter_path.exists():
            raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
            
    logger.info(f"Using model from: {model_path}")
    if adapter_path:
        logger.info(f"Using LoRA adapter from: {adapter_path}")
    
    return model_path, adapter_path, model_type

def load_model(model_path: Path, adapter_path: Path | None, config: dict, hf_token: str):
    try:
        logger.info(f"Loading base model from: {model_path}")
        model_loader = ModelLoader(
            config={**config['model'], 'local_path': str(model_path)},
            hf_token=hf_token
        )
        base_model, tokenizer = model_loader.load()
        
        if adapter_path:
            logger.info(f"Loading LoRA from: {adapter_path}")
            try:
                model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=False,
                    use_safetensors=False
                )
                
                if verify_lora_loading(model):
                    logger.info("LoRA verified successfully")
                    logger.info("Merging LoRA weights...")
                    model = model.merge_and_unload()
                else:
                    raise RuntimeError("Failed to load LoRA")
                
            except Exception as e:
                logger.error(f"LoRA error: {str(e)}")
                logger.error("Falling back to base model")
                model = base_model
        else:
            model = base_model
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Model loading error: {str(e)}")
        raise

def verify_lora_loading(model):
    if hasattr(model, 'peft_config'):
        logger.info(f"Active adapters: {model.active_adapters}")
        
        if not hasattr(model.peft_config['default'], 'r'):
            logger.warning("No LoRA rank found")
            return False
            
        logger.info(f"LoRA rank: {model.peft_config['default'].r}")
        
        lora_params_found = False
        total_params = nonzero_params = 0
        
        for name, param in model.named_parameters():
            if 'lora' in name:
                total_params += 1
                if not torch.all(param == 0):
                    nonzero_params += 1
                logger.info(f"LoRA param: {name} (shape: {param.shape}, nonzero: {torch.count_nonzero(param).item()})")
                lora_params_found = True
        
        if not lora_params_found:
            logger.warning("No LoRA parameters found")
            return False
        
        if nonzero_params == 0:
            logger.warning("All LoRA parameters are zero")
            return False
            
        logger.info(f"Found {nonzero_params}/{total_params} nonzero LoRA params")
        return nonzero_params > 0
        
    return False

def save_evaluation_results(model_path: Path, adapter_path: Path | None, results: dict):
    save_path = adapter_path if adapter_path else model_path
    eval_path = save_path / 'evaluations'
    eval_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = eval_path / f'evaluation_results_{timestamp}.json'
    latest_file = eval_path / 'evaluation_results_latest.json'
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to:\nLatest: {latest_file}\nTimestamped: {result_file}")

def verify_model(config_path: str = "config/config.yaml"):
    try:
        config = load_config(config_path)
        device = torch.device(config['model']['device'])
        hf_token = get_hf_token()
        
        model_path, adapter_path, model_type = get_model_paths(config)
        
        model_config = sparsification_stats = None
        if (model_path / "config.json").exists():
            with open(model_path / "config.json") as f:
                model_config = json.load(f)
        if (model_path / "sparsification_stats.json").exists():
            with open(model_path / "sparsification_stats.json") as f:
                sparsification_stats = json.load(f)
        
        model, tokenizer = load_model(model_path, adapter_path, config, hf_token)
        
        logger.info("Creating evaluation dataloader...")
        # eval_dataloader, _ = create_mmlu_dataloader(
        #     tokenizer=tokenizer,
        #     config=config,
        #     split="validation"
        # )
        
        metrics_tracker = MetricsTracker(
            save_dir=model_path / "metrics",
            device=device,
            tokenizer=tokenizer, 
            config=config,
            use_wandb=False
        )
        
        torch.cuda.empty_cache()
        
        logger.info("\nEvaluating model...")
        metrics = metrics_tracker.evaluate_model(model, tokenizer, verbose=True)
        
        logger.info("\n" + "="*50)
        logger.info(f"MODEL VERIFICATION RESULTS ({model_type.upper()})")
        logger.info("="*50)
        
        logger.info("\n=== Performance ===")
        logger.info(f"MMLU Accuracy: {metrics.accuracy.mmlu:.4f}")
        logger.info(f"OpenBookQA: {metrics.accuracy.openbookqa:.4f}")
        logger.info(f"Winogrande: {metrics.accuracy.winogrande:.4f}")
        logger.info(f"HellaSWAG: {metrics.accuracy.hellaswag:.4f}")
        logger.info(f"Zero-shot Average: {metrics.accuracy.zero_shot_average:.4f}")
        logger.info(f"Latency: {metrics.latency:.2f} ms")
        logger.info(f"Throughput: {metrics.throughput:.2f} samples/second")
        logger.info(f"GPU Memory: {metrics.gpu_memory_mb:.2f} MB")
        
        logger.info("\n=== Model Size ===")
        logger.info(f"Total Parameters: {metrics.parameter_count:,}")
        logger.info(f"Active Parameters: {metrics.active_parameter_count:,}")
        logger.info(f"Overall Sparsity: {metrics.sparsity * 100:.2f}%")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'model_path': str(model_path),
            'adapter_path': str(adapter_path) if adapter_path else None,
            'model_config': model_config,
            'sparsification_stats': sparsification_stats,
            'metrics': {
                'accuracy': {
                        'mmlu': metrics.accuracy.mmlu,
                        'openbookqa': metrics.accuracy.openbookqa,
                        'winogrande': metrics.accuracy.winogrande,
                        'hellaswag': metrics.accuracy.hellaswag,
                        'zero_shot_average': metrics.accuracy.zero_shot_average},
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
                'compute_metrics': vars(metrics.compute_metrics),
                'cost_metrics': vars(metrics.cost_metrics)
            }
        }
        
        save_evaluation_results(model_path, adapter_path, results)
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    verify_model(args.config)

if __name__ == "__main__":
    main()