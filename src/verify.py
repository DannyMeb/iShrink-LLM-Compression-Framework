import os
import sys
import torch
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Use absolute imports
from src.model_loader import ModelLoader
from src.metrics import MetricsTracker
from src.data import create_mmlu_dataloader

# Setup logging
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

def verify_model(config_path: str = "config/config.yaml"):
    """Load and verify pruned model performance"""
    try:
        # Load configuration
        config = load_config(config_path)
        device = torch.device(config['model']['device'])
        
        # Get HuggingFace token
        hf_token = get_hf_token()
        
        # Use PROJECT_ROOT to resolve save_dir
        save_dir = PROJECT_ROOT / config['system']['save_dir']
        final_model_path = save_dir / 'final_model'
        
        logger.info(f"Looking for model at: {final_model_path}")
        
        if not final_model_path.exists():
            raise FileNotFoundError(f"Pruned model not found at {final_model_path}")
        
        config['model']['local_path'] = str(final_model_path)
        
        # Initialize model loader and load model
        logger.info("Loading pruned model...")
        model_loader = ModelLoader(config=config['model'], hf_token=hf_token)
        model, tokenizer = model_loader.load()
        
        # Create evaluation dataloader
        logger.info("Creating evaluation dataloader...")
        eval_dataloader, _ = create_mmlu_dataloader(
            tokenizer=tokenizer,
            config=config,
            split="validation"
        )
        
        # Setup metrics tracker
        metrics_tracker = MetricsTracker(
            save_dir=save_dir,
            device=device,
            tokenizer=tokenizer,
            config=config,
            use_wandb=False  # Disable wandb for verification
        )
        
        # Clear GPU memory before evaluation
        torch.cuda.empty_cache()
        
        # Evaluate model
        logger.info("\nEvaluating model...")
        metrics = metrics_tracker.evaluate_model(model, tokenizer, verbose=True)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("MODEL VERIFICATION RESULTS")
        logger.info("="*50)
        
        logger.info("\n=== Basic Performance ===")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Latency: {metrics.latency:.2f} ms")
        logger.info(f"Throughput: {metrics.throughput:.2f} samples/second")
        
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
        
        logger.info("\n=== Compute Metrics ===")
        logger.info(f"FLOPs: {metrics.flops:,}")
        logger.info(f"MACs: {metrics.macs:,}")
        
        logger.info("\n=== Memory Usage ===")
        logger.info(f"GPU Allocated: {metrics.memory_footprint['gpu_allocated']:.2f} MB")
        logger.info(f"GPU Cached: {metrics.memory_footprint['gpu_cached']:.2f} MB")
        logger.info(f"CPU Memory: {metrics.memory_footprint['cpu_memory']:.2f} MB")
        logger.info(f"Activation Memory: {metrics.activation_memory_mb:.2f} MB")
        
        logger.info("\n=== Performance Metrics ===")
        logger.info(f"Bandwidth Usage: {metrics.bandwidth_usage:.2f} GB/s")
        logger.info(f"Cache Hit Rate: {metrics.cache_hits * 100:.2f}%")
        
        logger.info("\n=== Environmental Impact ===")
        logger.info(f"Power Usage: {metrics.power_watts:.2f} watts")
        logger.info(f"CO2 Emissions: {metrics.co2_emissions:.4f} grams")
        logger.info(f"Cost per Inference: ${metrics.cost_per_inference:.6f}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(final_model_path),
            'metrics': {
                'accuracy': metrics.accuracy,
                'latency_ms': metrics.latency,
                'throughput': metrics.throughput,
                'parameter_count': metrics.parameter_count,
                'active_parameters': metrics.active_parameter_count,
                'sparsity': metrics.sparsity,
                'flops': metrics.flops,
                'macs': metrics.macs,
                'memory_footprint': metrics.memory_footprint,
                'power_watts': metrics.power_watts,
                'co2_emissions': metrics.co2_emissions,
                'cost_per_inference': metrics.cost_per_inference,
                'compute_metrics': vars(metrics.compute_metrics),
                'cost_metrics': vars(metrics.cost_metrics)
            }
        }
        
        # Save verification results
        verification_path = save_dir / 'verification_results.json'
        with open(verification_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nVerification results saved to: {verification_path}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        raise
    finally:
        # Clean up
        torch.cuda.empty_cache()

if __name__ == "__main__":
    verify_model()