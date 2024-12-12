# verify.py

import os
import sys
import torch
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.model_loader import ModelLoader
from src.metrics import MetricsTracker
from src.data import create_mmlu_dataloader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelVerifier:
    """Handles verification of pruned models with comprehensive metrics tracking and detailed logging"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the model verifier with configuration settings
        
        Args:
            config_path: Path to configuration file
        """
        self.project_root = Path(__file__).parent.parent.absolute()
        sys.path.insert(0, str(self.project_root))
        
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['model']['device'])
        self.save_dir = self.project_root / self.config['system']['save_dir']
        self.final_model_path = self.save_dir / 'final_model'
        
        # Initialize as None, will be set during verification
        self.model_loader = None
        self.metrics_tracker = None
        self.model = None
        self.tokenizer = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file"""
        config_path = self.project_root / config_path
        logger.info(f"Loading config from: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _get_hf_token(self) -> str:
        """Retrieve HuggingFace token from environment or user input"""
        token = os.environ.get("HF_TOKEN")
        if not token:
            token = input("Please enter your HuggingFace token: ").strip()
            if not token:
                raise ValueError("HuggingFace token is required")
        return token
    
    def _setup_model(self) -> None:
        """Initialize and load the pruned model"""
        if not self.final_model_path.exists():
            raise FileNotFoundError(f"Pruned model not found at {self.final_model_path}")
            
        self.config['model']['local_path'] = str(self.final_model_path)
        logger.info(f"Looking for model at: {self.final_model_path}")
        
        logger.info("Loading pruned model...")
        self.model_loader = ModelLoader(
            config=self.config['model'],
            hf_token=self._get_hf_token()
        )
        self.model, self.tokenizer = self.model_loader.load()
    
    def _setup_metrics(self) -> None:
        """Initialize metrics tracking"""
        self.metrics_tracker = MetricsTracker(
            save_dir=self.save_dir,
            device=self.device,
            tokenizer=self.tokenizer,
            config=self.config,
            use_wandb=False
        )
    
    def _log_detailed_metrics(self, metrics: Any) -> None:
        """Log comprehensive evaluation metrics with detailed breakdowns"""
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
    
    def _save_verification_results(self, metrics: Any) -> None:
        """Save detailed verification results to JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.final_model_path),
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
        
        verification_path = self.save_dir / 'verified_results.json'
        with open(verification_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nVerification results saved to: {verification_path}")
        logger.info("="*50)
    
    def verify(self) -> Optional[Any]:
        """
        Execute the complete verification process with comprehensive logging
        
        Returns:
            Optional[Any]: The verification metrics if successful, None otherwise
        """
        try:
            # Clear any existing GPU memory
            torch.cuda.empty_cache()
            
            # Setup components
            self._setup_model()
            self._setup_metrics()
            
            # Clear GPU memory before evaluation
            torch.cuda.empty_cache()
            
            # Evaluate model
            logger.info("\nEvaluating model...")
            metrics = self.metrics_tracker.evaluate_model(
                self.model,
                self.tokenizer,
                verbose=True
            )
            
            # Log and save comprehensive results
            self._log_detailed_metrics(metrics)
            self._save_verification_results(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}")
            raise
        finally:
            # Cleanup
            if self.model is not None:
                del self.model
            torch.cuda.empty_cache()

def verify_model(config_path: str = "config/config.yaml") -> Optional[Any]:
    """
    Convenience function to run verification process
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Optional[Any]: Verification metrics if successful, None otherwise
    """
    verifier = ModelVerifier(config_path)
    return verifier.verify()

if __name__ == "__main__":
    verify_model()