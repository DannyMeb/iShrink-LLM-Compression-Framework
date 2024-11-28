import torch
import time
import json
from typing import Dict, Optional
from dataclasses import dataclass
import logging
import psutil
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Stores comprehensive model metrics"""
    accuracy: float
    latency: float  # ms
    throughput: float  # samples/second
    memory_footprint: Dict[str, float]  # Memory stats in MB
    parameter_count: int

class MetricsTracker:
    def __init__(self,
                 save_dir: Path,
                 device: torch.device,
                 use_wandb: bool = False):
        self.save_dir = save_dir
        self.device = device
        self.use_wandb = use_wandb
        self.metrics_dir = save_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, 
                      model: torch.nn.Module,
                      eval_dataloader: torch.utils.data.DataLoader,
                      ) -> ModelMetrics:
        """Comprehensive model evaluation"""
        try:
            # 1. Measure accuracy
            accuracy = self._measure_accuracy(model, eval_dataloader)
            
            # 2. Measure latency and throughput
            latency, throughput = self._measure_performance(model, eval_dataloader)
            
            # 3. Measure memory footprint
            memory_stats = self._measure_memory_usage(model)
            
            # 4. Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                latency=latency,
                throughput=throughput,
                memory_footprint=memory_stats,
                parameter_count=param_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise

    def _measure_accuracy(self, 
                         model: torch.nn.Module,
                         dataloader: torch.utils.data.DataLoader) -> float:
        """Measure model accuracy on MMLU"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                logits = outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Calculate accuracy
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)
        
        accuracy = correct / total
        return accuracy

    def _measure_performance(self, 
                           model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader) -> tuple[float, float]:
        """Measure model latency and throughput"""
        model.eval()
        batch = next(iter(dataloader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(**batch)
        
        # Measure latency
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(**batch)
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency = (total_time / 10) * 1000  # Convert to ms
        throughput = (10 * batch['input_ids'].size(0)) / total_time  # samples/second
        
        return latency, throughput

    def _measure_memory_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """Measure model memory footprint"""
        memory_stats = {
            'gpu_allocated': torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
            'gpu_cached': torch.cuda.memory_reserved(self.device) / 1024**2,      # MB
            'cpu_memory': psutil.Process().memory_info().rss / 1024**2            # MB
        }
        return memory_stats

    def save_metrics(self, metrics: ModelMetrics, filename: str):
        """Save metrics to file"""
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'latency_ms': metrics.latency,
            'throughput_samples_per_sec': metrics.throughput,
            'memory_footprint_mb': metrics.memory_footprint,
            'parameter_count': metrics.parameter_count
        }
        
        save_path = self.metrics_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        logger.info(f"Saved metrics to {save_path}")
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics_dict)