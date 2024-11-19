import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelSize:
    """Tracks model size metrics"""
    total_params: int
    total_memory: float  # MB
    param_counts: Dict[str, int] = field(default_factory=dict)
    memory_per_layer: Dict[str, float] = field(default_factory=dict)
    
    def compression_ratio(self, original_size: 'ModelSize') -> Dict[str, float]:
        """Calculate compression ratios"""
        return {
            'params_ratio': self.total_params / original_size.total_params,
            'memory_ratio': self.total_memory / original_size.total_memory
        }

@dataclass
class LatencyMetrics:
    """Tracks model latency metrics"""
    avg_latency: float  # ms
    std_latency: float
    p90_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float  # samples/second
    
    @classmethod
    def from_measurements(cls, latencies: List[float], batch_size: int) -> 'LatencyMetrics':
        """Create metrics from latency measurements"""
        return cls(
            avg_latency=np.mean(latencies),
            std_latency=np.std(latencies),
            p90_latency=np.percentile(latencies, 90),
            p95_latency=np.percentile(latencies, 95),
            p99_latency=np.percentile(latencies, 99),
            throughput=batch_size / (np.mean(latencies) / 1000)  # Convert ms to seconds
        )

@dataclass
class AccuracyMetrics:
    """Tracks model accuracy metrics"""
    accuracy: float
    perplexity: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

@dataclass
class PruningMetrics:
    """Comprehensive pruning metrics"""
    step: int
    model_size: ModelSize
    latency: LatencyMetrics
    accuracy: AccuracyMetrics
    pruned_groups: List[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format"""
        return {
            'step': self.step,
            'timestamp': self.timestamp,
            'model_size': {
                'total_params': self.model_size.total_params,
                'total_memory': self.model_size.total_memory,
                'param_counts': dict(self.model_size.param_counts),
                'memory_per_layer': dict(self.model_size.memory_per_layer)
            },
            'latency': {
                'avg_latency': self.latency.avg_latency,
                'std_latency': self.latency.std_latency,
                'p90_latency': self.latency.p90_latency,
                'p95_latency': self.latency.p95_latency,
                'p99_latency': self.latency.p99_latency,
                'throughput': self.latency.throughput
            },
            'accuracy': {
                'accuracy': self.accuracy.accuracy,
                'perplexity': self.accuracy.perplexity,
                'loss': self.accuracy.loss,
                'f1_score': self.accuracy.f1_score,
                'precision': self.accuracy.precision,
                'recall': self.accuracy.recall
            },
            'pruned_groups': self.pruned_groups
        }

class MetricsTracker:
    """Tracks and manages pruning metrics"""
    
    def __init__(self, 
                 save_dir: str,
                 original_model: torch.nn.Module,
                 device: str = 'cuda',
                 num_warmup_steps: int = 10):
        """
        Initialize metrics tracker
        
        Args:
            save_dir: Directory to save metrics
            original_model: Original unpruned model
            device: Device for computations
            num_warmup_steps: Number of warmup steps for latency measurement
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.num_warmup_steps = num_warmup_steps
        self.metrics_history: List[PruningMetrics] = []
        
        # Get original model metrics
        self.original_size = self._measure_model_size(original_model)
        
        logger.info(f"Initialized MetricsTracker. Original model size: {self.original_size.total_params:,} parameters")
    
    def measure_pruning_metrics(self,
                              model: torch.nn.Module,
                              eval_dataloader: DataLoader,
                              pruned_groups: List[str],
                              step: int) -> PruningMetrics:
        """Measure comprehensive metrics for pruned model"""
        # Ensure model is in eval mode
        model.eval()
        
        # Measure model size
        model_size = self._measure_model_size(model)
        
        # Measure latency
        latency = self._measure_latency(model, eval_dataloader)
        
        # Measure accuracy
        accuracy = self._measure_accuracy(model, eval_dataloader)
        
        # Create metrics object
        metrics = PruningMetrics(
            step=step,
            model_size=model_size,
            latency=latency,
            accuracy=accuracy,
            pruned_groups=pruned_groups
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        return metrics
    
    def _measure_model_size(self, model: torch.nn.Module) -> ModelSize:
        """Measure model size metrics"""
        total_params = 0
        total_memory = 0
        param_counts = defaultdict(int)
        memory_per_layer = defaultdict(float)
        
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]
            num_params = param.numel()
            memory = num_params * param.element_size() / (1024 * 1024)  # MB
            
            total_params += num_params
            total_memory += memory
            param_counts[layer_name] += num_params
            memory_per_layer[layer_name] += memory
        
        return ModelSize(
            total_params=total_params,
            total_memory=total_memory,
            param_counts=dict(param_counts),
            memory_per_layer=dict(memory_per_layer)
        )
    
    def _measure_latency(self, 
                        model: torch.nn.Module,
                        dataloader: DataLoader) -> LatencyMetrics:
        """Measure model latency metrics"""
        latencies = []
        batch_size = dataloader.batch_size
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(self.num_warmup_steps):
                batch = next(iter(dataloader))[0].to(self.device)
                model(batch)
        
        # Actual measurements
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(self.device)
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                model(batch)
                end.record()
                
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
        
        return LatencyMetrics.from_measurements(latencies, batch_size)
    
    def _measure_accuracy(self, 
                         model: torch.nn.Module,
                         dataloader: DataLoader) -> AccuracyMetrics:
        """Measure model accuracy metrics"""
        total_correct = 0
        total_samples = 0
        total_loss = 0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(inputs)
                
                # Compute accuracy
                predicted = outputs.logits.argmax(dim=-1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
                
                # Compute loss
                loss = outputs.loss
                total_loss += loss.item() * targets.size(0)
                
                # Store predictions for F1 score
                predictions.extend(predicted.cpu().numpy())
                labels.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        # Calculate F1, precision, recall
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return AccuracyMetrics(
            accuracy=accuracy,
            loss=avg_loss,
            f1_score=f1,
            precision=precision,
            recall=recall
        )
    
    def _save_metrics(self, metrics: PruningMetrics):
        """Save metrics to disk"""
        metrics_file = self.save_dir / f"metrics_step_{metrics.step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """Plot metrics history"""
        save_dir = Path(save_dir) if save_dir else self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics for plotting
        steps = [m.step for m in self.metrics_history]
        accuracies = [m.accuracy.accuracy for m in self.metrics_history]
        latencies = [m.latency.avg_latency for m in self.metrics_history]
        memory_ratios = [m.model_size.total_memory / self.original_size.total_memory 
                        for m in self.metrics_history]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy plot
        axes[0, 0].plot(steps, accuracies, marker='o')
        axes[0, 0].set_title('Accuracy vs Steps')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Latency plot
        axes[0, 1].plot(steps, latencies, marker='o', color='orange')
        axes[0, 1].set_title('Latency vs Steps')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Latency (ms)')
        axes[0, 1].grid(True)
        
        # Memory ratio plot
        axes[1, 0].plot(steps, memory_ratios, marker='o', color='green')
        axes[1, 0].set_title('Memory Ratio vs Steps')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Memory Ratio')
        axes[1, 0].grid(True)
        
        # Pareto front plot
        axes[1, 1].scatter(latencies, memory_ratios, c=accuracies, cmap='viridis')
        axes[1, 1].set_title('Pareto Front (Latency vs Memory)')
        axes[1, 1].set_xlabel('Latency (ms)')
        axes[1, 1].set_ylabel('Memory Ratio')
        axes[1, 1].grid(True)
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Accuracy')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'pruning_metrics.png')
        plt.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pruning results"""
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        best_accuracy = max(m.accuracy.accuracy for m in self.metrics_history)
        
        return {
            'final_metrics': final_metrics.to_dict(),
            'best_accuracy': best_accuracy,
            'compression_ratio': final_metrics.model_size.compression_ratio(self.original_size),
            'latency_improvement': 1 - (final_metrics.latency.avg_latency / 
                                     self.metrics_history[0].latency.avg_latency),
            'num_pruned_groups': len(final_metrics.pruned_groups)
        }
    
    def export_results(self, filename: str):
        """Export complete results to file"""
        results = {
            'summary': self.get_summary(),
            'metrics_history': [m.to_dict() for m in self.metrics_history]
        }
        
        with open(self.save_dir / filename, 'w') as f:
            json.dump(results, f, indent=2)