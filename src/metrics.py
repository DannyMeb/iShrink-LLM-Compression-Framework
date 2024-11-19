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
import wandb

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
            'memory_ratio': self.total_memory / original_size.total_memory,
            'layer_ratios': {
                layer: self.memory_per_layer.get(layer, 0) / orig_mem
                for layer, orig_mem in original_size.memory_per_layer.items()
            }
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
    batch_latencies: List[float] = field(default_factory=list)
    
    @classmethod
    def from_measurements(cls, 
                         latencies: List[float], 
                         batch_size: int) -> 'LatencyMetrics':
        """Create metrics from latency measurements"""
        return cls(
            avg_latency=np.mean(latencies),
            std_latency=np.std(latencies),
            p90_latency=np.percentile(latencies, 90),
            p95_latency=np.percentile(latencies, 95),
            p99_latency=np.percentile(latencies, 99),
            throughput=batch_size / (np.mean(latencies) / 1000),  # Convert ms to seconds
            batch_latencies=latencies
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
    predictions: List[int] = field(default_factory=list)
    targets: List[int] = field(default_factory=list)

@dataclass
class PruningMetrics:
    """Comprehensive pruning metrics"""
    step: int
    model_size: ModelSize
    latency: LatencyMetrics
    accuracy: AccuracyMetrics
    pruned_groups: List[str]
    compression_ratio: float
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
            'pruned_groups': self.pruned_groups,
            'compression_ratio': self.compression_ratio
        }

class MetricsTracker:
    """Tracks and manages pruning metrics"""
    
    def __init__(self, 
                 save_dir: str,
                 original_model: torch.nn.Module,
                 device: str = 'cuda',
                 num_warmup_steps: int = 10,
                 use_wandb: bool = False):
        """Initialize metrics tracker"""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.num_warmup_steps = num_warmup_steps
        self.metrics_history: List[PruningMetrics] = []
        self.use_wandb = use_wandb
        
        # Get original model metrics
        self.original_size = self._measure_model_size(original_model)
        
        # Initialize plots directory
        self.plots_dir = self.save_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized MetricsTracker. Original model size: "
                   f"{self.original_size.total_params:,} parameters")
    
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
        
        # Calculate compression ratio
        compression_ratio = model_size.total_memory / self.original_size.total_memory
        
        # Create metrics object
        metrics = PruningMetrics(
            step=step,
            model_size=model_size,
            latency=latency,
            accuracy=accuracy,
            pruned_groups=pruned_groups,
            compression_ratio=compression_ratio
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        # Log to wandb if enabled
        if self.use_wandb:
            self._log_to_wandb(metrics)
        
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
        targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, target_labels = batch[0].to(self.device), batch[1].to(self.device)
                outputs = model(inputs)
                
                # Compute accuracy
                predicted = outputs.logits.argmax(dim=-1)
                total_correct += (predicted == target_labels).sum().item()
                total_samples += target_labels.size(0)
                
                # Compute loss
                loss = outputs.loss
                total_loss += loss.item() * target_labels.size(0)
                
                # Store predictions for F1 score
                predictions.extend(predicted.cpu().numpy())
                targets.extend(target_labels.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted'
        )
        
        return AccuracyMetrics(
            accuracy=total_correct / total_samples,
            loss=total_loss / total_samples,
            f1_score=f1,
            precision=precision,
            recall=recall,
            predictions=predictions,
            targets=targets
        )
    
    def _save_metrics(self, metrics: PruningMetrics):
        """Save metrics to disk"""
        metrics_file = self.save_dir / f"metrics_step_{metrics.step}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def _log_to_wandb(self, metrics: PruningMetrics):
        """Log metrics to Weights & Biases"""
        wandb.log({
            'step': metrics.step,
            'accuracy': metrics.accuracy.accuracy,
            'loss': metrics.accuracy.loss,
            'f1_score': metrics.accuracy.f1_score,
            'latency': metrics.latency.avg_latency,
            'throughput': metrics.latency.throughput,
            'compression_ratio': metrics.compression_ratio,
            'memory_used': metrics.model_size.total_memory
        })
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """Plot metrics history"""
        save_dir = Path(save_dir) if save_dir else self.plots_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metrics for plotting
        steps = [m.step for m in self.metrics_history]
        accuracies = [m.accuracy.accuracy for m in self.metrics_history]
        latencies = [m.latency.avg_latency for m in self.metrics_history]
        compression_ratios = [m.compression_ratio for m in self.metrics_history]
        
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
        
        # Compression ratio plot
        axes[1, 0].plot(steps, compression_ratios, marker='o', color='green')
        axes[1, 0].set_title('Compression Ratio vs Steps')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].grid(True)
        
        # Pareto front plot
        scatter = axes[1, 1].scatter(latencies, compression_ratios, 
                                   c=accuracies, cmap='viridis')
        axes[1, 1].set_title('Pareto Front (Latency vs Compression)')
        axes[1, 1].set_xlabel('Latency (ms)')
        axes[1, 1].set_ylabel('Compression Ratio')
        axes[1, 1].grid(True)
        plt.colorbar(scatter, ax=axes[1, 1], label='Accuracy')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'pruning_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_layer_metrics(self):
        """Plot layer-wise metrics"""
        if not self.metrics_history:
            return
        
        latest_metrics = self.metrics_history[-1]
        layer_memories = latest_metrics.model_size.memory_per_layer
        original_memories = self.original_size.memory_per_layer
        
        # Create layer compression ratio plot
        plt.figure(figsize=(12, 6))
        layers = list(original_memories.keys())
        ratios = [layer_memories.get(layer, 0) / original_memories[layer] 
                 for layer in layers]
        
        plt.bar(layers, ratios)
        plt.title('Layer-wise Compression Ratios')
        plt.xlabel('Layers')
        plt.ylabel('Compression Ratio')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'layer_compression.png')
        plt.close()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pruning results"""
        if not self.metrics_history:
            return {}
        
        final_metrics = self.metrics_history[-1]
        best_accuracy = max(m.accuracy.accuracy for m in self.metrics_history)
        initial_latency = self.metrics_history[0].latency.avg_latency
        
        return {
            'final_metrics': final_metrics.to_dict(),
            'best_accuracy': best_accuracy,
            'compression_ratio': final_metrics.compression_ratio,
            'latency_improvement': 1 - (final_metrics.latency.avg_latency / initial_latency),
            'total_steps': len(self.metrics_history),
            'pruned_groups_count': len(final_metrics.pruned_groups),
            'memory_saved': self.original_size.total_memory - final_metrics.model_size.total_memory,
            'accuracy_drop': self.metrics_history[0].accuracy.accuracy - final_metrics.accuracy.accuracy
        }
    
    def export_results(self, filename: str):
        """Export complete results to file"""
        results = {
            'summary': self.get_summary(),
            'metrics_history': [m.to_dict() for m in self.metrics_history],
            'original_model_size': {
                'total_params': self.original_size.total_params,
                'total_memory': self.original_size.total_memory,
                'memory_per_layer': dict(self.original_size.memory_per_layer)
            }
        }
        
        output_path = self.save_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Exported results to {output_path}")
    
    def plot_pareto_frontier(self):
        """Plot Pareto frontier of accuracy vs compression vs latency"""
        if not self.metrics_history:
            return
        
        # Extract metrics
        accuracies = [m.accuracy.accuracy for m in self.metrics_history]
        compressions = [m.compression_ratio for m in self.metrics_history]
        latencies = [m.latency.avg_latency for m in self.metrics_history]
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(compressions, latencies, accuracies,
                           c=range(len(accuracies)), cmap='viridis')
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel('Latency (ms)')
        ax.set_zlabel('Accuracy')
        
        plt.colorbar(scatter, label='Step')
        plt.title('Pruning Pareto Frontier')
        plt.savefig(self.plots_dir / 'pareto_frontier.png')
        plt.close()
    
    def get_best_model_checkpoint(self) -> Tuple[int, PruningMetrics]:
        """Get the step and metrics for the best performing model"""
        if not self.metrics_history:
            return -1, None
        
        # Define weights for multi-objective optimization
        weights = {
            'accuracy': 0.6,
            'compression': 0.2,
            'latency': 0.2
        }
        
        best_score = float('-inf')
        best_step = -1
        best_metrics = None
        
        for metrics in self.metrics_history:
            # Normalize metrics
            accuracy_score = metrics.accuracy.accuracy
            compression_score = 1 - metrics.compression_ratio
            latency_score = 1 - (metrics.latency.avg_latency / 
                               self.metrics_history[0].latency.avg_latency)
            
            # Compute weighted score
            score = (weights['accuracy'] * accuracy_score +
                    weights['compression'] * compression_score +
                    weights['latency'] * latency_score)
            
            if score > best_score:
                best_score = score
                best_step = metrics.step
                best_metrics = metrics
        
        return best_step, best_metrics
    
    def generate_report(self, output_file: str):
        """Generate comprehensive pruning report"""
        summary = self.get_summary()
        best_step, best_metrics = self.get_best_model_checkpoint()
        
        report = {
            'summary': summary,
            'best_model': {
                'step': best_step,
                'metrics': best_metrics.to_dict() if best_metrics else None
            },
            'compression_analysis': {
                'layer_wise_compression': {
                    layer: {
                        'original_size': self.original_size.memory_per_layer[layer],
                        'final_size': self.metrics_history[-1].model_size.memory_per_layer[layer],
                        'compression_ratio': (self.metrics_history[-1].model_size.memory_per_layer[layer] /
                                           self.original_size.memory_per_layer[layer])
                    }
                    for layer in self.original_size.memory_per_layer.keys()
                },
                'total_compression': summary['compression_ratio']
            },
            'performance_analysis': {
                'latency': {
                    'initial': self.metrics_history[0].latency.avg_latency,
                    'final': self.metrics_history[-1].latency.avg_latency,
                    'improvement': summary['latency_improvement'],
                    'throughput_improvement': (self.metrics_history[-1].latency.throughput /
                                            self.metrics_history[0].latency.throughput)
                },
                'accuracy': {
                    'initial': self.metrics_history[0].accuracy.accuracy,
                    'final': self.metrics_history[-1].accuracy.accuracy,
                    'drop': summary['accuracy_drop'],
                    'best': summary['best_accuracy']
                }
            },
            'pruning_statistics': {
                'total_steps': summary['total_steps'],
                'pruned_groups': summary['pruned_groups_count'],
                'memory_saved_mb': summary['memory_saved'],
                'pruning_progression': [
                    {
                        'step': m.step,
                        'groups_pruned': len(m.pruned_groups),
                        'accuracy': m.accuracy.accuracy,
                        'compression': m.compression_ratio
                    }
                    for m in self.metrics_history
                ]
            }
        }
        
        # Save report
        output_path = self.save_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate plots
        self.plot_metrics()
        self.plot_layer_metrics()
        self.plot_pareto_frontier()
        
        logger.info(f"Generated comprehensive report at {output_path}")
        
        return report