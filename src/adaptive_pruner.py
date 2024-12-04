# layer_progressive_pruner.py

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
from src.dependency_graph import PruningUnit
from src.metrics import ModelMetrics, MetricsTracker

logger = logging.getLogger(__name__)

@dataclass
class LayerPruningQuota:
    """Tracks pruning quota and status for each layer"""
    layer_idx: int
    total_units: int
    max_prune_ratio: float  # Maximum ratio of units that can be pruned
    units_pruned: int = 0
    importance_score: float = 0.0  # Layer importance score
    is_frozen: bool = False  # Whether to stop pruning this layer

@dataclass
class PruningResult:
    """Stores results of pruning process"""
    pruned_units: List[str]
    memory_reduction: float  # In MB
    accuracy: float
    performance_impact: float
    layer_stats: Dict[str, Any]  # Per-layer statistics

class LayerProgressivePruner:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        initial_metrics: ModelMetrics,
        metrics_tracker: MetricsTracker,
        eval_dataloader: torch.utils.data.DataLoader,
        tokenizer: Any
    ):
        self.model = model
        self.config = config
        self.device = device
        self.initial_metrics = initial_metrics
        self.metrics_tracker = metrics_tracker
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        
        # Extract pruning targets
        self.target_memory = initial_metrics.memory_footprint['gpu_allocated'] * \
                           config['pruning']['targets']['compression_target']
        self.min_accuracy = initial_metrics.accuracy * \
                          config['pruning']['targets']['min_accuracy_ratio']
        self.max_accuracy_drop = 1 - config['pruning']['targets']['min_accuracy_ratio']
        self.units_per_step = config['pruning']['targets']['units_per_step']
        
        # Layer-wise pruning parameters
        self.layer_pruning_config = {
            'early_layers_ratio': 0.3,  # Prune less from early layers
            'middle_layers_ratio': 0.7,  # Prune more from middle layers
            'final_layers_ratio': 0.4    # Moderate pruning for final layers
        }
        
        self.saved_states = {}
        
        logger.info(
            f"Initialized layer-wise progressive pruner:\n"
            f"Initial accuracy: {initial_metrics.accuracy:.4f}\n"
            f"Initial memory: {initial_metrics.memory_footprint['gpu_allocated']:.2f} MB\n"
            f"Target memory: {self.target_memory:.2f} MB\n"
            f"Min accuracy: {self.min_accuracy:.4f}\n"
            f"Units per step: {self.units_per_step}"
        )

    def compute_layer_quotas(self, pruning_units: List[PruningUnit]) -> Dict[int, LayerPruningQuota]:
        """Establish layer-wise pruning quotas."""
        # Group units by layer
        layer_units = {}
        for unit in pruning_units:
            if unit.layer_idx not in layer_units:
                layer_units[unit.layer_idx] = []
            layer_units[unit.layer_idx].append(unit)
        
        total_layers = len(layer_units)
        quotas = {}
        
        # Compute layer importance scores
        for layer_idx, units in layer_units.items():
            avg_importance = np.mean([unit.importance_score for unit in units])
            
            # Determine layer position quota
            if layer_idx < total_layers * 0.3:
                max_ratio = self.layer_pruning_config['early_layers_ratio']
            elif layer_idx < total_layers * 0.7:
                max_ratio = self.layer_pruning_config['middle_layers_ratio']
            else:
                max_ratio = self.layer_pruning_config['final_layers_ratio']
            
            quotas[layer_idx] = LayerPruningQuota(
                layer_idx=layer_idx,
                total_units=len(units),
                max_prune_ratio=max_ratio,
                importance_score=avg_importance
            )
            
            logger.info(
                f"Layer {layer_idx}: {len(units)} units, "
                f"max prune ratio: {max_ratio:.2f}, "
                f"importance: {avg_importance:.4f}"
            )
        
        return quotas

    def optimize_pruning(self, pruning_units: List[PruningUnit]) -> PruningResult:
        """Execute layer-wise progressive pruning."""
        try:
            # Initialize layer quotas and group units
            layer_quotas = self.compute_layer_quotas(pruning_units)
            layer_units = {}
            for unit in pruning_units:
                if unit.layer_idx not in layer_units:
                    layer_units[unit.layer_idx] = []
                layer_units[unit.layer_idx].append(unit)
            
            # Sort units within each layer by importance
            for layer_idx in layer_units:
                layer_units[layer_idx].sort(key=lambda u: u.importance_score)
            
            pruned_units = []
            current_memory = self.initial_metrics.memory_footprint['gpu_allocated']
            memory_target = self.initial_metrics.memory_footprint['gpu_allocated'] - self.target_memory
            
            # Initialize statistics tracking
            pruning_stats = {
                'steps': 0,
                'total_pruned': 0,
                'layer_stats': {
                    layer_idx: {
                        'units_pruned': 0,
                        'accuracy_drops': [],
                        'memory_saved': 0.0,
                        'is_frozen': False
                    }
                    for layer_idx in layer_quotas
                }
            }
            
            # Setup progress bars
            with tqdm(total=memory_target, desc="Memory Reduction (MB)", unit='MB') as mem_pbar, \
                 tqdm(total=100, desc="Overall Progress", unit='%') as prog_pbar:
                
                last_memory = current_memory
                while current_memory > self.target_memory:
                    pruned_this_round = False
                    pruning_stats['steps'] += 1
                    
                    # Log layer status every 5 steps
                    if pruning_stats['steps'] % 5 == 0:
                        logger.info("\nLayer Status:")
                        for layer_idx, quota in layer_quotas.items():
                            logger.info(
                                f"Layer {layer_idx}: "
                                f"{pruning_stats['layer_stats'][layer_idx]['units_pruned']}/{int(quota.total_units * quota.max_prune_ratio)} units "
                                f"({'frozen' if pruning_stats['layer_stats'][layer_idx]['is_frozen'] else 'active'})"
                            )
                    
                    # Try to prune from each unfrozen layer
                    for layer_idx, quota in layer_quotas.items():
                        if pruning_stats['layer_stats'][layer_idx]['is_frozen']:
                            continue
                        
                        if pruning_stats['layer_stats'][layer_idx]['units_pruned'] >= int(quota.total_units * quota.max_prune_ratio):
                            pruning_stats['layer_stats'][layer_idx]['is_frozen'] = True
                            continue
                        
                        # Get candidate units
                        remaining_units = [
                            u for u in layer_units[layer_idx] 
                            if u.id not in pruned_units
                        ][:self.units_per_step]
                        
                        if not remaining_units:
                            pruning_stats['layer_stats'][layer_idx]['is_frozen'] = True
                            continue
                        
                        # Try pruning
                        for unit in remaining_units:
                            self._save_unit_state(unit)
                            self._zero_out_unit(unit)
                        
                        # Evaluate
                        metrics = self.metrics_tracker.evaluate_model(self.model, self.tokenizer)
                        accuracy_drop = self.initial_metrics.accuracy - metrics.accuracy
                        
                        if accuracy_drop > self.max_accuracy_drop or metrics.accuracy < self.min_accuracy:
                            # Restore and freeze layer
                            logger.info(
                                f"Layer {layer_idx}: Accuracy drop {accuracy_drop:.4f} too high. "
                                f"Freezing layer."
                            )
                            for unit in remaining_units:
                                self._restore_unit_state(unit)
                            pruning_stats['layer_stats'][layer_idx]['is_frozen'] = True
                        else:
                            # Accept pruning
                            memory_saved = sum(
                                sum(p.numel() * p.element_size() for p in u.parameters.values())
                                for u in remaining_units
                            ) / (1024 * 1024)  # MB
                            
                            current_memory -= memory_saved
                            pruned_units.extend(u.id for u in remaining_units)
                            pruning_stats['total_pruned'] += len(remaining_units)
                            pruning_stats['layer_stats'][layer_idx]['units_pruned'] += len(remaining_units)
                            pruning_stats['layer_stats'][layer_idx]['accuracy_drops'].append(float(accuracy_drop))
                            pruning_stats['layer_stats'][layer_idx]['memory_saved'] += float(memory_saved)
                            pruned_this_round = True
                            
                            # Update progress bars
                            mem_pbar.update(last_memory - current_memory)
                            progress = (1 - (current_memory - self.target_memory) / 
                                      (self.initial_metrics.memory_footprint['gpu_allocated'] - self.target_memory)) * 100
                            prog_pbar.update(progress - prog_pbar.n)
                            
                            # Update progress bar postfix
                            prog_pbar.set_postfix({
                                'acc': f"{metrics.accuracy:.4f}",
                                'mem_red': f"{(1 - current_memory/self.initial_metrics.memory_footprint['gpu_allocated'])*100:.1f}%",
                                'active_layers': sum(1 for stats in pruning_stats['layer_stats'].values() if not stats['is_frozen'])
                            })
                            
                            last_memory = current_memory
                            
                            logger.info(
                                f"Layer {layer_idx}: Pruned {len(remaining_units)} units. "
                                f"Accuracy: {metrics.accuracy:.4f}, Memory: {current_memory:.2f} MB"
                            )
                    
                    if not pruned_this_round:
                        logger.info("No more layers can be pruned. Stopping.")
                        break
            
            # Final evaluation
            final_metrics = self.metrics_tracker.evaluate_model(self.model, self.tokenizer)
            
            # Log final layer-wise statistics
            logger.info("\nFinal Layer Statistics:")
            for layer_idx, stats in pruning_stats['layer_stats'].items():
                logger.info(
                    f"Layer {layer_idx}: "
                    f"Pruned {stats['units_pruned']} units, "
                    f"Saved {stats['memory_saved']:.2f} MB"
                )
            
            return PruningResult(
                pruned_units=pruned_units,
                memory_reduction=self.initial_metrics.memory_footprint['gpu_allocated'] - 
                               final_metrics.memory_footprint['gpu_allocated'],
                accuracy=final_metrics.accuracy,
                performance_impact=(self.initial_metrics.accuracy - final_metrics.accuracy) / 
                                 self.initial_metrics.accuracy,
                layer_stats=pruning_stats
            )
            
        except Exception as e:
            logger.error(f"Error during pruning optimization: {str(e)}")
            raise
    
    def _save_unit_state(self, unit: PruningUnit):
        """Save parameter state of a unit."""
        self.saved_states[unit.id] = {
            name: param.data.clone() 
            for name, param in unit.parameters.items()
        }
    
    def _restore_unit_state(self, unit: PruningUnit):
        """Restore parameter state of a unit."""
        if unit.id in self.saved_states:
            for name, param in unit.parameters.items():
                param.data.copy_(self.saved_states[unit.id][name])
            del self.saved_states[unit.id]
    
    def _zero_out_unit(self, unit: PruningUnit):
        """Zero out parameters of a unit."""
        for param in unit.parameters.values():
            param.data.zero_()