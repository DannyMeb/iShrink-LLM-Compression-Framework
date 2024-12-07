# adaptive_pruner.py

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict
from src.dependency_graph import PruningUnit
from src.metrics import ModelMetrics, MetricsTracker

logger = logging.getLogger(__name__)

@dataclass
class LayerPruningStats:
    """Detailed statistics for each layer's pruning"""
    total_units_pruned: int
    attention_units_pruned: int
    mlp_units_pruned: int
    memory_saved: float
    total_attention_units: int
    total_mlp_units: int

@dataclass
class PruningBatchStats:
    """Statistics for a pruning batch"""
    batch_number: int
    units_pruned: int
    accuracy: float
    memory_reduction: float
    layer_distribution: Dict[int, LayerPruningStats]  # layer_idx -> stats
    remaining_memory: float
    accuracy_drop: float

@dataclass
class PruningResult:
    """Stores results of pruning process"""
    pruned_units: List[str]
    memory_reduction: float  # In MB
    accuracy: float
    performance_impact: float
    layer_stats: Dict[int, LayerPruningStats]  # Per-layer statistics
    batch_stats: List[PruningBatchStats]  # Stats for each batch
    final_metrics: ModelMetrics

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
        
        self.saved_states = {}
        
        logger.info(
            f"Initialized batch-wise pruner:\n"
            f"Initial accuracy: {initial_metrics.accuracy:.4f}\n"
            f"Initial memory: {initial_metrics.memory_footprint['gpu_allocated']:.2f} MB\n"
            f"Target memory: {self.target_memory:.2f} MB\n"
            f"Min accuracy: {self.min_accuracy:.4f}"
        )

    def _count_layer_units(self, pruning_units: List[PruningUnit]) -> Dict[int, Tuple[int, int]]:
        """Count total attention and MLP units per layer"""
        layer_counts = defaultdict(lambda: [0, 0])  # [attention_units, mlp_units]
        for unit in pruning_units:
            layer_idx = unit.layer_idx
            is_attention = 'attn' in unit.id
            layer_counts[layer_idx][0 if is_attention else 1] += 1
        return {k: tuple(v) for k, v in layer_counts.items()}

    def _calculate_memory_per_unit(self, unit: PruningUnit) -> float:
        """Calculate memory usage of a single unit in MB"""
        total_params = sum(p.numel() for p in unit.parameters.values())
        return total_params * 2 / (1024 * 1024)  # 2 bytes per parameter for float16

    def _estimate_required_units(self, pruning_units: List[PruningUnit]) -> int:
        """Estimate number of units needed to reach target compression"""
        current_memory = self.initial_metrics.memory_footprint['gpu_allocated']
        target_reduction = current_memory - self.target_memory
        
        units_with_memory = [(unit, self._calculate_memory_per_unit(unit)) 
                           for unit in pruning_units]
        units_with_memory.sort(key=lambda x: x[0].importance_score)
        
        cumulative_reduction = 0
        units_needed = 0
        
        for _, unit_memory in units_with_memory:
            cumulative_reduction += unit_memory
            units_needed += 1
            if cumulative_reduction >= target_reduction:
                break
        
        logger.info(f"Estimated {units_needed} units needed to reach "
                   f"target memory reduction of {target_reduction:.2f} MB")
        return units_needed

    def _prepare_pruning_batches(self, pruning_units: List[PruningUnit], 
                               total_units: int) -> List[List[PruningUnit]]:
        """Prepare batches of units for pruning"""
        sorted_units = sorted(pruning_units, key=lambda x: x.importance_score)
        units_to_prune = sorted_units[:total_units]
        
        batch_size = (total_units + 3) // 4  # Ceil division
        batches = [units_to_prune[i:i + batch_size] 
                  for i in range(0, len(units_to_prune), batch_size)]
        
        return batches

    def _save_unit_state(self, unit: PruningUnit):
        """Save parameter state of a unit"""
        self.saved_states[unit.id] = {
            name: param.data.clone() 
            for name, param in unit.parameters.items()
        }
    
    def _restore_unit_state(self, unit: PruningUnit):
        """Restore parameter state of a unit"""
        if unit.id in self.saved_states:
            for name, param in unit.parameters.items():
                param.data.copy_(self.saved_states[unit.id][name])
            del self.saved_states[unit.id]
    
    def _zero_out_unit(self, unit: PruningUnit):
        """Zero out parameters of a unit"""
        for param in unit.parameters.values():
            param.data.zero_()

    def _get_layer_distribution(self, units: List[PruningUnit], 
                              total_units: Dict[int, Tuple[int, int]], 
                              layer_stats: Dict[int, LayerPruningStats]) -> Dict[int, LayerPruningStats]:
        """Get distribution of pruned units across layers"""
        distribution = defaultdict(lambda: {"attn": 0, "mlp": 0, "memory": 0.0})
        
        # Count new units being pruned
        for unit in units:
            layer_idx = unit.layer_idx
            is_attention = 'attn' in unit.id
            memory = self._calculate_memory_per_unit(unit)
            
            distribution[layer_idx]["attn" if is_attention else "mlp"] += 1
            distribution[layer_idx]["memory"] += memory
        
        # Create updated LayerPruningStats
        result = {}
        for layer_idx, counts in distribution.items():
            total_attn, total_mlp = total_units.get(layer_idx, (0, 0))
            prev_stats = layer_stats.get(layer_idx, LayerPruningStats(0, 0, 0, 0.0, total_attn, total_mlp))
            
            result[layer_idx] = LayerPruningStats(
                total_units_pruned=prev_stats.total_units_pruned + counts["attn"] + counts["mlp"],
                attention_units_pruned=prev_stats.attention_units_pruned + counts["attn"],
                mlp_units_pruned=prev_stats.mlp_units_pruned + counts["mlp"],
                memory_saved=prev_stats.memory_saved + counts["memory"],
                total_attention_units=total_attn,
                total_mlp_units=total_mlp
            )
            
        return result

    
    def optimize_pruning(self, pruning_units: List[PruningUnit]) -> PruningResult:
        """Execute batch-wise progressive pruning"""
        try:
            def verify_pruned_units(pruning_units: List[PruningUnit], verbose: bool = True) -> bool:
                """
                Verify that pruned units are actually zeroed out
                
                Args:
                    pruning_units: List of pruned units to verify
                    verbose: Whether to print detailed information about non-zero parameters
                
                Returns:
                    bool: True if all units are properly zeroed out, False otherwise
                """
                all_zeroed = True
                
                for unit in pruning_units:
                    unit_zeroed = True
                    for name, (param, slice_idx) in unit.param_references.items():
                        # Handle different slice types
                        if isinstance(slice_idx, tuple):
                            param_slice = param.data[slice_idx[0], slice_idx[1]]
                        else:
                            param_slice = param.data[slice_idx]
                        
                        # Check if all parameters are zero
                        if not torch.all(param_slice == 0):
                            all_zeroed = False
                            unit_zeroed = False
                            if verbose:
                                non_zero_count = torch.count_nonzero(param_slice).item()
                                total_params = param_slice.numel()
                                logger.error(
                                    f"Unit {unit.id} parameter '{name}' not fully zeroed!\n"
                                    f"Non-zero parameters: {non_zero_count}/{total_params} "
                                    f"({non_zero_count/total_params*100:.2f}%)"
                                )
                    
                    if verbose and unit_zeroed:
                        logger.info(f"Unit {unit.id} successfully verified - all parameters zeroed")
                
                return all_zeroed

            # Calculate total units needed and count units per layer
            total_units_needed = self._estimate_required_units(pruning_units)
            total_units_per_layer = self._count_layer_units(pruning_units)
            
            # Prepare pruning batches
            pruning_batches = self._prepare_pruning_batches(pruning_units, total_units_needed)
            
            pruned_units = []
            batch_stats = []
            current_memory = self.initial_metrics.memory_footprint['gpu_allocated']
            layer_stats = {}  # Will be populated with LayerPruningStats
            
            logger.info(f"Starting batch pruning with {len(pruning_batches)} batches "
                      f"of approximately {len(pruning_batches[0])} units each")
            
            # Process each batch
            for batch_idx, batch in enumerate(pruning_batches, 1):
                logger.info(f"\nProcessing batch {batch_idx}/{len(pruning_batches)}")
                
                # Zero out all units in batch
                for unit in batch:
                    self._save_unit_state(unit)
                    self._zero_out_unit(unit)
                

                # Verify pruning was successful
                verify_result = verify_pruned_units(batch)
                if not verify_result:
                    raise RuntimeError("Pruning verification failed - parameters not properly zeroed!")
                
                # Evaluate model with batch pruned
                metrics = self.metrics_tracker.evaluate_model(self.model, self.tokenizer, verbose=False)
                accuracy_drop = self.initial_metrics.accuracy - metrics.accuracy
                
                if accuracy_drop > self.max_accuracy_drop or metrics.accuracy < self.min_accuracy:
                    logger.info(f"Accuracy drop ({accuracy_drop:.4f}) exceeded threshold. "
                              f"Restoring batch and stopping.")
                    
                    # Restore this batch
                    for unit in batch:
                        self._restore_unit_state(unit)
                    break
                    
                # Update memory tracking
                memory_reduction = sum(self._calculate_memory_per_unit(unit) for unit in batch)
                current_memory -= memory_reduction
                
                # Update layer statistics
                layer_stats = self._get_layer_distribution(batch, total_units_per_layer, layer_stats)
                
                # Record batch statistics
                batch_stats.append(PruningBatchStats(
                    batch_number=batch_idx,
                    units_pruned=len(batch),
                    accuracy=metrics.accuracy,
                    memory_reduction=memory_reduction,
                    layer_distribution=layer_stats.copy(),
                    remaining_memory=current_memory,
                    accuracy_drop=accuracy_drop
                ))
                
                pruned_units.extend(unit.id for unit in batch)
                
                # Log progress
                logger.info(f"Batch {batch_idx} stats:")
                logger.info(f"Accuracy: {metrics.accuracy:.4f} (drop: {accuracy_drop:.4f})")
                logger.info(f"Memory reduction: {memory_reduction:.2f} MB")
                logger.info("Layer distribution:")
                for layer_idx, stats in sorted(layer_stats.items()):
                    logger.info(
                        f"Layer {layer_idx}: "
                        f"Attention {stats.attention_units_pruned}/{stats.total_attention_units}, "
                        f"MLP {stats.mlp_units_pruned}/{stats.total_mlp_units}"
                    )
            
            # Final evaluation
            final_metrics = self.metrics_tracker.evaluate_model(self.model, self.tokenizer)
            
            return PruningResult(
                pruned_units=pruned_units,
                memory_reduction=self.initial_metrics.memory_footprint['gpu_allocated'] - 
                               current_memory,
                accuracy=final_metrics.accuracy,
                performance_impact=(self.initial_metrics.accuracy - final_metrics.accuracy) / 
                                 self.initial_metrics.accuracy,
                layer_stats=layer_stats,
                batch_stats=batch_stats,
                final_metrics=final_metrics
            )
            
        except Exception as e:
            logger.error(f"Error during pruning optimization: {str(e)}")
            raise