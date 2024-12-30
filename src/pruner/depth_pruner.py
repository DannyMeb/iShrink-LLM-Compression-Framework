# depth_pruner.py

import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

@dataclass
class DepthPruningResult:
    """Stores results of depth pruning"""
    original_layers: int               # Number of layers before pruning
    pruned_layers: int                # Number of layers after pruning
    kept_layer_indices: List[int]     # Indices of kept layers
    layer_importance_scores: Dict[int, float]  # Importance score per layer
    pruned_model: torch.nn.Module     # The pruned model
    params_before: int                # Total parameters before pruning
    params_after: int                 # Total parameters after pruning
    compression_ratio: float          # Achieved compression ratio

class DepthPruner:
    def __init__(
        self,
        model: torch.nn.Module,
        depth_sparsity: float,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize depth pruner.
        
        Args:
            model: The model to prune
            depth_sparsity: Fraction of layers to remove (0-1)
            config: Optional configuration dictionary
        """
        if not 0 <= depth_sparsity <= 1:
            raise ValueError("Depth sparsity must be between 0 and 1")
            
        self.model = model
        self.depth_sparsity = depth_sparsity
        self.config = config or {}
        
        # Get model dimensions
        self.num_layers = len(model.model.layers)
        self.new_num_layers = max(
            self.config.get('min_layers', 1),
            int(self.num_layers * (1 - depth_sparsity))
        )
        
        # Get pruning configuration
        self.layer_group_size = self.config.get('layer_group_size', 1)
        self.recompute_importance = self.config.get('recompute_importance', True)
        self.preserve_layers = set(self.config.get('preserve_layers', []))
        self.importance_method = self.config.get('importance_method', 'combined')
        self.granularity = self.config.get('granularity', 'block')
        
        self._validate_config()
        self._log_initialization()

    def _validate_config(self):
        """Validate pruning configuration"""
        if self.granularity not in ['block', 'individual']:
            raise ValueError(f"Invalid granularity: {self.granularity}")
            
        if self.importance_method not in ['taylor', 'gradient', 'combined']:
            raise ValueError(f"Invalid importance method: {self.importance_method}")
            
        if any(idx >= self.num_layers for idx in self.preserve_layers):
            raise ValueError("Invalid layer index in preserve_layers")
            
        if self.layer_group_size > 1 and self.granularity != 'block':
            logger.warning("Layer grouping is only applied when granularity='block'")

    def _log_initialization(self):
        """Log initialization details"""
        logger.info("\n=== Depth Pruner Configuration ===")
        logger.info(f"Original layers: {self.num_layers}")
        logger.info(f"Target layers: {self.new_num_layers}")
        logger.info(f"Depth sparsity: {self.depth_sparsity:.2%}")
        logger.info(f"Granularity: {self.granularity}")
        logger.info(f"Layer group size: {self.layer_group_size}")
        logger.info(f"Preserved layers: {sorted(self.preserve_layers)}")
        logger.info(f"Importance method: {self.importance_method}")

    def calculate_layer_importance(self, pruning_units: List[Any]) -> Dict[int, float]:
        """Calculate importance score for each layer"""
        layer_importance = {i: 0.0 for i in range(self.num_layers)}
        
        # Group units by layer
        layer_units = {i: [] for i in range(self.num_layers)}
        for unit in pruning_units:
            layer_units[unit.layer_idx].append(unit)
        
        # Calculate layer scores based on method
        if self.importance_method == 'taylor':
            for layer_idx, units in layer_units.items():
                if units:
                    layer_importance[layer_idx] = np.mean([
                        unit.taylor_score for unit in units 
                        if hasattr(unit, 'taylor_score')
                    ])
        
        elif self.importance_method == 'gradient':
            for layer_idx, units in layer_units.items():
                if units:
                    layer_importance[layer_idx] = np.mean([
                        unit.gradient_norm for unit in units 
                        if hasattr(unit, 'gradient_norm')
                    ])
        
        else:  # combined
            for layer_idx, units in layer_units.items():
                if units:
                    layer_importance[layer_idx] = np.mean([
                        unit.importance_score for unit in units
                    ])
        
        # Handle missing scores
        max_score = max(layer_importance.values())
        if max_score > 0:
            layer_importance = {k: v/max_score for k, v in layer_importance.items()}
        
        return layer_importance

    def _handle_block_pruning(self, layer_importance: Dict[int, float]) -> List[int]:
        """Handle pruning when operating on blocks of layers"""
        if self.layer_group_size > 1:
            # Average importance scores within blocks
            block_importance = {}
            for i in range(0, self.num_layers, self.layer_group_size):
                block_idx = i // self.layer_group_size
                block_scores = [
                    layer_importance[j] 
                    for j in range(i, min(i + self.layer_group_size, self.num_layers))
                ]
                block_importance[block_idx] = np.mean(block_scores)
            
            # Select blocks to keep
            num_blocks = len(block_importance)
            blocks_to_keep = max(1, int(num_blocks * (1 - self.depth_sparsity)))
            kept_blocks = sorted(
                block_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:blocks_to_keep]
            
            # Convert back to layer indices
            kept_layers = []
            for block_idx, _ in kept_blocks:
                start_idx = block_idx * self.layer_group_size
                end_idx = min((block_idx + 1) * self.layer_group_size, self.num_layers)
                kept_layers.extend(range(start_idx, end_idx))
            
        else:
            # Sort layers by importance
            layer_scores = sorted(
                layer_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            kept_layers = [idx for idx, _ in layer_scores[:self.new_num_layers]]
        
        return sorted(kept_layers)

    def _handle_individual_pruning(self, layer_importance: Dict[int, float]) -> List[int]:
        """Handle pruning when operating on individual layers"""
        # Add preserved layers
        kept_layers = list(self.preserve_layers)
        remaining_slots = self.new_num_layers - len(kept_layers)
        
        if remaining_slots > 0:
            # Sort remaining layers by importance
            remaining_layers = [
                (idx, score) for idx, score in layer_importance.items()
                if idx not in self.preserve_layers
            ]
            remaining_layers.sort(key=lambda x: x[1], reverse=True)
            
            # Add top remaining layers
            kept_layers.extend(idx for idx, _ in remaining_layers[:remaining_slots])
        
        return sorted(kept_layers)

    def select_layers_to_keep(self, layer_importance: Dict[int, float]) -> List[int]:
        """Select which layers to keep based on importance scores"""
        if self.granularity == 'block':
            kept_layers = self._handle_block_pruning(layer_importance)
        else:
            kept_layers = self._handle_individual_pruning(layer_importance)
        
        # Ensure minimum layers are kept
        if len(kept_layers) < self.config.get('min_layers', 1):
            logger.warning(f"Too few layers selected ({len(kept_layers)}), "
                         f"adding more to meet minimum requirement")
            missing_layers = set(range(self.num_layers)) - set(kept_layers)
            additional_needed = self.config.get('min_layers', 1) - len(kept_layers)
            kept_layers.extend(sorted(list(missing_layers))[:additional_needed])
            kept_layers.sort()
        
        return kept_layers

    def prune_layers(self, pruning_units: List[Any]) -> DepthPruningResult:
        """Execute depth pruning"""
        try:
            params_before = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Initial parameter count: {params_before:,}")
            
            # Calculate layer importance and select layers to keep
            layer_importance = self.calculate_layer_importance(pruning_units)
            kept_layer_indices = self.select_layers_to_keep(layer_importance)
            
            # Create pruned model
            pruned_model = deepcopy(self.model)
            
            # Create new layers list with only kept layers
            new_layers = torch.nn.ModuleList([
                pruned_model.model.layers[i] for i in kept_layer_indices
            ])
            
            # Update model
            pruned_model.model.layers = new_layers
            pruned_model.config.num_hidden_layers = len(kept_layer_indices)
            
            # Calculate compression
            params_after = sum(p.numel() for p in pruned_model.parameters())
            compression_ratio = (params_before - params_after) / params_before
            
            # Log results
            logger.info(f"\nKept layers: {kept_layer_indices}")
            logger.info(f"Parameters before: {params_before:,}")
            logger.info(f"Parameters after: {params_after:,}")
            logger.info(f"Compression ratio: {compression_ratio:.2%}")
            
            return DepthPruningResult(
                original_layers=self.num_layers,
                pruned_layers=len(kept_layer_indices),
                kept_layer_indices=kept_layer_indices,
                layer_importance_scores=layer_importance,
                pruned_model=pruned_model,
                params_before=params_before,
                params_after=params_after,
                compression_ratio=compression_ratio
            )
            
        except Exception as e:
            logger.error(f"Error during depth pruning: {str(e)}")
            raise

    def verify_pruned_layers(self, pruned_model: torch.nn.Module, expected_layers: int):
        """Verify pruned model has correct number of layers"""
        actual_layers = len(pruned_model.model.layers)
        config_layers = pruned_model.config.num_hidden_layers
        
        assert actual_layers == expected_layers, \
            f"Expected {expected_layers} layers, got {actual_layers}"
        assert config_layers == expected_layers, \
            f"Config shows {config_layers} layers, expected {expected_layers}"
            
        logger.info("Layer verification successful")