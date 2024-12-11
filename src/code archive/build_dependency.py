#build_dependency.py

import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LayerDependency:
    """Tracks dependencies between transformer layers"""
    layer_idx: int
    input_layers: Set[int]    # Layers that feed into this layer
    output_layers: Set[int]   # Layers that this layer feeds into
    residual_connections: Set[int]  # Layers connected via residual paths
    attention_dependencies: Dict[int, List[int]]  # Maps attention heads to dependent heads
    mlp_dependencies: Dict[int, List[int]]  # Maps MLP units to dependent units

@dataclass
class DependencyGroup:
    """Group of units that should be pruned together"""
    layer_idx: int
    units: List[Any]  # Pruning units in this group
    attention_heads: Set[int]  # Attention heads in group
    mlp_units: Set[int]  # MLP units in group
    connected_layers: Set[int]  # Other affected layers

class DependencyBuilder:
    """Builds dependency information for transformer model pruning"""
    
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.num_layers = len(model.model.layers)
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.q_per_kv = self.num_heads // self.num_kv_heads
        self.intermediate_size = model.config.intermediate_size
        self.mlp_group_size = config.get('pruning', {}).get('dependency', {}).get('mlp_group_size', 8)
        
        # Initialize base dependencies
        self.layer_dependencies = self._initialize_layer_dependencies()

    def _initialize_layer_dependencies(self) -> Dict[int, LayerDependency]:
        """Initialize the basic dependency structure between layers"""
        dependencies = {}
        
        for layer_idx in range(self.num_layers):
            # Standard feed-forward connections
            input_layers = {layer_idx - 1} if layer_idx > 0 else set()
            output_layers = {layer_idx + 1} if layer_idx < self.num_layers - 1 else set()
            
            # Add residual connections (typically span 2-3 layers)
            residual_span = min(3, self.num_layers - layer_idx - 1)
            residual_connections = {layer_idx + i for i in range(1, residual_span + 1)}
            
            # Initialize attention head dependencies (for GQA)
            attention_dependencies = {}
            for head_idx in range(self.num_kv_heads):
                q_start = head_idx * self.q_per_kv
                q_end = (head_idx + 1) * self.q_per_kv
                attention_dependencies[head_idx] = list(range(q_start, q_end))
            
            # Initialize MLP unit dependencies
            mlp_dependencies = {}
            mlp_units = self.intermediate_size // self.mlp_group_size
            for unit_idx in range(mlp_units):
                # Consider adjacent units as dependent
                dependent_units = [i for i in range(max(0, unit_idx - 1), 
                                                  min(mlp_units, unit_idx + 2))]
                mlp_dependencies[unit_idx] = dependent_units
            
            dependencies[layer_idx] = LayerDependency(
                layer_idx=layer_idx,
                input_layers=input_layers,
                output_layers=output_layers,
                residual_connections=residual_connections,
                attention_dependencies=attention_dependencies,
                mlp_dependencies=mlp_dependencies
            )
        
        return dependencies

    def create_dependency_groups(self, pruning_units: List[Any]) -> List[DependencyGroup]:
        """Create groups of units that should be pruned together"""
        dependency_groups = []
        processed_layers = set()
        
        for layer_idx in range(self.num_layers):
            if layer_idx in processed_layers:
                continue
            
            # Get layer dependencies
            dep = self.layer_dependencies[layer_idx]
            
            # Create group with all connected layers
            connected_layers = dep.input_layers.union(
                dep.output_layers, 
                dep.residual_connections
            )
            
            # Get units for this group
            group_units = []
            attention_heads = set()
            mlp_units = set()
            
            for unit in pruning_units:
                if unit.layer_idx == layer_idx or unit.layer_idx in connected_layers:
                    group_units.append(unit)
                    if 'attn' in unit.id:
                        attention_heads.add((unit.layer_idx, unit.head_idx))
                    else:
                        mlp_units.add((unit.layer_idx, unit.head_idx))
            
            dependency_groups.append(DependencyGroup(
                layer_idx=layer_idx,
                units=group_units,
                attention_heads=attention_heads,
                mlp_units=mlp_units,
                connected_layers=connected_layers
            ))
            
            processed_layers.add(layer_idx)
            processed_layers.update(connected_layers)
        
        return dependency_groups

    def get_attention_dependencies(self, layer_idx: int, head_idx: int) -> List[int]:
        """Get dependent attention heads for a given head"""
        if layer_idx not in self.layer_dependencies:
            return []
        return self.layer_dependencies[layer_idx].attention_dependencies.get(head_idx, [])

    def get_mlp_dependencies(self, layer_idx: int, unit_idx: int) -> List[int]:
        """Get dependent MLP units for a given unit"""
        if layer_idx not in self.layer_dependencies:
            return []
        return self.layer_dependencies[layer_idx].mlp_dependencies.get(unit_idx, [])
