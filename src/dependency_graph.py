#dependency_graph.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class PruningUnit:
    """Represents a prunable unit (attention head or MLP neurons group)"""
    def __init__(self, layer_idx: int, head_idx: int, param_references: Dict[str, Tuple[torch.nn.Parameter, slice]]):
        """
        Initialize a pruning unit with references to original parameters
        
        Args:
            layer_idx: Index of the transformer layer
            head_idx: Index of the attention head or MLP group
            param_references: Dict mapping parameter names to tuples of (parameter, slice)
                            where slice indicates which part of the parameter belongs to this unit
        """
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.param_references = param_references
        self.id = f"{'attn' if 'q_proj' in param_references else 'mlp'}_{layer_idx}_{head_idx}"
        self.importance_score = None
        
    def get_param_slice(self, name: str) -> Tuple[torch.nn.Parameter, slice]:
        """Get the original parameter and its relevant slice for this unit"""
        return self.param_references[name]
    
    @property
    def parameters(self) -> Dict[str, torch.nn.Parameter]:
        """Get all parameters associated with this unit (for compatibility)"""
        return {name: param for name, (param, _) in self.param_references.items()}

class DependencyGraphBuilder:
    """Builds pruning units from model architecture"""
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        
        # Get model configuration directly from the model
        self.num_layers = len(self.model.model.layers)
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.num_heads = self.model.config.num_attention_heads
        self.hidden_size = self.model.config.hidden_size
        self.head_dim = self.model.config.head_dim
        self.intermediate_size = self.model.config.intermediate_size
        self.mlp_group_size = config.get('mlp_group_size', 128)  # From our config
        
        logger.info("Initialized pruning units builder")
        
    def build(self, percent: float = 100.0) -> Tuple[List[PruningUnit], Dict[str, List[str]]]:
        """
        Build pruning units and their dependency graph
        
        Args:
            percent: Percentage of layers to include (0-100)
                    100 means all layers, 0 means no layers
        
        Returns:
            List of pruning units and dictionary of dependencies
        """
        try:
            transformer_layers = self.model.model.layers
            num_layers = len(transformer_layers)
            
            # Calculate how many layers to process based on percentage
            num_layers_to_process = int(round(num_layers * (percent / 100.0)))
            if num_layers_to_process == 0 and percent > 0:
                num_layers_to_process = 1  # Ensure at least one layer if percentage is not 0
            
            logger.info(f"Found {num_layers} transformer layers, processing {num_layers_to_process} layers ({percent}%)")
            
            pruning_units = []
            
            # Create pruning units for selected layers
            for layer_idx in range(num_layers_to_process):
                # Add attention head units
                attention_units = self._create_attention_units(layer_idx)
                pruning_units.extend(attention_units)
                
                # Add MLP units
                mlp_units = self._create_mlp_units(layer_idx)
                pruning_units.extend(mlp_units)
            
            logger.info(f"Created {len([u for u in pruning_units if 'attn' in u.id])} attention units "
                    f"and {len([u for u in pruning_units if 'mlp' in u.id])} MLP units")
            
            # Return units with empty dependencies
            return pruning_units, {}
            
        except Exception as e:
            logger.error(f"Error building pruning units: {str(e)}")
            raise
        
    def _create_attention_units(self, layer_idx: int) -> List[PruningUnit]:
        """Create pruning units for GQK attention groups in a layer"""
        layer = self.model.model.layers[layer_idx].self_attn
        head_dim = self.hidden_size // self.num_heads
        q_heads_per_kv = self.num_heads // self.num_kv_heads
        
        units = []
        for kv_idx in range(self.num_kv_heads):
            # Calculate indices
            q_start = kv_idx * q_heads_per_kv * head_dim
            q_end = (kv_idx + 1) * q_heads_per_kv * head_dim
            kv_start = kv_idx * head_dim
            kv_end = (kv_idx + 1) * head_dim
            
            # Create parameter references with slices
            param_references = {
                'q_proj': (layer.q_proj.weight, slice(q_start, q_end)),
                'k_proj': (layer.k_proj.weight, slice(kv_start, kv_end)),
                'v_proj': (layer.v_proj.weight, slice(kv_start, kv_end)),
                'o_proj': (layer.o_proj.weight, (slice(None), slice(q_start, q_end)))
            }
            
            units.append(PruningUnit(layer_idx, kv_idx, param_references))
            
        return units

    def _create_mlp_units(self, layer_idx: int) -> List[PruningUnit]:
        """Create pruning units for MLP groups in a layer"""
        layer = self.model.model.layers[layer_idx].mlp
        num_groups = self.intermediate_size // self.mlp_group_size
        
        units = []
        for group_idx in range(num_groups):
            start_idx = group_idx * self.mlp_group_size
            end_idx = min((group_idx + 1) * self.mlp_group_size, self.intermediate_size)
            
            param_references = {
                'gate_proj': (layer.gate_proj.weight, (slice(start_idx, end_idx), slice(None))),
                'up_proj': (layer.up_proj.weight, (slice(start_idx, end_idx), slice(None))),
                'down_proj': (layer.down_proj.weight, (slice(None), slice(start_idx, end_idx)))
            }
            
            units.append(PruningUnit(layer_idx, group_idx, param_references))
            
        return units
    
    def _add_layer_dependencies(self, dependencies: Dict[str, List[str]], 
                              attn_units: List[PruningUnit], 
                              mlp_units: List[PruningUnit]):
        """Add dependencies between units in the same layer"""
        for unit in attn_units + mlp_units:
            dependencies[unit.id] = []
            
            # Add dependencies between attention heads and MLP groups
            if 'attn' in unit.id:
                for mlp_unit in mlp_units:
                    if mlp_unit.layer_idx == unit.layer_idx:
                        dependencies[unit.id].append(mlp_unit.id)