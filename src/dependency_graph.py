#dependency_graph.py

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class PruningUnit:
    """Represents a pruning unit (GQA group or MLP neuron group)"""
    id: str
    layer_idx: int
    head_idx: int  # for attention heads or mlp group index
    parameters: Dict[str, torch.Tensor]  
    importance_score: float = 0.0

class DependencyGraphBuilder:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        if hasattr(self.model, 'model'):
            self.transformer = self.model.model
        else:
            self.transformer = self.model
            
        # Get model architecture parameters from model config
        model_config = self.model.config
        self.hidden_size = model_config.hidden_size
        self.num_heads = model_config.num_attention_heads
        self.num_kv_heads = getattr(model_config, 'num_key_value_heads', self.num_heads)  # Default to num_heads if not GQA
        self.intermediate_size = model_config.intermediate_size
        
        # MLP grouping parameter
        self.mlp_group_size = config.get('mlp_group_size', 128)
        
        self.pruning_units = []
        logger.info(f"Initialized pruning units builder")
    
    def build(self) -> Tuple[List[PruningUnit], None]:
        """Build pruning units for both attention and MLP"""
        try:
            if hasattr(self.transformer, 'layers'):
                layers = self.transformer.layers
            else:
                raise ValueError("Could not find transformer layers")
            
            total_layers = len(layers)
            start_layer = int(total_layers * 0.98)  # Last 20%
            
            logger.info(f"Found {len(layers)} transformer layers")
            logger.info(f"Processing last 20% of layers: {start_layer} to {total_layers-1}")
            
            # Create attention pruning units (GQA groups)
            for layer_idx in range(start_layer, total_layers):
                layer = layers[layer_idx]
                for kv_idx in range(self.num_kv_heads):
                    unit = self._create_attention_unit(layer_idx, kv_idx, layer)
                    self.pruning_units.append(unit)
            
            # Create MLP pruning units (grouped neurons)
            num_groups = (self.intermediate_size + self.mlp_group_size - 1) // self.mlp_group_size
            for layer_idx in range(start_layer, total_layers):
                layer = layers[layer_idx]
                for group_idx in range(num_groups):
                    unit = self._create_mlp_unit(layer_idx, group_idx, layer)
                    if unit is not None:  # Only add if group is valid
                        self.pruning_units.append(unit)
                
            num_attn_units = (total_layers - start_layer) * self.num_kv_heads
            num_mlp_units = (total_layers - start_layer) * num_groups
            
            logger.info(f"Created {num_attn_units} attention units and {num_mlp_units} MLP units")
            return self.pruning_units, None
            
        except Exception as e:
            logger.error(f"Error creating pruning units: {str(e)}")
            raise
    
    def _create_attention_unit(self, layer_idx: int, kv_idx: int, layer: nn.Module) -> PruningUnit:
        """Create a pruning unit from a GQA group"""
        try:
            # Calculate head dimensions
            head_dim = self.hidden_size // self.num_heads
            q_heads_per_kv = self.num_heads // self.num_kv_heads
            
            # Calculate indices for Q, K, V projections
            q_start = kv_idx * q_heads_per_kv * head_dim
            q_end = (kv_idx + 1) * q_heads_per_kv * head_dim
            kv_start = kv_idx * head_dim
            kv_end = (kv_idx + 1) * head_dim
            
            parameters = {
                'q_proj': layer.self_attn.q_proj.weight[q_start:q_end, :],
                'k_proj': layer.self_attn.k_proj.weight[kv_start:kv_end, :],
                'v_proj': layer.self_attn.v_proj.weight[kv_start:kv_end, :],
                'o_proj': layer.self_attn.o_proj.weight[:, q_start:q_end]
            }
            
            return PruningUnit(
                id=f"attn_{layer_idx}_{kv_idx}",
                layer_idx=layer_idx,
                head_idx=kv_idx,
                parameters=parameters
            )
            
        except AttributeError as e:
            logger.error(f"Error creating attention unit: {str(e)}")
            raise
            
    def _create_mlp_unit(self, layer_idx: int, group_idx: int, layer: nn.Module) -> PruningUnit:
        start_idx = group_idx * self.mlp_group_size
        end_idx = min(start_idx + self.mlp_group_size, self.intermediate_size)
        
        parameters = {
            'gate_proj': layer.mlp.gate_proj.weight.data[start_idx:end_idx, :].requires_grad_(True),
            'up_proj': layer.mlp.up_proj.weight.data[start_idx:end_idx, :].requires_grad_(True),
            'down_proj': layer.mlp.down_proj.weight.data[:, start_idx:end_idx].requires_grad_(True)
        }
        
        return PruningUnit(
            id=f"mlp_{layer_idx}_{group_idx}",
            layer_idx=layer_idx,
            head_idx=group_idx,
            parameters=parameters
        )