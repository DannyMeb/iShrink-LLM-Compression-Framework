import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class PruningUnit:
    """Represents a pruning unit (GQA group or MLP neuron)"""
    id: str
    layer_idx: int
    head_idx: int  # for attention heads or mlp neuron index
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
            
        # Model architecture parameters
        self.hidden_size = config.get('hidden_size', 2048)
        self.num_heads = config.get('num_attention_heads', 32)
        self.num_kv_heads = config.get('num_key_value_heads', 8)
        self.head_dim = config.get('head_dim', 64)
        self.intermediate_size = config.get('intermediate_size', 8192)
        
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
            start_layer = int(total_layers * 0.8)  # Last 20%
            
            logger.info(f"Found {len(layers)} transformer layers")
            logger.info(f"Processing last 20% of layers: {start_layer} to {total_layers-1}")
            
            # Create attention pruning units (GQA groups)
            for layer_idx in range(start_layer, total_layers):
                layer = layers[layer_idx]
                for kv_idx in range(self.num_kv_heads):
                    unit = self._create_attention_unit(layer_idx, kv_idx, layer)
                    self.pruning_units.append(unit)
            
            # Create MLP pruning units (per neuron)
            for layer_idx in range(start_layer, total_layers):
                layer = layers[layer_idx]
                for neuron_idx in range(self.intermediate_size):
                    unit = self._create_mlp_unit(layer_idx, neuron_idx, layer)
                    self.pruning_units.append(unit)
                
            logger.info(f"Created {len(self.pruning_units)} pruning units")
            return self.pruning_units, None
            
        except Exception as e:
            logger.error(f"Error creating pruning units: {str(e)}")
            raise
    
    def _create_attention_unit(self, layer_idx: int, kv_idx: int, layer: nn.Module) -> PruningUnit:
        """Create a pruning unit from a GQA group (4 Q heads per KV head)"""
        try:
            q_heads_per_kv = self.num_heads // self.num_kv_heads
            q_start = kv_idx * q_heads_per_kv * self.head_dim
            q_end = (kv_idx + 1) * q_heads_per_kv * self.head_dim
            kv_start = kv_idx * self.head_dim
            kv_end = (kv_idx + 1) * self.head_dim
            
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
            
    def _create_mlp_unit(self, layer_idx: int, neuron_idx: int, layer: nn.Module) -> PruningUnit:
        """Create a pruning unit from a single MLP neuron"""
        try:
            parameters = {
                'gate_proj': layer.mlp.gate_proj.weight[neuron_idx:neuron_idx+1, :],  # One row
                'up_proj': layer.mlp.up_proj.weight[neuron_idx:neuron_idx+1, :],      # One row
                'down_proj': layer.mlp.down_proj.weight[:, neuron_idx:neuron_idx+1]   # One column
            }
            
            return PruningUnit(
                id=f"mlp_{layer_idx}_{neuron_idx}",
                layer_idx=layer_idx,
                head_idx=neuron_idx,  # Using head_idx for neuron index
                parameters=parameters
            )
            
        except AttributeError as e:
            logger.error(f"Error creating MLP unit: {str(e)}")
            raise