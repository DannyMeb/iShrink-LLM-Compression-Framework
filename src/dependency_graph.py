# src/dependency_graph.py

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class PruningUnit:
    """Represents a single attention head as an independent pruning unit"""
    id: str
    layer_idx: int
    head_idx: int
    parameters: Dict[str, torch.Tensor]  # QKV parameters for this head
    importance_score: float = 0.0

class DependencyGraphBuilder:
    """Creates pruning units for each attention head"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Extract the actual transformer layers
        if hasattr(self.model, 'model'):
            # LlamaForCausalLM case
            self.transformer = self.model.model
        else:
            self.transformer = self.model
            
        # Model architecture parameters
        self.hidden_size = config.get('hidden_size', 768)
        self.num_heads = config.get('num_heads', 12)
        self.head_dim = self.hidden_size // self.num_heads
        
        self.pruning_units = []
        logger.info(f"Initialized head-level pruning units builder")
    
    def build(self) -> Tuple[List[PruningUnit], None]:
        """Build independent pruning units for each attention head"""
        try:
            # Get transformer layers
            if hasattr(self.transformer, 'layers'):
                layers = self.transformer.layers
            else:
                raise ValueError("Could not find transformer layers in model structure")
            
            logger.info(f"Found {len(layers)} transformer layers")
            
            # Calculate which layers to process (last 20%)
            total_layers = len(layers)
            start_layer = int(total_layers * 0.92)  # Start from 80% mark
            layers_to_process = layers[start_layer:]
            
            logger.info(f"Processing last 20% of layers: {start_layer} to {total_layers-1}")
            
            # Create pruning units only for selected layers
            for layer_idx in range(start_layer, total_layers):
                layer = layers[layer_idx]
                for head_idx in range(self.num_heads):
                    unit = self._create_head_unit(layer_idx, head_idx, layer)
                    self.pruning_units.append(unit)
            
            logger.info(f"Created {len(self.pruning_units)} head pruning units")
            return self.pruning_units, None
            
        except Exception as e:
            logger.error(f"Error creating pruning units: {str(e)}")
            raise RuntimeError(f"Error creating pruning units: {str(e)}")
    
    def _create_head_unit(self, layer_idx: int, head_idx: int, layer: nn.Module) -> PruningUnit:
        """Create a pruning unit from an attention head"""
        try:
            attn = layer.self_attn
            
            # Get head's QKV parameters
            start_idx = head_idx * self.head_dim
            end_idx = (head_idx + 1) * self.head_dim
            
            parameters = {
                f'layer_{layer_idx}_head_{head_idx}_q': attn.q_proj.weight[start_idx:end_idx, :],
                f'layer_{layer_idx}_head_{head_idx}_k': attn.k_proj.weight[start_idx:end_idx, :],
                f'layer_{layer_idx}_head_{head_idx}_v': attn.v_proj.weight[start_idx:end_idx, :],
                f'layer_{layer_idx}_head_{head_idx}_o': attn.o_proj.weight[:, start_idx:end_idx]
            }
            
            return PruningUnit(
                id=f"head_{layer_idx}_{head_idx}",
                layer_idx=layer_idx,
                head_idx=head_idx,
                parameters=parameters
            )
            
        except AttributeError as e:
            logger.error(f"Error creating head unit for layer {layer_idx}, head {head_idx}: {str(e)}")
            raise