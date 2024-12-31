import torch
import logging
import copy
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DepthPruner:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        """Initialize depth pruning operations"""
        self.model = model
        self.config = config
        
        # Model dimensions
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads 
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.head_dim
        self.intermediate_size = model.config.intermediate_size
        
        # Get pruning config
        self.num_layers_to_prune = config.get('pruning', {}).get('depth_pruning', {}).get('num_layers_to_prune', 0)
        self.new_num_layers = self.num_layers - self.num_layers_to_prune
        
        logger.info(f"Initializing depth pruner:")
        logger.info(f"Original layers: {self.num_layers}")
        logger.info(f"Layers to prune: {self.num_layers_to_prune}")
        logger.info(f"Target layers: {self.new_num_layers}")

    def prune_layers(self, model: torch.nn.Module, layer_importance_scores: List[float]) -> torch.nn.Module:
        try:
            # Identify layers to remove
            layer_scores = sorted(enumerate(layer_importance_scores), key=lambda x: x[1], reverse=True)
            self.pruned_indices = sorted([idx for idx, _ in layer_scores[self.new_num_layers:]])
            
            logger.info("\n=== Layer Pruning Details ===")
            logger.info(f"Removing layers: {self.pruned_indices}")
            for idx in self.pruned_indices:
                logger.info(f"Layer {idx} removed (importance score: {layer_importance_scores[idx]:.4f})")

            # Move model to CPU to save memory
            model = model.cpu()
            torch.cuda.empty_cache()

            # Modify layer structure directly
            remaining_layers = []
            for i in range(len(model.model.layers)):
                if i not in self.pruned_indices:
                    remaining_layers.append(model.model.layers[i])
            
            # Update model
            model.model.layers = torch.nn.ModuleList(remaining_layers)
            model.config.num_hidden_layers = len(remaining_layers)

            return model

        except Exception as e:
            logger.error(f"Error in layer pruning: {str(e)}")
            torch.cuda.empty_cache()
            raise

    def _copy_layer_weights(self, source_layer: torch.nn.Module, target_layer: torch.nn.Module):
        """Copy weights between transformer layers"""
        with torch.no_grad():
            # Copy attention weights
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                target_layer.self_attn.__dict__[proj].weight.data.copy_(
                    source_layer.self_attn.__dict__[proj].weight.data
                )
            
            # Copy MLP weights 
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                target_layer.mlp.__dict__[proj].weight.data.copy_(
                    source_layer.mlp.__dict__[proj].weight.data
                )

            # Copy layer norms
            target_layer.input_layernorm.weight.data.copy_(
                source_layer.input_layernorm.weight.data
            )
            target_layer.post_attention_layernorm.weight.data.copy_(
                source_layer.post_attention_layernorm.weight.data
            )