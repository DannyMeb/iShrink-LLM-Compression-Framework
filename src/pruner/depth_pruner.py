# depth_pruner.py

import torch
import logging
import copy
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DepthPruner:
    """Handles model depth pruning operations"""
    
    def __init__(
        self, 
        model: torch.nn.Module,
        config: Dict[str, Any]
    ):
        """
        Initialize depth pruning operations.
        
        Args:
            model: The model to prune
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Read model dimensions
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.head_dim
        self.intermediate_size = model.config.intermediate_size
        
        # Get layer pruning config
        self.num_layers_to_prune = config.get('pruning', {}).get('num_layers_to_prune', 0)
        
        # Calculate new dimensions
        self.new_num_layers = self.num_layers - self.num_layers_to_prune
        
        logger.info(f"Initializing depth pruner:")
        logger.info(f"Original layers: {self.num_layers}")
        logger.info(f"Layers to prune: {self.num_layers_to_prune}")
        logger.info(f"Target layers: {self.new_num_layers}")

    def prune_layers(
    self,
    model: torch.nn.Module,
    layer_importance_scores: List[float]) -> torch.nn.Module:
        """
        Create pruned model by removing least important layers.
        
        Args:
            model: The model to prune.
            layer_importance_scores: Importance scores for each layer.
        
        Returns:
            Pruned model with fewer layers.
        """
        try:
            # Create new config with fewer layers
            config = copy.deepcopy(model.config)
            config.num_hidden_layers = self.new_num_layers
            
            # Initialize new model with fewer layers
            pruned_model = type(model)(config)
            
            # Get indices of layers to remove
            layer_indices = list(range(self.num_layers))
            layer_scores = list(zip(layer_indices, layer_importance_scores))
            sorted_layers = sorted(layer_scores, key=lambda x: x[1], reverse=True)
            
            # Keep the most important layers
            keep_indices = sorted([idx for idx, _ in sorted_layers[:self.new_num_layers]])
            self.pruned_indices = sorted([idx for idx, _ in sorted_layers[self.new_num_layers:]])
            
            logger.info(f"Keeping layers: {keep_indices}")
            logger.info(f"Pruning layers: {self.pruned_indices}")
            
            # Copy weights for unpruned components
            pruned_model = self._copy_unpruned_weights(model, pruned_model, keep_indices)
            
            return pruned_model
        
        except Exception as e:
            logger.error(f"Error in layer pruning: {str(e)}")
            raise


    def _copy_unpruned_weights(
        self,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        keep_indices: List[int],
    ) -> torch.nn.Module:
        """
        Copy weights from original model to pruned model, mapping layers appropriately.
        
        Args:
            source_model: Original model
            target_model: Pruned model with fewer layers
            keep_indices: Indices of layers to keep
            
        Returns:
            Model with copied weights
        """
        try:
            # Copy embeddings
            target_model.model.embed_tokens.weight.data.copy_(
                source_model.model.embed_tokens.weight.data
            )
            
            # Copy final layer norm
            target_model.model.norm.weight.data.copy_(
                source_model.model.norm.weight.data
            )
            
            # Copy language model head
            target_model.lm_head.weight.data.copy_(
                source_model.lm_head.weight.data
            )
            
            # Copy kept transformer layers
            for new_idx, old_idx in enumerate(keep_indices):
                logger.info(f"Copying layer {old_idx} to position {new_idx}")
                
                # Get source and target layers
                source_layer = source_model.model.layers[old_idx]
                target_layer = target_model.model.layers[new_idx]
                
                # Copy attention weights
                self._copy_attention_weights(source_layer, target_layer)
                
                # Copy MLP weights
                self._copy_mlp_weights(source_layer, target_layer)
                
                # Copy layer norm weights
                target_layer.input_layernorm.weight.data.copy_(
                    source_layer.input_layernorm.weight.data
                )
                target_layer.post_attention_layernorm.weight.data.copy_(
                    source_layer.post_attention_layernorm.weight.data
                )
            
            return target_model
            
        except Exception as e:
            logger.error(f"Error copying weights: {str(e)}")
            raise

    def _copy_attention_weights(
        self,
        source_layer: torch.nn.Module,
        target_layer: torch.nn.Module
    ):
        """Copy attention weights between layers"""
        # Copy Q, K, V projections
        target_layer.self_attn.q_proj.weight.data.copy_(
            source_layer.self_attn.q_proj.weight.data
        )
        target_layer.self_attn.k_proj.weight.data.copy_(
            source_layer.self_attn.k_proj.weight.data
        )
        target_layer.self_attn.v_proj.weight.data.copy_(
            source_layer.self_attn.v_proj.weight.data
        )
        
        # Copy output projection
        target_layer.self_attn.o_proj.weight.data.copy_(
            source_layer.self_attn.o_proj.weight.data
        )

    def _copy_mlp_weights(
        self,
        source_layer: torch.nn.Module,
        target_layer: torch.nn.Module
    ):
        """Copy MLP weights between layers"""
        # Copy gate projection
        target_layer.mlp.gate_proj.weight.data.copy_(
            source_layer.mlp.gate_proj.weight.data
        )
        
        # Copy up projection
        target_layer.mlp.up_proj.weight.data.copy_(
            source_layer.mlp.up_proj.weight.data
        )
        
        # Copy down projection
        target_layer.mlp.down_proj.weight.data.copy_(
            source_layer.mlp.down_proj.weight.data
        )