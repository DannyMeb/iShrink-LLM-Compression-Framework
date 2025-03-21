#adaptive_pruner.py

import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from .width_pruner import WidthPruner
from .depth_pruner import DepthPruner

logger = logging.getLogger(__name__)

@dataclass
class PruningResult:
    """Stores results of structural pruning"""
    original_size: Dict[str, int]      # Original model dimensions
    pruned_size: Dict[str, int]        # New model dimensions
    attention_mask: Dict[int, torch.Tensor]  # Kept heads per layer
    mlp_mask: Dict[int, torch.Tensor]       # Kept MLP units per layer
    compression_ratio: float
    params_before: int                 # Total parameters before pruning
    params_after: int                  # Total parameters after pruning
    actual_compression: float          # Achieved compression ratio
    pruned_model: torch.nn.Module
    pruned_layers: Optional[List[int]] = None 

class StructuralPruner:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        attention_sparsity: float = 0.25,
        mlp_sparsity: float = 0.25,
        apply_head_collapse: bool = False
    ):
        """Initialize structural pruner"""
        if not (0 <= attention_sparsity <= 1 and 0 <= mlp_sparsity <= 1):
            raise ValueError("Sparsity values must be between 0 and 1")
            
        self.model = model
        self.config = config
        self.attention_sparsity = attention_sparsity
        self.mlp_sparsity = mlp_sparsity
        self.apply_head_collapse = apply_head_collapse
        
        # Initialize width pruner for dimension handling
        self.width_pruner = WidthPruner(model, config)
        
        # Initialize DepthPruner
        self.depth_pruner = DepthPruner(model, config)

        # Get model dimensions from width pruner
        self.num_layers = self.width_pruner.num_layers
        self.hidden_size = self.width_pruner.hidden_size
        self.num_heads = self.width_pruner.num_heads
        self.num_kv_heads = self.width_pruner.num_kv_heads
        self.head_dim = self.width_pruner.head_dim
        self.intermediate_size = self.width_pruner.intermediate_size
        self.mlp_group_size = self.width_pruner.mlp_group_size
        
        # Get target dimensions from width pruner
        self.new_kv_heads = self.width_pruner.new_kv_heads
        self.new_num_heads = self.width_pruner.new_num_heads
        self.new_intermediate_size = self.width_pruner.new_intermediate_size
        
        logger.info(f"Head collapsing is {'enabled' if apply_head_collapse else 'disabled'}")
        self._log_initialization()

    def _log_initialization(self):
        """Log initialization details"""
        logger.info("Initialized StructuralPruner with:")
        logger.info(f"Attention sparsity: {self.attention_sparsity:.2%}")
        logger.info(f"MLP sparsity: {self.mlp_sparsity:.2%}")
        logger.info(f"Layers: {self.num_layers}")
        logger.info(f"Original dimensions:")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Q heads: {self.num_heads}, KV heads: {self.num_kv_heads}")
        logger.info(f"  Intermediate size: {self.intermediate_size}")
        logger.info(f"Target dimensions:")
        logger.info(f"  Q heads: {self.new_num_heads}, KV heads: {self.new_kv_heads}")
        logger.info(f"  Intermediate size: {self.new_intermediate_size}")
        logger.info(f"MLP group size: {self.mlp_group_size}")

    def _select_units_to_keep(self, pruning_units: List[Any]) -> Tuple[Dict, Dict]:
        """Select units to keep based on importance scores"""
        attention_masks = {}
        mlp_masks = {}
        
        # Group units by layer and type
        layer_units = {i: {"attention": [], "mlp": []} for i in range(self.num_layers)}
        for unit in pruning_units:
            unit_type = "attention" if "attn" in unit.id else "mlp"
            layer_units[unit.layer_idx][unit_type].append(unit)
        
        # Calculate units to remove
        kv_heads_to_remove = self.num_kv_heads - self.new_kv_heads
        mlp_groups = self.intermediate_size // self.mlp_group_size
        mlp_units_to_remove = mlp_groups - (self.new_intermediate_size // self.mlp_group_size)
        
        logger.info(f"Will remove {kv_heads_to_remove} KV heads ({self.attention_sparsity:.1%}) and "
                   f"{mlp_units_to_remove} MLP units ({self.mlp_sparsity:.1%}) per layer")
        
        for layer_idx in range(self.num_layers):
            # Handle attention units
            attn_units = sorted(layer_units[layer_idx]["attention"], 
                              key=lambda x: x.importance_score, reverse=True)
            
            attention_mask = torch.ones(self.num_heads, dtype=torch.bool)
            if attn_units and kv_heads_to_remove > 0:
                for unit in attn_units[-kv_heads_to_remove:]:
                    q_start = unit.head_idx * (self.num_heads // self.num_kv_heads)
                    q_end = (unit.head_idx + 1) * (self.num_heads // self.num_kv_heads)
                    attention_mask[q_start:q_end] = False
            
            attention_masks[layer_idx] = attention_mask
            
            # Handle MLP units
            mlp_units = sorted(layer_units[layer_idx]["mlp"],
                             key=lambda x: x.importance_score, reverse=True)
            
            mlp_mask = torch.ones(mlp_groups, dtype=torch.bool)
            if mlp_units and mlp_units_to_remove > 0:
                for unit in mlp_units[-mlp_units_to_remove:]:
                    mlp_mask[unit.head_idx] = False
            
            mlp_masks[layer_idx] = mlp_mask
            
            # Log pruning decisions
            kept_attention = attention_mask.sum().item()
            kept_mlp = mlp_mask.sum().item()
            logger.info(f"Layer {layer_idx}: Keeping {kept_attention}/{self.num_heads} attention heads "
                       f"and {kept_mlp}/{mlp_groups} MLP units")
        
        return attention_masks, mlp_masks

    def optimize_pruning(self, pruning_units: List[Any]) -> PruningResult:
        """Execute structural pruning based on configuration."""
        try:
            params_before = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Initial parameter count: {params_before:,}")

            pruned_model = deepcopy(self.model)
            pruned_indices = []
            current_num_layers = self.num_layers

            # Perform depth pruning first if enabled
            if self.config['pruning']['depth_pruning']['enabled']:
                num_layers_to_prune = self.config['pruning']['depth_pruning']['num_layers_to_prune']
                if num_layers_to_prune > 0:
                    logger.info("\n=== Depth Pruning ===")
                    keep_first_layers = self.config['pruning']['depth_pruning']['keep_first_layers']
                    keep_last_layers = self.config['pruning']['depth_pruning']['keep_last_layers']

                    # Calculate layer importance scores
                    layer_importance_scores = [
                        sum(unit.importance_score for unit in pruning_units if unit.layer_idx == i)
                        for i in range(self.num_layers)
                    ]

                    # Adjust importance scores to prioritize keeping first/last layers
                    for i in range(self.num_layers):
                        if i < keep_first_layers or i >= self.num_layers - keep_last_layers:
                            layer_importance_scores[i] = float('inf')

                    # Perform depth pruning
                    pruned_model = self.depth_pruner.prune_layers(pruned_model, layer_importance_scores)
                    pruned_indices = self.depth_pruner.pruned_indices
                    current_num_layers = len(pruned_model.model.layers)

                    # Update pruning units to only include remaining layers
                    pruning_units = [unit for unit in pruning_units if unit.layer_idx not in pruned_indices]
                    
                    # Update layer indices to match new structure
                    layer_map = {old_idx: new_idx for new_idx, old_idx in 
                            enumerate(sorted(set(range(self.num_layers)) - set(pruned_indices)))}
                    for unit in pruning_units:
                        unit.layer_idx = layer_map[unit.layer_idx]
                else:
                    logger.info("Depth pruning enabled but num_layers_to_prune=0, skipping...")

            # Perform width pruning if enabled
            attention_masks = None
            mlp_masks = None
            if self.config['pruning']['width_pruning']['enabled']:
                logger.info("\n=== Width Pruning ===")
                # Get masks only for remaining layers
                attention_masks, mlp_masks = self._select_units_to_keep(pruning_units)

                # Apply width pruning to remaining layers
                for layer_idx in range(current_num_layers):
                    layer = pruned_model.model.layers[layer_idx]
                    self.width_pruner.restructure_layer(
                        layer=layer,
                        attention_mask=attention_masks[layer_idx],
                        mlp_mask=mlp_masks[layer_idx],
                        apply_head_collapse=self.config['pruning']['width_pruning']['apply_head_collapse']
                    )

            # Calculate final metrics
            params_after = sum(p.numel() for p in pruned_model.parameters())
            actual_compression = (params_before - params_after) / params_before

            logger.info("\n=== Size Verification ===")
            logger.info(f"Parameters before: {params_before:,}")
            logger.info(f"Parameters after: {params_after:,}")
            logger.info(f"Actual compression: {actual_compression:.2%}")

            assert params_after < params_before, "Model size not reduced!"

            # Get final dimensions
            final_num_heads = self.width_pruner.new_num_heads if self.config['pruning']['width_pruning']['enabled'] else self.num_heads
            final_num_kv_heads = self.width_pruner.new_kv_heads if self.config['pruning']['width_pruning']['enabled'] else self.num_kv_heads
            final_intermediate_size = self.width_pruner.new_intermediate_size if self.config['pruning']['width_pruning']['enabled'] else self.intermediate_size

            return PruningResult(
                original_size={
                    'num_heads': self.num_heads,
                    'num_kv_heads': self.num_kv_heads,
                    'intermediate_size': self.intermediate_size,
                    'hidden_size': self.hidden_size
                },
                pruned_size={
                    'num_heads': final_num_heads,
                    'num_kv_heads': final_num_kv_heads,
                    'intermediate_size': final_intermediate_size,
                    'hidden_size': self.hidden_size
                },
                attention_mask=attention_masks if self.config['pruning']['width_pruning']['enabled'] else None,
                mlp_mask=mlp_masks if self.config['pruning']['width_pruning']['enabled'] else None,
                compression_ratio=1 - max(self.config['pruning']['width_pruning'].get('attention_sparsity', 0),
                                        self.config['pruning']['width_pruning'].get('mlp_sparsity', 0)),
                params_before=params_before,
                params_after=params_after,
                actual_compression=actual_compression,
                pruned_model=pruned_model,
                pruned_layers=pruned_indices
            )

        except Exception as e:
            logger.error(f"Error during pruning optimization: {str(e)}")
            raise



    def save_model(self, pruning_result: PruningResult, tokenizer: AutoTokenizer, save_path: Path):
        """Save pruned model and verify it"""
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            model = pruning_result.pruned_model
            
            # Get actual dimensions from model
            sample_layer = model.model.layers[0]
            actual_hidden_size = sample_layer.self_attn.q_proj.weight.shape[1]
            actual_intermediate_size = sample_layer.mlp.gate_proj.weight.shape[0]
            
            # Update all dimension-related config attributes
            model.config.hidden_size = actual_hidden_size
            model.config.intermediate_size = actual_intermediate_size
            model.config.num_attention_heads = pruning_result.pruned_size['num_heads']
            model.config.num_key_value_heads = pruning_result.pruned_size['num_kv_heads']
            model.config.num_hidden_layers = len(model.model.layers)
            
            # Log configuration before saving
            logger.info(f"Saving model with configuration:")
            logger.info(f"  hidden_size: {actual_hidden_size}")
            logger.info(f"  intermediate_size: {actual_intermediate_size}")
            logger.info(f"  num_attention_heads: {model.config.num_attention_heads}")
            logger.info(f"  num_key_value_heads: {model.config.num_key_value_heads}")
            logger.info(f"  num_hidden_layers: {model.config.num_hidden_layers}")
            
            # Save model and tokenizer
            model.config.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path, safe_serialization=True)
            
            # Verify saved model
            self._verify_saved_model(save_path, pruning_result.params_after)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e 

    def _verify_saved_model(self, save_path: Path, expected_params: int):
        """Verify the saved model"""
        try:
            logger.info("\n=== Verifying Saved Model ===")
            test_load = AutoModelForCausalLM.from_pretrained(
                save_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Verify parameter count
            test_size = sum(p.numel() for p in test_load.parameters())
            assert test_size == expected_params, \
                f"Loaded model size ({test_size:,}) doesn't match pruned size ({expected_params:,})"
                
            # Verify model configuration
            self._verify_model_config(test_load)
            
            logger.info(f"Successfully verified saved model at: {save_path}")
            
        except Exception as e:
            logger.error(f"Error verifying saved model: {str(e)}")
            raise

    def _verify_model_config(self, loaded_model: torch.nn.Module):
        """Verify model configuration"""
        try:
            sample_layer = loaded_model.model.layers[0]
            actual_hidden_size = sample_layer.self_attn.q_proj.weight.shape[1]
            actual_intermediate_size = sample_layer.mlp.gate_proj.weight.shape[0]

            assert loaded_model.config.hidden_size == actual_hidden_size, \
                f"Mismatched hidden size: config={loaded_model.config.hidden_size}, actual={actual_hidden_size}"
            assert loaded_model.config.intermediate_size == actual_intermediate_size, \
                f"Mismatched intermediate size: config={loaded_model.config.intermediate_size}, actual={actual_intermediate_size}"
            assert loaded_model.config.num_attention_heads == self.new_num_heads, \
                f"Mismatched attention heads: config={loaded_model.config.num_attention_heads}, expected={self.new_num_heads}"
            assert loaded_model.config.num_key_value_heads == self.new_kv_heads, \
                f"Mismatched KV heads: config={loaded_model.config.num_key_value_heads}, expected={self.new_kv_heads}"
                
            logger.info("Model configuration verified successfully")
                
        except AssertionError as e:
            logger.error(f"Configuration verification failed: {str(e)}")
            raise
            
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics"""
        return {
            'attention_sparsity': self.attention_sparsity,
            'mlp_sparsity': self.mlp_sparsity,
            'attention_compression': 1 - (self.new_num_heads / self.num_heads),
            'kv_compression': 1 - (self.new_kv_heads / self.num_kv_heads),
            'mlp_compression': 1 - (self.new_intermediate_size / self.intermediate_size),
            'total_heads_removed': self.num_heads - self.new_num_heads,
            'total_kv_heads_removed': self.num_kv_heads - self.new_kv_heads,
            'total_mlp_units_removed': self.intermediate_size - self.new_intermediate_size
        }