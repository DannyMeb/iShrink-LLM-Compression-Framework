import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

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

class StructuralPruner:
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        attention_sparsity: float = 0.25,
        mlp_sparsity: float = 0.25,
        apply_head_collapse: bool = False
    ):
        """
        Initialize pruner with separate sparsity values for attention and MLP.
        
        Args:
            model: The model to prune
            config: Configuration dictionary
            attention_sparsity: Fraction of attention heads to remove (0-1)
            mlp_sparsity: Fraction of MLP units to remove (0-1)
            apply_head_collapse: Whether to apply head collapsing during pruning
        """
        # Validate sparsity values
        if not (0 <= attention_sparsity <= 1 and 0 <= mlp_sparsity <= 1):
            raise ValueError("Sparsity values must be between 0 and 1")
            
        self.model = model
        self.config = config
        self.attention_sparsity = attention_sparsity
        self.mlp_sparsity = mlp_sparsity
        self.apply_head_collapse = apply_head_collapse
        
        # Read model dimensions
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.head_dim
        self.intermediate_size = model.config.intermediate_size
        
        # Set MLP group size with validation
        self.mlp_group_size = self._validate_mlp_group_size(
            config.get('pruning', {}).get('dependency', {}).get('mlp_group_size', 8)
        )
        
        # Calculate GQA ratio
        self.q_per_kv = self.num_heads // self.num_kv_heads
        
        # Calculate new dimensions based on sparsity
        self.new_kv_heads = max(1, int(self.num_kv_heads * (1 - self.attention_sparsity)))
        self.new_num_heads = self.new_kv_heads * self.q_per_kv
        
        mlp_groups = self.intermediate_size // self.mlp_group_size
        self.new_mlp_groups = max(1, int(mlp_groups * (1 - self.mlp_sparsity)))
        self.new_intermediate_size = self.new_mlp_groups * self.mlp_group_size
        
        logger.info(f"Head collapsing is {'enabled' if apply_head_collapse else 'disabled'}")
        self._log_initialization()

    def _validate_mlp_group_size(self, group_size: int) -> int:
        """Validate and adjust MLP group size if necessary"""
        if self.intermediate_size % group_size != 0:
            logger.warning(f"MLP group size {group_size} doesn't evenly divide intermediate_size {self.intermediate_size}")
            for size in [8, 4, 2, 1]:
                if self.intermediate_size % size == 0:
                    logger.info(f"Adjusted MLP group size to {size}")
                    return size
        return group_size

    def _log_initialization(self):
        """Log initialization details"""
        logger.info("Initialized StructuralPruner with:")
        logger.info(f"Attention sparsity: {self.attention_sparsity:.2%}")
        logger.info(f"MLP sparsity: {self.mlp_sparsity:.2%}")
        logger.info(f"Layers: {self.num_layers}")
        logger.info(f"Original dimensions:")
        logger.info(f"  Q heads: {self.num_heads}, KV heads: {self.num_kv_heads} (ratio: {self.q_per_kv}:1)")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Head dimension: {self.head_dim}")
        logger.info(f"  Intermediate size: {self.intermediate_size}")
        logger.info(f"Target dimensions:")
        logger.info(f"  Q heads: {self.new_num_heads}, KV heads: {self.new_kv_heads}")
        logger.info(f"  Intermediate size: {self.new_intermediate_size}")
        logger.info(f"MLP group size: {self.mlp_group_size}")

    def _select_units_to_keep(self, pruning_units: List[Any]) -> Tuple[Dict, Dict]:
        """Select units to keep based on importance scores and sparsity values"""
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
        mlp_units_to_remove = mlp_groups - self.new_mlp_groups
        
        logger.info(f"Will remove {kv_heads_to_remove} KV heads ({self.attention_sparsity:.1%}) and "
                   f"{mlp_units_to_remove} MLP units ({self.mlp_sparsity:.1%}) per layer")
        
        for layer_idx in range(self.num_layers):
            # Handle attention units
            attn_units = sorted(layer_units[layer_idx]["attention"], 
                              key=lambda x: x.importance_score, reverse=True)
            
            attention_mask = torch.ones(self.num_heads, dtype=torch.bool)
            if attn_units and kv_heads_to_remove > 0:
                for unit in attn_units[-kv_heads_to_remove:]:
                    q_start = unit.head_idx * self.q_per_kv
                    q_end = (unit.head_idx + 1) * self.q_per_kv
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

    def _restructure_attention(self, layer: torch.nn.Module, mask: torch.Tensor):
        """Restructure attention layer with new dimensions"""
        try:
            # Apply head collapsing if enabled
            if self.apply_head_collapse:
                K = self.new_num_heads
                L = self.num_heads
                
                if hasattr(self.model.config, 'num_key_value_heads'):
                    # GQA: Only collapse query heads
                    q_weight = layer.self_attn.q_proj.weight
                    q_3d = q_weight.view(self.num_heads, self.head_dim, self.hidden_size)
                    for i in range(K - (L - K), K):
                        mirror_idx = 2*K - i + 1
                        if mirror_idx < L:
                            residual = q_3d[i] - q_3d[mirror_idx]
                            q_3d[i] = q_3d[i] + residual
                else:
                    # Standard attention: Collapse all Q,K,V
                    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                        weight = getattr(layer.self_attn, proj_name).weight
                        weight_3d = weight.view(-1, self.head_dim, self.hidden_size)
                        for i in range(K - (L - K), K):
                            mirror_idx = 2*K - i + 1
                            if mirror_idx < L:
                                residual = weight_3d[i] - weight_3d[mirror_idx]
                                weight_3d[i] = weight_3d[i] + residual
            
            # Calculate output sizes
            new_qkv_size = self.new_num_heads * self.head_dim
            new_kv_size = self.new_kv_heads * self.head_dim
            
            # Reshape Q projection
            q_weight = layer.self_attn.q_proj.weight
            q_3d = q_weight.view(self.num_heads, self.head_dim, self.hidden_size)
            kept_q = q_3d[mask]
            kept_q = kept_q.reshape(new_qkv_size, self.hidden_size)
            
            # Reshape K projection
            k_weight = layer.self_attn.k_proj.weight
            k_3d = k_weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)
            kv_mask = mask.view(-1, self.q_per_kv)[:, 0]  # Get KV mask from Q mask
            kept_k = k_3d[kv_mask]
            kept_k = kept_k.reshape(new_kv_size, self.hidden_size)
            
            # Reshape V projection
            v_weight = layer.self_attn.v_proj.weight
            v_3d = v_weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)
            kept_v = v_3d[kv_mask]
            kept_v = kept_v.reshape(new_kv_size, self.hidden_size)
            
            # Reshape O projection
            o_weight = layer.self_attn.o_proj.weight
            o_3d = o_weight.view(self.hidden_size, self.num_heads, self.head_dim)
            kept_o = o_3d[:, mask, :]
            kept_o = kept_o.reshape(self.hidden_size, new_qkv_size)
            
            # Create new linear layers
            new_q = torch.nn.Linear(self.hidden_size, new_qkv_size, bias=False)
            new_k = torch.nn.Linear(self.hidden_size, new_kv_size, bias=False)
            new_v = torch.nn.Linear(self.hidden_size, new_kv_size, bias=False)
            new_o = torch.nn.Linear(new_qkv_size, self.hidden_size, bias=False)
            
            # Assign weights
            new_q.weight.data = kept_q
            new_k.weight.data = kept_k
            new_v.weight.data = kept_v
            new_o.weight.data = kept_o
            
            # Update layer
            layer.self_attn.q_proj = new_q
            layer.self_attn.k_proj = new_k
            layer.self_attn.v_proj = new_v
            layer.self_attn.o_proj = new_o
            
            # Update attention parameters
            layer.self_attn.num_heads = self.new_num_heads
            layer.self_attn.num_key_value_heads = self.new_kv_heads
            layer.self_attn.head_dim = self.head_dim
            
        except Exception as e:
            logger.error(f"Error in attention restructuring: {str(e)}")
            raise

    def _restructure_mlp(self, layer: torch.nn.Module, mask: torch.Tensor):
        """Restructure MLP layer with new dimensions"""
        try:
            mask_expanded = mask.repeat_interleave(self.mlp_group_size)
            
            # Reshape gate projection
            gate_weight = layer.mlp.gate_proj.weight
            kept_gate = gate_weight[mask_expanded]
            new_gate = torch.nn.Linear(self.hidden_size, self.new_intermediate_size, bias=False)
            new_gate.weight.data = kept_gate
            layer.mlp.gate_proj = new_gate
            
            # Reshape up projection
            up_weight = layer.mlp.up_proj.weight
            kept_up = up_weight[mask_expanded]
            new_up = torch.nn.Linear(self.hidden_size, self.new_intermediate_size, bias=False)
            new_up.weight.data = kept_up
            layer.mlp.up_proj = new_up
            
            # Reshape down projection
            down_weight = layer.mlp.down_proj.weight
            kept_down = down_weight[:, mask_expanded]
            new_down = torch.nn.Linear(self.new_intermediate_size, self.hidden_size, bias=False)
            new_down.weight.data = kept_down
            layer.mlp.down_proj = new_down
            
        except Exception as e:
            logger.error(f"Error in MLP restructuring: {str(e)}")
            raise

    def optimize_pruning(self, pruning_units: List[Any]) -> PruningResult:
        """Execute structural pruning with separate sparsity values"""
        try:
            params_before = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Initial parameter count: {params_before:,}")
            
            pruned_model = deepcopy(self.model)
            attention_masks, mlp_masks = self._select_units_to_keep(pruning_units)
            
            for layer_idx in range(self.num_layers):
                layer = pruned_model.model.layers[layer_idx]
                
                # Restructure attention using pre-calculated dimensions
                self._restructure_attention(layer, attention_masks[layer_idx])
                
                # Restructure MLP using pre-calculated dimensions
                self._restructure_mlp(layer, mlp_masks[layer_idx])
            
            params_after = sum(p.numel() for p in pruned_model.parameters())
            actual_compression = (params_before - params_after) / params_before
            
            logger.info("\n=== Size Verification ===")
            logger.info(f"Parameters before: {params_before:,}")
            logger.info(f"Parameters after: {params_after:,}")
            logger.info(f"Actual compression: {actual_compression:.2%}")
            
            assert params_after < params_before, "Model size not reduced!"
            
            return PruningResult(
                original_size={
                    'num_heads': self.num_heads,
                    'num_kv_heads': self.num_kv_heads,
                    'intermediate_size': self.intermediate_size,
                    'hidden_size': self.hidden_size
                },
                pruned_size={
                    'num_heads': self.new_num_heads,
                    'num_kv_heads': self.new_kv_heads,
                    'intermediate_size': self.new_intermediate_size,
                    'hidden_size': self.hidden_size
                },
                attention_mask=attention_masks,
                mlp_mask=mlp_masks,
                compression_ratio=1 - max(self.attention_sparsity, self.mlp_sparsity),
                params_before=params_before,
                params_after=params_after,
                actual_compression=actual_compression,
                pruned_model=pruned_model
            )
            
        except Exception as e:
            logger.error(f"Error during pruning optimization: {str(e)}")
            raise

    def save_model(self, pruning_result: PruningResult, tokenizer: AutoTokenizer, save_path: Path):
        """
        Save pruned model with updated config and verify the saved model.
        
        Args:
            pruning_result: The result of pruning containing model and metrics
            tokenizer: The tokenizer associated with the model
            save_path: Path where the model should be saved
        """
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            model = pruning_result.pruned_model
            
            # Update model config with new dimensions
            model.config.num_attention_heads = pruning_result.pruned_size['num_heads']
            model.config.num_key_value_heads = pruning_result.pruned_size['num_kv_heads']
            model.config.intermediate_size = pruning_result.pruned_size['intermediate_size']
            
            # Save artifacts
            logger.info(f"Saving pruned model to {save_path}")
            model.config.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path, safe_serialization=True)
            
            # Verify saved model
            self._verify_saved_model(save_path, pruning_result.params_after)
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def _verify_saved_model(self, save_path: Path, expected_params: int):
        """Verify that the saved model can be loaded and has correct parameters"""
        try:
            logger.info("\n=== Verifying Saved Model ===")
            test_load = AutoModelForCausalLM.from_pretrained(
                save_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Match original dtype
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
        """Verify that loaded model's configuration matches expected dimensions"""
        try:
            assert loaded_model.config.num_attention_heads == self.new_num_heads, \
                "Mismatched number of attention heads"
            assert loaded_model.config.num_key_value_heads == self.new_kv_heads, \
                "Mismatched number of KV heads"
            assert loaded_model.config.intermediate_size == self.new_intermediate_size, \
                "Mismatched intermediate size"
            assert loaded_model.config.hidden_size == self.hidden_size, \
                "Mismatched hidden size"
                
            logger.info("Model configuration verified successfully")
            
        except AssertionError as e:
            logger.error(f"Configuration verification failed: {str(e)}")
            raise

    def get_compression_stats(self) -> Dict[str, float]:
        """Get detailed statistics about the compression achieved"""
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