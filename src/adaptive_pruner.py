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
        target_sparsity: float = 0.25
    ):
        self.model = model
        self.config = config
        self.target_sparsity = target_sparsity
        
        # Read model dimensions from model.config
        self.num_layers = len(model.model.layers)
        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.head_dim = model.config.head_dim
        self.intermediate_size = model.config.intermediate_size
        
        # Set MLP group size - either from config or default to 8
        try:
            self.mlp_group_size = config.get('pruning', {}).get('dependency', {}).get('mlp_group_size', 8)
        except (KeyError, AttributeError):
            self.mlp_group_size = 8
            
        # Verify MLP group size is valid
        if self.intermediate_size % self.mlp_group_size != 0:
            logger.warning(f"MLP group size {self.mlp_group_size} doesn't evenly divide intermediate_size {self.intermediate_size}")
            for size in [8, 4, 2, 1]:
                if self.intermediate_size % size == 0:
                    self.mlp_group_size = size
                    logger.info(f"Adjusted MLP group size to {size}")
                    break
        
        # GQA ratio
        self.q_per_kv = self.num_heads // self.num_kv_heads
        
        logger.info("Initialized StructuralPruner with:")
        logger.info(f"Target sparsity: {target_sparsity}")
        logger.info(f"Layers: {self.num_layers}")
        logger.info(f"Q heads: {self.num_heads}, KV heads: {self.num_kv_heads} (ratio: {self.q_per_kv}:1)")
        logger.info(f"Hidden size: {self.hidden_size}")
        logger.info(f"Head dimension: {self.head_dim}")
        logger.info(f"Intermediate size: {self.intermediate_size}")
        logger.info(f"MLP group size: {self.mlp_group_size}")

    def _select_units_to_keep(self, pruning_units: List[Any]) -> Tuple[Dict, Dict]:
        """Select which units to keep based on importance scores"""
        # Calculate target dimensions respecting GQA structure
        keep_kv_heads = int(self.num_kv_heads * (1 - self.target_sparsity))
        keep_attention = keep_kv_heads * self.q_per_kv
        keep_mlp = int((self.intermediate_size // self.mlp_group_size) * (1 - self.target_sparsity))
        
        logger.info(f"Will keep {keep_kv_heads} KV heads (with {self.q_per_kv} Q heads each = {keep_attention} Q heads)")
        logger.info(f"Will keep {keep_mlp} MLP units out of {self.intermediate_size // self.mlp_group_size}")
        
        attention_masks = {}
        mlp_masks = {}
        
        # Group units by layer
        layer_units = {i: {"attention": [], "mlp": []} for i in range(self.num_layers)}
        for unit in pruning_units:
            unit_type = "attention" if "attn" in unit.id else "mlp"
            layer_units[unit.layer_idx][unit_type].append(unit)
        
        # Create masks for each layer
        for layer_idx in range(self.num_layers):
            # Handle attention units
            attn_units = sorted(layer_units[layer_idx]["attention"], 
                              key=lambda x: x.importance_score, reverse=True)
            
            attention_mask = torch.zeros(self.num_heads, dtype=torch.bool)
            kept_kv_units = attn_units[:keep_kv_heads]
            
            logger.info(f"Layer {layer_idx}: Keeping {len(kept_kv_units)} KV units")
            for unit in kept_kv_units:
                # Set mask for all Q heads associated with this KV head
                q_start = unit.head_idx * self.q_per_kv
                q_end = (unit.head_idx + 1) * self.q_per_kv
                attention_mask[q_start:q_end] = True
            
            attention_masks[layer_idx] = attention_mask
            
            # Verify attention mask
            assert attention_mask.sum() == keep_attention, \
                f"Attention mask sum {attention_mask.sum()} doesn't match target {keep_attention}"
            
            # Handle MLP units
            mlp_units = sorted(layer_units[layer_idx]["mlp"],
                             key=lambda x: x.importance_score, reverse=True)
            
            mlp_mask = torch.zeros(self.intermediate_size // self.mlp_group_size, dtype=torch.bool)
            for unit in mlp_units[:keep_mlp]:
                mlp_mask[unit.head_idx] = True
            
            mlp_masks[layer_idx] = mlp_mask
            
            # Verify MLP mask
            assert mlp_mask.sum() == keep_mlp, \
                f"MLP mask sum {mlp_mask.sum()} doesn't match target {keep_mlp}"
        
        return attention_masks, mlp_masks

    def _restructure_attention(self, layer: torch.nn.Module, mask: torch.Tensor, 
                             new_num_heads: int, new_kv_heads: int):
        """Restructure attention layer with new dimensions"""
        # Debug dimensions
        q_weight = layer.self_attn.q_proj.weight
        logger.debug(f"Original Q weight shape: {q_weight.shape}")
        
        new_qkv_size = new_num_heads * self.head_dim
        new_kv_size = new_kv_heads * self.head_dim
        
        # Reshape Q projection
        q_3d = q_weight.view(self.num_heads, self.head_dim, self.hidden_size)
        kept_q = q_3d[mask]
        kept_q = kept_q.reshape(new_qkv_size, self.hidden_size)
        
        new_q = torch.nn.Linear(self.hidden_size, new_qkv_size, bias=False)
        new_q.weight.data = kept_q
        layer.self_attn.q_proj = new_q
        
        # Reshape K projection
        k_weight = layer.self_attn.k_proj.weight
        k_3d = k_weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)
        kv_mask = mask.view(-1, self.q_per_kv)[:, 0]  # Get KV mask from Q mask
        kept_k = k_3d[kv_mask]
        kept_k = kept_k.reshape(new_kv_size, self.hidden_size)
        
        new_k = torch.nn.Linear(self.hidden_size, new_kv_size, bias=False)
        new_k.weight.data = kept_k
        layer.self_attn.k_proj = new_k
        
        # Reshape V projection
        v_weight = layer.self_attn.v_proj.weight
        v_3d = v_weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)
        kept_v = v_3d[kv_mask]
        kept_v = kept_v.reshape(new_kv_size, self.hidden_size)
        
        new_v = torch.nn.Linear(self.hidden_size, new_kv_size, bias=False)
        new_v.weight.data = kept_v
        layer.self_attn.v_proj = new_v
        
        # Reshape O projection
        o_weight = layer.self_attn.o_proj.weight
        o_3d = o_weight.view(self.hidden_size, self.num_heads, self.head_dim)
        kept_o = o_3d[:, mask, :]
        kept_o = kept_o.reshape(self.hidden_size, new_qkv_size)
        
        new_o = torch.nn.Linear(new_qkv_size, self.hidden_size, bias=False)
        new_o.weight.data = kept_o
        layer.self_attn.o_proj = new_o
        
        # Update attention parameters
        layer.self_attn.num_heads = new_num_heads
        layer.self_attn.num_key_value_heads = new_kv_heads
        layer.self_attn.head_dim = self.head_dim

    def _restructure_mlp(self, layer: torch.nn.Module, mask: torch.Tensor, new_intermediate_size: int):
        """Restructure MLP layer with new dimensions"""
        mask_expanded = mask.repeat_interleave(self.mlp_group_size)
        
        # Reshape gate projection
        gate_weight = layer.mlp.gate_proj.weight
        kept_gate = gate_weight[mask_expanded]
        new_gate = torch.nn.Linear(self.hidden_size, new_intermediate_size, bias=False)
        new_gate.weight.data = kept_gate
        layer.mlp.gate_proj = new_gate
        
        # Reshape up projection
        up_weight = layer.mlp.up_proj.weight
        kept_up = up_weight[mask_expanded]
        new_up = torch.nn.Linear(self.hidden_size, new_intermediate_size, bias=False)
        new_up.weight.data = kept_up
        layer.mlp.up_proj = new_up
        
        # Reshape down projection
        down_weight = layer.mlp.down_proj.weight
        kept_down = down_weight[:, mask_expanded]
        new_down = torch.nn.Linear(new_intermediate_size, self.hidden_size, bias=False)
        new_down.weight.data = kept_down
        layer.mlp.down_proj = new_down

    def optimize_pruning(self, pruning_units: List[Any]) -> PruningResult:
        """Execute structural pruning"""
        # Get initial parameter count
        params_before = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Initial parameter count: {params_before:,}")
        
        # Create copy of model for pruning
        pruned_model = deepcopy(self.model)
        
        # Select units to keep
        attention_masks, mlp_masks = self._select_units_to_keep(pruning_units)
        
        # Calculate new dimensions
        new_kv_heads = int(self.num_kv_heads * (1 - self.target_sparsity))
        new_num_heads = new_kv_heads * self.q_per_kv
        new_intermediate_size = int(self.intermediate_size * (1 - self.target_sparsity))
        
        logger.info("New dimensions:")
        logger.info(f"Q heads: {new_num_heads}, KV heads: {new_kv_heads}")
        logger.info(f"Intermediate size: {new_intermediate_size}")
        
        # Restructure each layer
        for layer_idx in range(self.num_layers):
            layer = pruned_model.model.layers[layer_idx]
            
            # Restructure attention
            self._restructure_attention(
                layer, 
                attention_masks[layer_idx],
                new_num_heads,
                new_kv_heads
            )
            
            # Restructure MLP
            self._restructure_mlp(
                layer,
                mlp_masks[layer_idx],
                new_intermediate_size
            )
        
        # Verify size reduction
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
                'num_heads': new_num_heads,
                'num_kv_heads': new_kv_heads,
                'intermediate_size': new_intermediate_size,
                'hidden_size': self.hidden_size
            },
            attention_mask=attention_masks,
            mlp_mask=mlp_masks,
            compression_ratio=1 - self.target_sparsity,
            params_before=params_before,
            params_after=params_after,
            actual_compression=actual_compression,
            pruned_model=pruned_model
        )

    
    def save_model(self, pruning_result: PruningResult,  tokenizer: AutoTokenizer, save_path: Path):
        """Save pruned model with updated config"""
        save_path.mkdir(parents=True, exist_ok=True)
        model = pruning_result.pruned_model
        
        # Update model config
        model.config.num_attention_heads = pruning_result.pruned_size['num_heads']
        model.config.num_key_value_heads = pruning_result.pruned_size['num_kv_heads']
        model.config.intermediate_size = pruning_result.pruned_size['intermediate_size']
        
        # Save config first
        model.config.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        # Save model weights
        model.save_pretrained(save_path, safe_serialization=True)

        
        # Verify by loading with updated config
        try:
            logger.info("\n=== Verifying Saved Model ===")
            test_load = AutoModelForCausalLM.from_pretrained(
                save_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Match original dtype
                device_map="auto"
            )
            test_size = sum(p.numel() for p in test_load.parameters())
            assert test_size == pruning_result.params_after, \
                f"Loaded model size ({test_size:,}) doesn't match pruned size ({pruning_result.params_after:,})"
            logger.info(f"Successfully verified saved model at: {save_path}")
            
        except Exception as e:
            logger.error(f"Error verifying saved model: {str(e)}")
            raise