#width_pruner.py

import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class WidthPruner:
    """Handles model width pruning operations"""
    
    def __init__(
        self, 
        model: torch.nn.Module,
        config: Dict[str, Any]
    ):
        """
        Initialize width pruning operations.
        
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
        
        # Set MLP group size with validation
        self.mlp_group_size = self._validate_mlp_group_size(
            config.get('pruning', {}).get('dependency', {}).get('mlp_group_size', 8)
        )
        
        # Calculate GQA ratio
        self.q_per_kv = self.num_heads // self.num_kv_heads

        # Get sparsity values from config with the correct path
        width_config = config.get('pruning', {}).get('width_pruning', {})
        self.attention_sparsity = width_config.get('attention_sparsity', 0.0)
        self.mlp_sparsity = width_config.get('mlp_sparsity', 0.0)
        
        logger.info(f"Initializing WidthPruner with sparsity values:")
        logger.info(f"  attention_sparsity: {self.attention_sparsity}")
        logger.info(f"  mlp_sparsity: {self.mlp_sparsity}")

        # Calculate new dimensions
        self.new_kv_heads = max(1, int(self.num_kv_heads * (1 - self.attention_sparsity)))
        self.new_num_heads = self.new_kv_heads * self.q_per_kv
        
        mlp_groups = self.intermediate_size // self.mlp_group_size
        self.new_mlp_groups = max(1, int(mlp_groups * (1 - self.mlp_sparsity)))
        self.new_intermediate_size = self.new_mlp_groups * self.mlp_group_size

    def _validate_mlp_group_size(self, group_size: int) -> int:
        """Validate and adjust MLP group size if necessary"""
        if self.intermediate_size % group_size != 0:
            logger.warning(f"MLP group size {group_size} doesn't evenly divide intermediate_size {self.intermediate_size}")
            for size in [8, 4, 2, 1]:
                if self.intermediate_size % size == 0:
                    logger.info(f"Adjusted MLP group size to {size}")
                    return size
        return group_size

    def restructure_layer(
        self,
        layer: torch.nn.Module,
        attention_mask: torch.Tensor,
        mlp_mask: torch.Tensor,
        apply_head_collapse: bool = False
    ):
        """
        Restructure a transformer layer by applying both attention and MLP pruning.
        
        Args:
            layer: The transformer layer to restructure
            attention_mask: Mask for attention heads
            mlp_mask: Mask for MLP units
            apply_head_collapse: Whether to apply head collapsing
        """
        # Restructure attention
        self.restructure_attention(layer, attention_mask, apply_head_collapse)
        
        # Restructure MLP
        self.restructure_mlp(layer, mlp_mask)

    def restructure_attention(self, layer: torch.nn.Module, mask: torch.Tensor, apply_head_collapse: bool = False):
        """Restructure attention layer with new dimensions"""
        try:
            # Apply head collapsing if enabled
            if apply_head_collapse:
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

    def restructure_mlp(self, layer: torch.nn.Module, mask: torch.Tensor):
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