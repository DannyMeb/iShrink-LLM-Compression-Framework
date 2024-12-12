import torch
import logging
import json
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from src.importance_scorer import ImportanceScorer
from src.pruning_units import DependencyGraphBuilder, PruningUnit  # Ad
import torch.nn.functional as F

logger = logging.getLogger(__name__)

@dataclass
class PruningResult:
    """Stores the results of structural pruning and model compression.
    
    Attributes:
        original_size: Dictionary containing original model dimensions
        pruned_size: Dictionary containing final pruned model dimensions
        attention_mask: Dictionary mapping layer index to kept attention heads mask
        mlp_mask: Dictionary mapping layer index to kept MLP units mask
        compression_ratio: Target compression ratio achieved
        params_before: Total parameters before pruning
        params_after: Total parameters after pruning
        actual_compression: Actually achieved compression ratio
        pruned_model: The resulting pruned model
        current_attention_sparsity: Final attention sparsity achieved
        current_mlp_sparsity: Final MLP sparsity achieved
    """
    original_size: Dict[str, int]      
    pruned_size: Dict[str, int]        
    attention_mask: Dict[int, torch.Tensor]  
    mlp_mask: Dict[int, torch.Tensor]       
    compression_ratio: float
    params_before: int                 
    params_after: int                  
    actual_compression: float          
    pruned_model: torch.nn.Module
    current_attention_sparsity: float  
    current_mlp_sparsity: float

class StructuralPruner:
    """Handles iterative structural pruning of transformer models with attention and MLP pruning.
    
    This class implements a gradual pruning approach where:
    1. Pruning happens over multiple steps to reach target sparsity
    2. Importance scores are recalculated after each pruning step
    3. Optional knowledge preservation through head/unit collapsing
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        config: Dict[str, Any],
        attention_sparsity: float = 0.1,
        mlp_sparsity: float = 0.3,
        num_steps: int = 10,
        use_collapse: bool = False
    ):
        """Initialize the pruner with target sparsities and pruning configuration.
        
        Args:
            model: The transformer model to prune
            config: Configuration dictionary
            attention_sparsity: Target attention head sparsity (0-1)
            mlp_sparsity: Target MLP unit sparsity (0-1)
            num_steps: Number of gradual pruning steps
            use_collapse: Whether to use knowledge preservation via collapsing
        """
        if not (0 <= attention_sparsity <= 1 and 0 <= mlp_sparsity <= 1):
            raise ValueError("Sparsity values must be between 0 and 1")
            
        self.model = model
        self.tokenizer=tokenizer
        self.config = config
        self.save_dir = Path(self.config['system']['save_dir'])
        self.target_attention_sparsity = attention_sparsity
        self.target_mlp_sparsity = mlp_sparsity
        self.num_steps = num_steps
        self.use_collapse = use_collapse
        
        # Calculate per-step sparsity increments
        self.attention_step = attention_sparsity / num_steps
        self.mlp_step = mlp_sparsity / num_steps
        
        # Track current sparsity levels
        self.current_attention_sparsity = 0.0
        self.current_mlp_sparsity = 0.0
        
        # Initialize model dimensions
        self._initialize_dimensions()
        self._log_initialization()

    def _initialize_dimensions(self):
        """Initialize and store all relevant model dimensions"""
        self.num_layers = len(self.model.model.layers)
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.num_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.model.config.head_dim
        self.intermediate_size = self.model.config.intermediate_size
        
        self.mlp_group_size = self._validate_mlp_group_size(
            self.config.get('pruning', {}).get('dependency', {}).get('mlp_group_size', 8)
        )
        
        self.q_per_kv = self.num_heads // self.num_kv_heads
        
        # Initialize current dimensions
        self.current_kv_heads = self.num_kv_heads
        self.current_num_heads = self.num_heads
        self.current_intermediate_size = self.intermediate_size

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
        """Log detailed information about pruning configuration and model dimensions."""
        logger.info("Initialized StructuralPruner with:")
        logger.info(f"Target attention sparsity: {self.target_attention_sparsity:.2%}")
        logger.info(f"Target MLP sparsity: {self.target_mlp_sparsity:.2%}")
        logger.info(f"Number of pruning steps: {self.num_steps}")
        logger.info(f"Using collapse: {self.use_collapse}")
        logger.info(f"Original dimensions:")
        logger.info(f"  Layers: {self.num_layers}")
        logger.info(f"  Q heads: {self.num_heads}, KV heads: {self.num_kv_heads} (ratio: {self.q_per_kv}:1)")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Head dimension: {self.head_dim}")
        logger.info(f"  Intermediate size: {self.intermediate_size}")
        logger.info(f"  MLP group size: {self.mlp_group_size}")

    def _update_target_dimensions(self):
        """Update target dimensions based on current sparsity levels"""
        # Calculate new attention dimensions
        self.new_kv_heads = max(1, int(self.num_kv_heads * (1 - self.current_attention_sparsity)))
        self.new_num_heads = self.new_kv_heads * self.q_per_kv
        
        # Calculate new MLP dimensions
        mlp_groups = self.intermediate_size // self.mlp_group_size
        self.new_mlp_groups = max(1, int(mlp_groups * (1 - self.current_mlp_sparsity)))
        self.new_intermediate_size = self.new_mlp_groups * self.mlp_group_size

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
        # Calculate output sizes
        new_qkv_size = self.new_num_heads * self.head_dim
        new_kv_size = self.new_kv_heads * self.head_dim
        
        try:
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
   
    def _collapse_attention_heads(
        self, 
        layer: torch.nn.Module, 
        mask: torch.Tensor,
        kv_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collapse attention heads following the paper's preservation strategy.
        For each kept head i, add (head_i - head_{2K-i+1}) where applicable.
        """
        # Get original weights
        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight
        o_weight = layer.self_attn.o_proj.weight
        
        # Reshape into head dimension
        q_3d = q_weight.view(self.num_heads, self.head_dim, self.hidden_size)
        k_3d = k_weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)
        v_3d = v_weight.view(self.num_kv_heads, self.head_dim, self.hidden_size)
        o_3d = o_weight.view(self.hidden_size, self.num_heads, self.head_dim)
        
        # Get indices of kept heads
        kept_indices = torch.where(mask)[0]
        K = len(kept_indices)
        
        # Initialize new weights
        new_q = torch.zeros(self.new_num_heads, self.head_dim, self.hidden_size,
                          device=q_3d.device, dtype=q_3d.dtype)
        
        # Apply head collapsing for query heads
        for i, keep_idx in enumerate(kept_indices):
            if i >= K - (self.num_heads - K):  # Only apply to subset of kept heads
                pair_idx = 2*K - i + 1
                if pair_idx < self.num_heads:  # Ensure valid pairing
                    new_q[i] = q_3d[keep_idx] + (q_3d[keep_idx] - q_3d[pair_idx])
                else:
                    new_q[i] = q_3d[keep_idx]
            else:
                new_q[i] = q_3d[keep_idx]
        
        # Regular pruning for key and value (GQA)
        new_k = k_3d[kv_mask]
        new_v = v_3d[kv_mask]
        new_o = o_3d[:, mask, :]
        
        return new_q, new_k, new_v, new_o

    def _collapse_mlp_units(
        self, 
        layer: torch.nn.Module, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collapse MLP units by combining with their most similar pruned units."""
        # Get original weights
        gate_weight = layer.mlp.gate_proj.weight
        up_weight = layer.mlp.up_proj.weight
        down_weight = layer.mlp.down_proj.weight
        
        # Get indices of kept and pruned units
        kept_indices = torch.where(mask)[0].tolist()
        all_indices = list(range(len(mask)))
        pruned_indices = [i for i in all_indices if i not in kept_indices]
        
        # Initialize new weights
        new_gate = torch.zeros(self.new_intermediate_size, self.hidden_size,
                             device=gate_weight.device, dtype=gate_weight.dtype)
        new_up = torch.zeros_like(new_gate)
        new_down = torch.zeros(self.hidden_size, self.new_intermediate_size,
                             device=down_weight.device, dtype=down_weight.dtype)
        
        # Process each kept unit
        for i, keep_idx in enumerate(kept_indices):
            # Find most similar pruned unit
            similar_idx = self._find_similar_mlp_unit(
                up_weight[keep_idx],
                up_weight,
                kept_indices
            )
            
            # Calculate similarity score
            similarity = F.cosine_similarity(
                up_weight[keep_idx].unsqueeze(0),
                up_weight[similar_idx].unsqueeze(0)
            ).item()
            
            # Dynamic weighting based on similarity
            alpha = max(0.5, similarity)  # Minimum weight of 0.5 for kept unit
            beta = 1 - alpha
            
            # Combine weights
            new_gate[i] = alpha * gate_weight[keep_idx] + beta * gate_weight[similar_idx]
            new_up[i] = alpha * up_weight[keep_idx] + beta * up_weight[similar_idx]
            new_down[:, i] = alpha * down_weight[:, keep_idx] + beta * down_weight[:, similar_idx]
        
        return new_gate, new_up, new_down

    def _create_pruning_units(self, model: torch.nn.Module) -> List[Any]:
        """Create fresh pruning units for the current model state."""
        from src.pruning_units import DependencyGraphBuilder
        
        graph_builder = DependencyGraphBuilder(
            model=model,
            config=self.config
        )
        
        units, _ = graph_builder.build(percent=100.0)  # Always build for full model
        return units

    def _compute_importance_scores(self, model, tokenizer, eval_dataloader, 
                                 pruning_units, scores_path: Path, temp_path: Path):
        """Compute importance scores from scratch"""
        try:
            scorer = ImportanceScorer(
                model=model,
                tokenizer=tokenizer,
                config=self.config['pruning']['importance'],
                calibration_dataloader=eval_dataloader,
                device=self.device
            )
            
            pruning_units = scorer.compute_group_importances(pruning_units)
            
            scores_data = {
                unit.id: {
                    'importance_score': float(unit.importance_score),
                    'layer_idx': unit.layer_idx,
                    'head_idx': unit.head_idx
                }
                for unit in pruning_units
            }
            
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, 'w') as f:
                json.dump(scores_data, f, indent=2)
            
            temp_path.replace(scores_path)
            
            logger.info(f"Successfully computed and saved importance scores to {scores_path}")
            return pruning_units
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _find_similar_mlp_unit(
    self, unit: torch.Tensor, all_units: torch.Tensor, excluded_indices: List[int]) -> int:
        """Find the most similar MLP unit using cosine similarity.
        
        Args:
            unit: The unit to find matches for
            all_units: All available units to compare against
            excluded_indices: Indices to exclude from the search
            
        Returns:
            Index of the most similar unit
        """
        # Create mask for valid units
        valid_mask = torch.ones(all_units.size(0), dtype=torch.bool, device=unit.device)
        valid_mask[excluded_indices] = False
        
        # Calculate similarities only with valid units
        similarities = F.cosine_similarity(
            unit.unsqueeze(0),
            all_units[valid_mask],
            dim=1
        )
        
        # Get index of most similar valid unit
        max_idx = similarities.argmax().item()
        valid_indices = torch.where(valid_mask)[0]
        
        return valid_indices[max_idx].item()

    def optimize_pruning(self,eval_dataloader: Any,pruning_units: List[Any]) -> PruningResult:
        """
        Execute gradual structural pruning with importance recalculation.
        This method performs iterative pruning by:
        1. Starting with initial model and pruning units with their scores
        2. For each step:
        - Calculate current step's target sparsity
        - Apply pruning to the model
        - Create fresh pruning units and compute their importance
        - Use these for next iteration
        3. Return final pruning result
        """
        # Create importance scorer
        importance_scorer = ImportanceScorer(
            model=self.model,
            tokenizer=self.tokenizer,
            config=self.config['pruning']['importance'],
            calibration_dataloader=eval_dataloader,
            device=torch.device(self.config['model']['device'])
        )
        
        # Record initial parameters for final comparison
        params_before = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Starting gradual pruning process with {params_before:,} parameters")
        logger.info(f"Target sparsities - Attention: {self.target_attention_sparsity:.2%}, "
                    f"MLP: {self.target_mlp_sparsity:.2%}")
        
        try:
            # Start with provided units and their scores
            current_units = pruning_units
            
            for step in range(self.num_steps):
                logger.info(f"\n=== Pruning Step {step + 1}/{self.num_steps} ===")
                
                # Add debug logs
                logger.info("1. Calculating sparsity targets...")
                
                # Calculate sparsity targets for this step
                self.current_attention_sparsity = min(
                    self.target_attention_sparsity,
                    (step + 1) * self.attention_step
                )
                self.current_mlp_sparsity = min(
                    self.target_mlp_sparsity,
                    (step + 1) * self.mlp_step
                )
                
                logger.info(f"Current step targets - Attention: {self.current_attention_sparsity:.2%}, "
                            f"MLP: {self.current_mlp_sparsity:.2%}")
                
                # Add debug log
                logger.info("2. Updating dimension targets...")
                self._update_target_dimensions()
                
                # Add debug log
                logger.info("3. Selecting units to keep...")
                attention_masks, mlp_masks = self._select_units_to_keep(current_units)
                
                # Add debug log
                logger.info("4. Applying pruning transformations...")
                # Apply pruning with optional knowledge preservation
                for layer_idx in range(self.num_layers):
                    logger.info(f"Processing layer {layer_idx}/{self.num_layers}")
                    layer = self.model.model.layers[layer_idx]
                    self._restructure_attention(layer, attention_masks[layer_idx])
                    self._restructure_mlp(layer, mlp_masks[layer_idx])
                
                # Add debug log
                logger.info("5. Preparing for next iteration...")
                # If not the last step, prepare for next iteration
                if step < self.num_steps - 1:
                    logger.info("5.1 Creating fresh pruning units...")
                    current_units = self._create_pruning_units(self.model)
                    
                    logger.info("5.2 Calculating new importance scores...")
                    current_units = importance_scorer.compute_group_importances(current_units)
                    
                    logger.info("5.3 Saving importance scores...")
                    scores_path = self.save_dir / 'importance_scores' / f'importance_scores_step_{step + 1}.json'
                    scores_path.parent.mkdir(parents=True, exist_ok=True)
                    scores_data = {
                        unit.id: {
                            'importance_score': float(unit.importance_score),
                            'layer_idx': unit.layer_idx,
                            'head_idx': unit.head_idx
                        }
                        for unit in current_units
                    }
                    with open(scores_path, 'w') as f:
                        json.dump(scores_data, f, indent=2)
            
            # Calculate final metrics
            params_after = sum(p.numel() for p in self.model.parameters())
            actual_compression = (params_before - params_after) / params_before
            
            # Log final size verification
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
                compression_ratio=1 - max(self.current_attention_sparsity, 
                                    self.current_mlp_sparsity),
                params_before=params_before,
                params_after=params_after,
                actual_compression=actual_compression,
                pruned_model=self.model,
                current_attention_sparsity=self.current_attention_sparsity,
                current_mlp_sparsity=self.current_mlp_sparsity
            )
            
        except Exception as e:
            logger.error(f"Error during pruning optimization: {str(e)}")
            raise

    def get_compression_stats(self) -> Dict[str, float]:
        """Get detailed statistics about the compression achieved."""
        return {
            'attention_sparsity': self.current_attention_sparsity,
            'mlp_sparsity': self.current_mlp_sparsity,
            'attention_compression': 1 - (self.new_num_heads / self.num_heads),
            'kv_compression': 1 - (self.new_kv_heads / self.num_kv_heads),
            'mlp_compression': 1 - (self.new_intermediate_size / self.intermediate_size),
            'total_heads_removed': self.num_heads - self.new_num_heads,
            'total_kv_heads_removed': self.num_kv_heads - self.new_kv_heads,
            'total_mlp_units_removed': self.intermediate_size - self.new_intermediate_size,
            'use_collapse': self.use_collapse,
            'num_steps': self.num_steps,
            'target_attention_sparsity': self.target_attention_sparsity,
            'target_mlp_sparsity': self.target_mlp_sparsity
        }

    def save_model(self, pruning_result: PruningResult, tokenizer: AutoTokenizer, save_path: Path):
        """Save pruned model with updated config and verify."""
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            model = pruning_result.pruned_model
            
            # Update model config
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
        """Verify that the saved model can be loaded and has correct parameters."""
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
        """Verify that the loaded model's configuration matches expected dimensions."""
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