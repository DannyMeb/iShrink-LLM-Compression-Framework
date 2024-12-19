import torch
import logging
from typing import List, Dict
from pathlib import Path
import json
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ProgressiveSparsifier:
    def __init__(self, model, tokenizer, save_dir: Path = Path("experiments/results")):
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir 
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.saved_states = {}
        self.sparsity_levels = [0.05,0.10, 0.15, 0.20, 0.25,0.30,0.35, 0.40, 0.45, 0.50]

    def _count_parameters(self, unit) -> int:
        """Count number of parameters in a unit"""
        total_params = 0
        for _, (param, slice_idx) in unit.param_references.items():
            if isinstance(slice_idx, tuple):
                shape = param[slice_idx[0], slice_idx[1]].shape
                total_params += shape.numel()
            else:
                shape = param[slice_idx].shape
                total_params += shape.numel()
        return total_params

    def _zero_out_unit(self, unit):
        """Zero out a unit's parameters"""
        for _, (param, slice_idx) in unit.param_references.items():
            if isinstance(slice_idx, tuple):
                param.data[slice_idx[0], slice_idx[1]] = 0
            else:
                param.data[slice_idx] = 0

    def _backup_unit(self, unit):
        """Backup a unit's parameters"""
        unit_state = {}
        for name, (param, slice_idx) in unit.param_references.items():
            if isinstance(slice_idx, tuple):
                unit_state[name] = param.data[slice_idx[0], slice_idx[1]].clone()
            else:
                unit_state[name] = param.data[slice_idx].clone()
        self.saved_states[unit.id] = unit_state

    def _restore_unit(self, unit):
        """Restore a unit's parameters"""
        if unit.id in self.saved_states:
            for name, (param, slice_idx) in unit.param_references.items():
                if isinstance(slice_idx, tuple):
                    param.data[slice_idx[0], slice_idx[1]] = self.saved_states[unit.id][name]
                else:
                    param.data[slice_idx] = self.saved_states[unit.id][name]
            del self.saved_states[unit.id]

    def _save_sparsified_model(self, sparsity: float, statistics: Dict):
        """Save sparsified model and statistics"""
        save_path = self.save_dir / f"sparsified_model_at_{int(sparsity*100)}%"
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving sparsified model ({sparsity*100:.1f}%) to {save_path}")
        
        self.model.config.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path, safe_serialization=True)
        
        stats_file = save_path / "sparsification_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(statistics, f, indent=2)

    def sparsify(self, units: List) -> Dict[str, float]:
        """Progressively sparsify model by zeroing out least important units"""
        timing_info = {}
        
        # Count parameters for each unit and total parameters
        logger.info("Counting parameters...")
        unit_params = {}
        total_params = 0
        attn_total = 0
        mlp_total = 0
        
        for unit in tqdm(units, desc="Counting parameters"):
            params = self._count_parameters(unit)
            unit_params[unit.id] = params
            total_params += params
            if 'attn' in unit.id:
                attn_total += params
            else:
                mlp_total += params

        # Sort units by importance score (ascending - least important first)
        sorted_units = sorted(units, key=lambda x: x.importance_score)
        
        logger.info("\nInitial Statistics:")
        logger.info(f"Total Parameters: {total_params:,}")
        logger.info(f"Attention Parameters: {attn_total:,} ({attn_total/total_params*100:.1f}%)")
        logger.info(f"MLP Parameters: {mlp_total:,} ({mlp_total/total_params*100:.1f}%)")
        logger.info(f"Total Units: {len(units)}")
        logger.info(f"Attention Units: {sum(1 for u in units if 'attn' in u.id)}")
        logger.info(f"MLP Units: {sum(1 for u in units if 'mlp' in u.id)}")
        
        try:
            for sparsity in self.sparsity_levels:
                stage_start = time.time()
                logger.info(f"\nEvaluating {sparsity*100:.1f}% sparsity")
                
                # Calculate target parameters to zero out
                target_params = int(total_params * sparsity)
                current_params = 0
                units_to_zero = []
                
                # Add units in order of importance until we reach target
                for unit in sorted_units:
                    if current_params >= target_params:
                        break
                    units_to_zero.append(unit)
                    current_params += unit_params[unit.id]
                
                # Count parameters by type being zeroed
                zeroed_attn_params = sum(unit_params[u.id] for u in units_to_zero if 'attn' in u.id)
                zeroed_mlp_params = sum(unit_params[u.id] for u in units_to_zero if 'mlp' in u.id)
                zeroed_attn_units = sum(1 for u in units_to_zero if 'attn' in u.id)
                zeroed_mlp_units = sum(1 for u in units_to_zero if 'mlp' in u.id)
                
                logger.info(f"\nZeroing out {len(units_to_zero)} units to achieve {sparsity*100:.1f}% sparsity:")
                logger.info(f"Total parameters to zero: {current_params:,} ({current_params/total_params*100:.1f}%)")
                logger.info(f"Attention units: {zeroed_attn_units} ({zeroed_attn_params/attn_total*100:.1f}% of attention params)")
                logger.info(f"MLP units: {zeroed_mlp_units} ({zeroed_mlp_params/mlp_total*100:.1f}% of MLP params)")
                
                # Backup states
                for unit in tqdm(units_to_zero, desc="Backing up states"):
                    self._backup_unit(unit)
                
                # Zero out units
                for unit in tqdm(units_to_zero, desc="Zeroing units"):
                    self._zero_out_unit(unit)
                
                # Save model and statistics
                statistics = {
                    "sparsity_target": sparsity,
                    "actual_sparsity": current_params/total_params,
                    "total_parameters": total_params,
                    "units_zeroed": {
                        "total": {
                            "count": len(units_to_zero),
                            "parameters": current_params,
                            "percentage": current_params/total_params
                        },
                        "attention": {
                            "count": zeroed_attn_units,
                            "parameters": zeroed_attn_params,
                            "percentage": zeroed_attn_params/attn_total
                        },
                        "mlp": {
                            "count": zeroed_mlp_units,
                            "parameters": zeroed_mlp_params,
                            "percentage": zeroed_mlp_params/mlp_total
                        }
                    },
                    "initial_distribution": {
                        "total_units": len(units),
                        "attention": {
                            "units": sum(1 for u in units if 'attn' in u.id),
                            "parameters": attn_total
                        },
                        "mlp": {
                            "units": sum(1 for u in units if 'mlp' in u.id),
                            "parameters": mlp_total
                        }
                    },
                    "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
                }
                
                self._save_sparsified_model(sparsity, statistics)
                
                # Restore states
                for unit in tqdm(units_to_zero, desc="Restoring states"):
                    self._restore_unit(unit)
                
                timing_info[f'sparsity_{int(sparsity*100)}'] = time.time() - stage_start
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during sparsification: {str(e)}")
            logger.info("Attempting to restore all saved states after error")
            for unit in units:
                if unit.id in self.saved_states:
                    self._restore_unit(unit)
            raise
        
        logger.info("\nCompleted progressive sparsification")
        return timing_info