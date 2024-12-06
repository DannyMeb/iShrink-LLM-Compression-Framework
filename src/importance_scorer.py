#importance_scorer.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional
from .dependency_graph import PruningUnit
import logging
import traceback
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImportanceScorer:
    def __init__(self, model: nn.Module, tokenizer: Any, config: Dict,
                calibration_dataloader: torch.utils.data.DataLoader, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config.get('pruning', {}).get('importance', {})
        
        # Scoring method configuration
        self.method = self.config.get('scoring_method', 'taylor')
        self.weights = self.config.get('weights', {
            'mse': 0.2,
            'gradient': 0.2,
            'taylor': 0.6
        })
        
        # Performance settings
        self.batch_size = self.config.get('batch_size_per_gpu', 2)
        self.use_mixed_precision = self.config.get('use_mixed_precision', True)
        self.grad_accum_steps = self.config.get('gradient_accumulation_steps', 8)
        
        # Prepare calibration data
        percent = self.config.get('calibration_percent', 1.0)
        self.subset = self._prepare_calibration_data(calibration_dataloader, percent)
        logger.info(f"Using {len(self.subset)} samples for importance scoring with method: {self.method}")

    def _prepare_calibration_data(self, dataloader, percent: float) -> List[Dict]:
        """Prepare subset of data for importance scoring"""
        total_samples = len(dataloader.dataset)
        num_samples = int(total_samples * percent)
        
        subset_data = []
        samples_collected = 0
        
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            subset_data.append({
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'labels': batch['labels'].to(self.device) if 'labels' in batch else None
            })
            samples_collected += batch['input_ids'].size(0)
            
        return subset_data

    def compute_mse_importance(self, pruning_unit: PruningUnit, batch: Dict) -> float:
        """Compute MSE-based importance by zeroing unit parameters"""
        with torch.amp.autocast('cuda', enabled=self.use_mixed_precision), torch.no_grad():
            # Get original outputs
            outputs_normal = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            ).logits
            
            # Save original values
            saved_values = {}
            for name, (param, slice_idx) in pruning_unit.param_references.items():
                if isinstance(slice_idx, tuple):
                    saved_values[name] = param[slice_idx[0], slice_idx[1]].clone()
                    param.data[slice_idx[0], slice_idx[1]] = 0
                else:
                    saved_values[name] = param[slice_idx].clone()
                    param.data[slice_idx] = 0
            
            # Get outputs with zeroed parameters
            outputs_zeroed = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            ).logits
            
            # Restore original values
            for name, (param, slice_idx) in pruning_unit.param_references.items():
                if isinstance(slice_idx, tuple):
                    param.data[slice_idx[0], slice_idx[1]] = saved_values[name]
                else:
                    param.data[slice_idx] = saved_values[name]
            
            return F.mse_loss(outputs_normal, outputs_zeroed).item()

    def compute_taylor_importance(self, pruning_unit: PruningUnit, batch: Dict) -> float:
        """Compute Taylor-based importance score"""
        total_importance = 0.0
        
        for step in range(self.grad_accum_steps):
            self.model.zero_grad()
            
            with torch.set_grad_enabled(True):
                with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    logits = outputs.logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch['input_ids'].view(-1)
                    ) / self.grad_accum_steps
                    
                    loss.backward()
            
            # Calculate importance using original parameters and their gradients
            step_importance = 0.0
            for name, (param, slice_idx) in pruning_unit.param_references.items():
                if param.grad is not None:
                    if isinstance(slice_idx, tuple):
                        grad = param.grad[slice_idx[0], slice_idx[1]]
                        weight = param.data[slice_idx[0], slice_idx[1]]
                    else:
                        grad = param.grad[slice_idx]
                        weight = param.data[slice_idx]
                    
                    step_importance += (weight.abs() * grad.abs()).sum().item()
                else:
                    logger.warning(f"No gradient for parameter {name}")
            
            total_importance += step_importance
            
            if self.config.get('clear_cache', True):
                torch.cuda.empty_cache()
        
        return total_importance / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0

    def compute_gradient_importance(self, pruning_unit: PruningUnit, batch: Dict) -> float:
        """Compute gradient-based importance score"""
        self.model.zero_grad()
        total_grad_norm = 0.0
        
        for step in range(self.grad_accum_steps):
            with torch.set_grad_enabled(True):
                with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    logits = outputs.logits
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch['input_ids'].view(-1)
                    ) / self.grad_accum_steps
                    
                    loss.backward()
            
            # Compute gradient norm for the unit's parameters
            step_grad_norm = 0.0
            for name, (param, slice_idx) in pruning_unit.param_references.items():
                if param.grad is not None:
                    if isinstance(slice_idx, tuple):
                        grad = param.grad[slice_idx[0], slice_idx[1]]
                    else:
                        grad = param.grad[slice_idx]
                    step_grad_norm += grad.norm().item() ** 2
            
            total_grad_norm += step_grad_norm
            
            if self.config.get('clear_cache', True):
                torch.cuda.empty_cache()
        
        return total_grad_norm / self.grad_accum_steps if self.grad_accum_steps > 0 else 0.0

    def compute_importance(self, pruning_unit: PruningUnit) -> float:
        """Compute importance score for a pruning unit"""
        try:
            importance_scores = []
            logger.debug(f"Processing unit: {pruning_unit.id}")
            
            was_training = self.model.training
            self.model.eval()  # Ensure model is in eval mode
            
            try:
                for batch_idx, batch in enumerate(self.subset):
                    n_samples = batch['input_ids'].size(0)
                    batch_importance = 0.0
                    
                    for i in range(0, n_samples, self.batch_size):
                        batch_slice = {
                            'input_ids': batch['input_ids'][i:i+self.batch_size].clone(),
                            'attention_mask': batch['attention_mask'][i:i+self.batch_size].clone()
                        }
                        
                        if self.method == 'combined':
                            mse_score = self.compute_mse_importance(pruning_unit, batch_slice)
                            grad_score = self.compute_gradient_importance(pruning_unit, batch_slice)
                            taylor_score = self.compute_taylor_importance(pruning_unit, batch_slice)
                            
                            batch_importance += (
                                self.weights['mse'] * mse_score +
                                self.weights['gradient'] * grad_score +
                                self.weights['taylor'] * taylor_score
                            )
                        elif self.method == 'mse':
                            batch_importance += self.compute_mse_importance(pruning_unit, batch_slice)
                        elif self.method == 'gradient':
                            batch_importance += self.compute_gradient_importance(pruning_unit, batch_slice)
                        else:  # default to taylor
                            batch_importance += self.compute_taylor_importance(pruning_unit, batch_slice)
                        
                        if self.config.get('clear_cache', True):
                            torch.cuda.empty_cache()
                    
                    importance_scores.append(batch_importance / ((n_samples + self.batch_size - 1) // self.batch_size))
            
            finally:
                if was_training:
                    self.model.train()
            
            if importance_scores:
                return sum(importance_scores) / len(importance_scores)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error computing importance for {pruning_unit.id}: {str(e)}")
            traceback.print_exc()
            return 0.0

    def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
        """
        Compute importance scores for all units using gradient-based scoring and z-score normalization.
        Importance is calculated globally across all units to maintain fair comparisons between layers.
        """
        logger.info(f"Computing importance scores using {self.method} method")
        was_training = self.model.training
        self.model.eval()
        
        try:
            # Initialize importance scores
            scores_dict = {unit.id: 0.0 for unit in pruning_units}
            
            # Process first 5 batches for speed
            num_batches = min(5, len(self.subset))
            logger.info(f"Using {num_batches} batches for importance calculation")
            
            # Compute raw importance scores
            for batch_idx, batch in enumerate(self.subset[:num_batches]):
                self.model.zero_grad()
                
                # Single forward and backward pass
                with torch.set_grad_enabled(True):
                    with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                        
                        logits = outputs.logits
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            batch['input_ids'].view(-1)
                        )
                        
                        loss.backward()
                
                # Compute raw importance for all units using the same gradients
                for unit in pruning_units:
                    importance = 0.0
                    for name, (param, slice_idx) in unit.param_references.items():
                        if param.grad is not None:
                            if isinstance(slice_idx, tuple):
                                grad = param.grad[slice_idx[0], slice_idx[1]]
                                weight = param.data[slice_idx[0], slice_idx[1]]
                            else:
                                grad = param.grad[slice_idx]
                                weight = param.data[slice_idx]
                            importance += (weight.abs() * grad.abs()).sum().item()
                    
                    scores_dict[unit.id] += importance
                    
                    # Log first batch progress
                    if batch_idx == 0:
                        logger.debug(f"Unit {unit.id} raw importance: {importance:.6f}")
                
                torch.cuda.empty_cache()
                logger.info(f"Processed batch {batch_idx + 1}/{num_batches}")
            
            # Average scores across batches
            for unit in pruning_units:
                unit.importance_score = scores_dict[unit.id] / num_batches
            
            # Convert to numpy for statistical calculations
            raw_scores = np.array([unit.importance_score for unit in pruning_units])
            
            # Log raw score statistics
            logger.info("\nRaw Score Statistics:")
            logger.info(f"Mean: {np.mean(raw_scores):.6f}")
            logger.info(f"Std: {np.std(raw_scores):.6f}")
            logger.info(f"Min: {np.min(raw_scores):.6f}")
            logger.info(f"Max: {np.max(raw_scores):.6f}")
            
            # Apply z-score normalization globally
            mean_score = np.mean(raw_scores)
            std_score = np.std(raw_scores)
            
            if std_score > 0:
                logger.info("\nApplying z-score normalization...")
                for unit in pruning_units:
                    unit.importance_score = (unit.importance_score - mean_score) / std_score
            else:
                logger.warning("Zero standard deviation in importance scores - using raw scores")
            
            # Sort units by normalized importance (higher z-scores = more important)
            pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
            
            # Validate scores
            if not self.validate_scores(pruning_units):
                logger.warning("Score validation failed - check results carefully")
            
            # Log distribution of normalized scores
            normalized_scores = np.array([unit.importance_score for unit in pruning_units])
            logger.info("\nNormalized Score Statistics:")
            logger.info(f"Mean: {np.mean(normalized_scores):.6f}")
            logger.info(f"Std: {np.std(normalized_scores):.6f}")
            logger.info(f"Min: {np.min(normalized_scores):.6f}")
            logger.info(f"Max: {np.max(normalized_scores):.6f}")
            
            # Log top and bottom units
            logger.info("\nTop 10 most important units:")
            for unit in pruning_units[:10]:
                logger.info(f"{unit.id}: {unit.importance_score:.6f}")
            
            logger.info("\nBottom 10 least important units:")
            for unit in pruning_units[-10:]:
                logger.info(f"{unit.id}: {unit.importance_score:.6f}")
            
            # Optional: Log layer-wise statistics
            layer_scores = {}
            for unit in pruning_units:
                layer_idx = unit.layer_idx
                if layer_idx not in layer_scores:
                    layer_scores[layer_idx] = []
                layer_scores[layer_idx].append(unit.importance_score)
            
            logger.info("\nLayer-wise Score Statistics:")
            for layer_idx, scores in sorted(layer_scores.items()):
                scores_array = np.array(scores)
                logger.info(f"Layer {layer_idx}:")
                logger.info(f"  Mean: {np.mean(scores_array):.6f}")
                logger.info(f"  Std: {np.std(scores_array):.6f}")
                logger.info(f"  Units: {len(scores)}")
            
            return pruning_units
            
        finally:
            if was_training:
                self.model.train()
            torch.cuda.empty_cache()

    # def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
    #     """Compute importance scores for all units more efficiently"""
    #     logger.info(f"Computing importance scores using {self.method} method")
    #     was_training = self.model.training
    #     self.model.eval()
        
    #     try:
    #         # Initialize importance scores
    #         scores_dict = {unit.id: 0.0 for unit in pruning_units}
            
    #         # Process first 5 batches for speed
    #         num_batches = min(5, len(self.subset))
            
    #         for batch_idx, batch in enumerate(self.subset[:num_batches]):
    #             self.model.zero_grad()
                
    #             # Single forward and backward pass
    #             with torch.set_grad_enabled(True):
    #                 with torch.amp.autocast('cuda', enabled=self.use_mixed_precision):
    #                     outputs = self.model(
    #                         input_ids=batch['input_ids'],
    #                         attention_mask=batch['attention_mask']
    #                     )
                        
    #                     logits = outputs.logits
    #                     loss = F.cross_entropy(
    #                         logits.view(-1, logits.size(-1)),
    #                         batch['input_ids'].view(-1)
    #                     )
                        
    #                     loss.backward()
                
    #             # Compute importance for all units using the same gradients
    #             for unit in pruning_units:
    #                 importance = 0.0
    #                 for name, (param, slice_idx) in unit.param_references.items():
    #                     if param.grad is not None:
    #                         if isinstance(slice_idx, tuple):
    #                             grad = param.grad[slice_idx[0], slice_idx[1]]
    #                             weight = param.data[slice_idx[0], slice_idx[1]]
    #                         else:
    #                             grad = param.grad[slice_idx]
    #                             weight = param.data[slice_idx]
    #                         importance += (weight.abs() * grad.abs()).sum().item()
                    
    #                 scores_dict[unit.id] += importance
                    
    #                 if batch_idx == 0:  # Log only first batch for visibility
    #                     logger.info(f"Unit {unit.id}: {importance:.6f}")
                
    #             torch.cuda.empty_cache()
            
    #         # Assign averaged scores to units
    #         for unit in pruning_units:
    #             unit.importance_score = scores_dict[unit.id] / num_batches
            
    #         # Normalize scores
    #         max_score = max(unit.importance_score for unit in pruning_units)
    #         if max_score > 0:
    #             for unit in pruning_units:
    #                 unit.importance_score /= max_score
            
    #         # Sort units by score
    #         pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
            
    #         if not self.validate_scores(pruning_units):
    #             logger.warning("Score validation failed - check results carefully")
            
    #         logger.info("\nTop 10 importance scores:")
    #         for unit in pruning_units[:10]:
    #             logger.info(f"{unit.id}: {unit.importance_score:.6f}")
            
    #         return pruning_units
            
    #     finally:
    #         if was_training:
    #             self.model.train()
    #         torch.cuda.empty_cache()

    def validate_scores(self, pruning_units: List[PruningUnit]) -> bool:
        """Validate computed importance scores"""
        scores = [unit.importance_score for unit in pruning_units if hasattr(unit, 'importance_score')]
        
        if not scores:
            logger.error("No importance scores found")
            return False
        
        if all(s == 0 for s in scores):
            logger.error("All importance scores are zero")
            return False
        
        if any(not isinstance(s, float) for s in scores):
            logger.error("Invalid score type found")
            return False
        
        if self.config.get('score_range_check', True):
            if any(s < 0 for s in scores):
                logger.error("Negative scores found")
                return False
        
        min_non_zero = self.config.get('validation', {}).get('min_non_zero_scores', 0.01)
        non_zero_ratio = sum(1 for s in scores if s > 0) / len(scores)
        if non_zero_ratio < min_non_zero:
            logger.error(f"Too few non-zero scores: {non_zero_ratio:.2%}")
            return False
        
        return True