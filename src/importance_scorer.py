# src/importance_scorer.py

import torch
import torch.nn as nn
from typing import Dict, List
from .dependency_graph import PruningUnit
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImportanceScorer:
    """Computes importance scores for attention heads"""
    
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: any,
                 config: Dict,
                 calibration_dataloader: torch.utils.data.DataLoader,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.dataloader = calibration_dataloader
        self.device = device
        self.N = config.get('num_samples', 10)  # Number of samples to use
        
        # Loss function for importance calculation
        self.loss_fct = nn.CrossEntropyLoss()
    
    def compute_importance(self, pruning_unit: PruningUnit) -> float:
        """Compute importance score for a single attention head"""
        try:
            importance = self._compute_head_importance(pruning_unit)
            pruning_unit.importance_score = importance
            return importance
            
        except Exception as e:
            logger.error(f"Error computing importance for head {pruning_unit.id}: {str(e)}")
            raise RuntimeError(f"Error computing importance: {str(e)}")
    
    def _compute_head_importance(self, unit: PruningUnit) -> float:
        """Compute importance of an attention head using gradient-based method"""
        total_importance = 0.0
        samples_processed = 0
        
        self.model.train()  # Enable gradient computation
        
        # Get calibration samples
        for batch_idx, batch in enumerate(self.dataloader):
            if batch_idx >= self.N:
                break
                
            try:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Get logits and compute loss
                logits = outputs.logits
                
                # Prepare labels for language modeling
                # Shift right by 1 position
                labels = batch['input_ids'].clone()
                labels = labels[:, 1:].contiguous()
                logits = logits[:, :-1, :].contiguous()
                
                # Compute loss
                loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Backward pass
                loss.backward()
                
                # Compute importance for this batch
                importance = self._compute_gradient_based_importance(unit)
                total_importance += importance
                
                # Clear gradients
                self.model.zero_grad()
                
                samples_processed += 1
                
            except Exception as e:
                logger.warning(f"Error processing batch for head {unit.id}: {str(e)}")
                continue
        
        # Average importance over samples
        return total_importance / samples_processed if samples_processed > 0 else 0.0
    
    def _compute_gradient_based_importance(self, unit: PruningUnit) -> float:
        """Compute gradient-based importance for head parameters"""
        importance = 0.0
        
        for param_name, param in unit.parameters.items():
            if param.grad is not None:
                # Compute importance as |param * grad|
                importance += torch.abs(param * param.grad).sum().item()
        
        return importance
    
    def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
        """Compute importance scores for all heads"""
        logger.info(f"Computing importance scores for {len(pruning_units)} attention heads")
        
        for unit in tqdm(pruning_units, desc="Calculating importance scores"):
            score = self.compute_importance(unit)
            logger.debug(f"Head {unit.id} importance: {score:.4f}")
        
        # Sort units by importance score
        pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
        
        return pruning_units