import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from .dependency_graph import PruningUnit
import logging
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImportanceScorer:
   def __init__(self, model: nn.Module, tokenizer: Any, config: Dict,
               calibration_dataloader: torch.utils.data.DataLoader, device: torch.device):
       self.model = model
       self.tokenizer = tokenizer
       self.device = device
       self.config = config
       
       # Get calibration subset
       percent = config.get('calibration_percent', 0.01) 
       total_samples = len(calibration_dataloader.dataset)
       num_samples = int(total_samples * percent)
       
       subset_data = []
       samples_collected = 0
       for batch in calibration_dataloader:
           if samples_collected >= num_samples:
               break
           subset_data.append({
               'input_ids': batch['input_ids'].to(device),
               'attention_mask': batch['attention_mask'].to(device),
               'labels': batch['labels'].to(device) if 'labels' in batch else None
           })
           samples_collected += batch['input_ids'].size(0)
           
       self.subset = subset_data
       logger.info(f"Using {samples_collected} samples for importance scoring")

   # importance_scorer.py - key method update
   def compute_importance(self, pruning_unit: PruningUnit) -> float:
        importance_scores = []
        batch_size = 4  # Reduced batch size
        
        for batch in self.subset:
            # Split into smaller batches
            n_samples = batch['input_ids'].size(0)
            for i in range(0, n_samples, batch_size):
                batch_slice = {
                    'input_ids': batch['input_ids'][i:i+batch_size].clone(),
                    'attention_mask': batch['attention_mask'][i:i+batch_size].clone()
                }
                
                with torch.cuda.amp.autocast(), torch.no_grad():
                    outputs_normal = self.model(
                        input_ids=batch_slice['input_ids'],
                        attention_mask=batch_slice['attention_mask']
                    ).logits
                    
                    # Save and zero parameters
                    saved_tensors = {}
                    for name, param in pruning_unit.parameters.items():
                        saved_tensors[name] = param.data.clone()
                        param.data.zero_()
                    
                    outputs_zeroed = self.model(
                        input_ids=batch_slice['input_ids'],
                        attention_mask=batch_slice['attention_mask']
                    ).logits
                    
                    # Restore parameters
                    for name, param in pruning_unit.parameters.items():
                        param.data.copy_(saved_tensors[name])
                
                diff = F.mse_loss(outputs_normal, outputs_zeroed).item()
                importance_scores.append(diff)
                
                torch.cuda.empty_cache()
        
        return sum(importance_scores) / len(importance_scores)

   def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
       """Compute importance scores for all pruning units"""
       logger.info(f"Computing importance scores for {len(pruning_units)} units")
       was_training = self.model.training
       self.model.eval()
       
       try:
           for unit in tqdm(pruning_units, desc="Computing importance scores"):
               importance = self.compute_importance(unit)
               unit.importance_score = importance
               logger.info(f"Unit {unit.id}: {importance:.6f}")
           
           # Normalize scores
           scores = [unit.importance_score for unit in pruning_units]
           max_score = max(scores) if scores else 1.0
           if max_score > 0:
               for unit in pruning_units:
                   unit.importance_score /= max_score
           
           # Sort by importance
           pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
           
           # Log top scores
           logger.info("\n Daniel, Top 10 importance scores:")
           for unit in pruning_units[:10]:
               logger.info(f"{unit.id}: {unit.importance_score:.6f}")
           
           return pruning_units
           
       finally:
           if was_training:
               self.model.train()
           torch.cuda.empty_cache()

   def validate_scores(self, pruning_units: List[PruningUnit]) -> bool:
       """Validate computed importance scores"""
       scores = [unit.importance_score for unit in pruning_units]
       if not scores:
           return False
           
       if all(s == 0 for s in scores):
           logger.error("All importance scores are zero")
           return False
           
       if any(not isinstance(s, float) for s in scores):
           logger.error("Invalid score type found")
           return False
           
       return True