import torch
import torch.nn as nn
from typing import Dict, List
from .dependency_graph import PruningUnit
from .layerwrapper import BiasGPT
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImportanceScorer:
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
        self.N = config.get('num_samples', 100)
        self.metric_type = config.get('metric_type', 'WIFN')
        self.wrapped_layers = {}
        
        logger.info(f"Initialized ImportanceScorer with metric type: {self.metric_type}")
    
    def _initialize_wrapped_layers(self, unit: PruningUnit):
        """Initialize wrapped layers and collect statistics"""
        try:
            logger.info("Initializing wrapped layers...")
            
            # Create wrapped layers
            for name, params in unit.parameters.items():
                module = params.module if hasattr(params, 'module') else None
                if module is not None:
                    self.wrapped_layers[name] = BiasGPT(module, self.metric_type)
            
            # Collect statistics through forward passes
            for batch_idx, batch in enumerate(tqdm(self.dataloader, 
                                                 desc="Collecting layer statistics",
                                                 leave=False)):
                if batch_idx >= self.N:
                    break
                
                # MMLU dataloader returns dictionary with 'input_ids' and 'attention_mask'
                inputs = batch['input_ids'].to(self.device)
                
                # Update statistics for each layer
                for name, wrapper in self.wrapped_layers.items():
                    wrapper.add_batch(inputs, None)
                    
            logger.info(f"Processed {batch_idx + 1} batches for statistics collection")
            
            # Finalize statistics
            for wrapper in self.wrapped_layers.values():
                wrapper.free()
                
        except Exception as e:
            logger.error(f"Error initializing wrapped layers: {str(e)}")
            raise

    def compute_importance(self, pruning_unit: PruningUnit) -> float:
        """Compute importance score for a pruning unit"""
        try:
            if not self.wrapped_layers:
                self._initialize_wrapped_layers(pruning_unit)
            
            importance = 0.0
            for name, weight in pruning_unit.parameters.items():
                if name not in self.wrapped_layers:
                    continue
                
                wrapper = self.wrapped_layers[name]
                
                if self.metric_type == 'WIFN':
                    importance += (torch.abs(weight) * 
                                 torch.sqrt(wrapper.scaler_inp.reshape((1, -1)))).mean().item()
                elif self.metric_type == 'WIFV':
                    importance += (wrapper.fluc_inp * 
                                 torch.sum(weight.pow(2), dim=0)).mean().item()
                else:  # IFV
                    importance += wrapper.fluc_inp.mean().item()
            
            pruning_unit.importance_score = importance
            return importance
            
        except Exception as e:
            logger.error(f"Error computing importance for unit {pruning_unit.id}: {str(e)}")
            raise
    
    def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
        """Compute importance scores for all units"""
        logger.info(f"Computing importance scores for {len(pruning_units)} units using {self.metric_type}")
        
        for unit in tqdm(pruning_units, desc="Calculating importance scores"):
            score = self.compute_importance(unit)
            logger.debug(f"Unit {unit.id} importance: {score:.4f}")
        
        # Clean up
        for wrapper in self.wrapped_layers.values():
            wrapper.free()
        self.wrapped_layers = {}
        torch.cuda.empty_cache()
        
        # Sort by importance score
        pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
        
        return pruning_units