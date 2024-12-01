import torch
import torch.nn as nn
from typing import Dict, List
from .dependency_graph import PruningUnit
from .layerwrapper import BiasGPT
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ImportanceScorer:
    """Computes importance scores for pruning units using FLAP metrics"""
    
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
        
        # FLAP uses WIFN as default metric
        self.metric_type = config.get('metric_type', 'WIFN')
        self.wrapped_layers = {}
        
        logger.info(f"Initialized ImportanceScorer with metric type: {self.metric_type}")
        
    def compute_importance(self, pruning_unit: PruningUnit) -> float:
        """
        Compute importance score for a pruning unit using selected metric
        Args:
            pruning_unit: Unit to evaluate (attention head or MLP neurons)
        Returns:
            float: Importance score
        """
        try:
            # Initialize wrapped layers if not done
            if not self.wrapped_layers:
                self._initialize_wrapped_layers(pruning_unit)
            
            importance = 0.0
            for name, weight in pruning_unit.parameters.items():
                if name not in self.wrapped_layers:
                    continue
                
                wrapper = self.wrapped_layers[name]
                
                if self.metric_type == 'WIFN':
                    # Weight magnitude * input activation norm
                    importance += (torch.abs(weight) * 
                                 torch.sqrt(wrapper.scaler_inp.reshape((1, -1)))).mean().item()
                    
                elif self.metric_type == 'WIFV':
                    # Input variance weighted by squared weights
                    importance += (wrapper.fluc_inp * 
                                 torch.sum(weight.pow(2), dim=0)).mean().item()
                    
                else:  # IFV
                    # Pure input feature variance
                    importance += wrapper.fluc_inp.mean().item()
            
            pruning_unit.importance_score = importance
            return importance
            
        except Exception as e:
            logger.error(f"Error computing importance for unit {pruning_unit.id}: {str(e)}")
            raise
    
    def _initialize_wrapped_layers(self, unit: PruningUnit):
        """Initialize FLAP layer wrappers and collect statistics"""
        try:
            logger.info("Initializing wrapped layers...")
            
            # Create wrapped layers
            for name, params in unit.parameters.items():
                module = params.module if hasattr(params, 'module') else None
                if module is not None:
                    self.wrapped_layers[name] = BiasGPT(module, self.metric_type)
            
            # Collect statistics through forward passes
            for batch_idx, batch in enumerate(self.dataloader):
                if batch_idx >= self.N:
                    break
                    
                # Move batch to device
                inputs = batch[0].to(self.device)
                
                # Update statistics for each layer
                for wrapper in self.wrapped_layers.values():
                    wrapper.add_batch(inputs, None)
                    
            logger.info(f"Processed {batch_idx + 1} batches for statistics collection")
            
            # Finalize statistics
            for wrapper in self.wrapped_layers.values():
                wrapper.free()
                
        except Exception as e:
            logger.error(f"Error initializing wrapped layers: {str(e)}")
            raise
    
    def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
        """
        Compute importance scores for a group of pruning units
        Args:
            pruning_units: List of units to evaluate
        Returns:
            List[PruningUnit]: Units sorted by importance score
        """
        logger.info(f"Computing importance scores for {len(pruning_units)} units using {self.metric_type}")
        
        # Process each unit
        for unit in tqdm(pruning_units, desc="Calculating importance scores"):
            score = self.compute_importance(unit)
            logger.debug(f"Unit {unit.id} importance: {score:.4f}")
        
        # Clean up resources
        for wrapper in self.wrapped_layers.values():
            wrapper.free()
        self.wrapped_layers = {}
        torch.cuda.empty_cache()
        
        # Sort by importance score
        pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
        
        return pruning_units