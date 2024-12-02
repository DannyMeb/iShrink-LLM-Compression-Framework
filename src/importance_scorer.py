#importance_scorer.py

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
            print(f"\nDebug: Starting initialization for unit {unit.id}")
            
            # Create wrapped layers for each parameter tensor
            for name, param_tensor in unit.parameters.items():
                print(f"\nDebug: Creating wrapper for {name}")
                print(f"Parameter shape: {param_tensor.shape}")
                print(f"Parameter stats: min={param_tensor.min().item():.6f}, "
                      f"max={param_tensor.max().item():.6f}, "
                      f"mean={param_tensor.mean().item():.6f}")
                
                # Create wrapper directly with parameter tensor
                self.wrapped_layers[name] = BiasGPT(
                    param_tensor, 
                    self.metric_type,
                    device=self.device
                )
            
            # Collect statistics through forward passes
            for batch_idx, batch in enumerate(tqdm(self.dataloader, 
                                                 desc="Collecting statistics",
                                                 leave=False)):
                if batch_idx >= self.N:
                    break
                
                # Process input batch
                inputs = batch['input_ids'].to(self.device)
                print(f"\nProcessing batch {batch_idx}")
                print(f"Input shape: {inputs.shape}")
                print(f"Input stats: min={inputs.min().item()}, "
                      f"max={inputs.max().item()}, "
                      f"mean={inputs.float().mean().item():.6f}")
                
                # Update statistics for all layers
                for name, wrapper in self.wrapped_layers.items():
                    wrapper.add_batch(inputs)
                    
                    # Debug statistics after update
                    if hasattr(wrapper, 'scaler_inp'):
                        print(f"\nLayer {name} stats after batch {batch_idx}:")
                        print(f"Scaler stats: min={wrapper.scaler_inp.min().item():.6f}, "
                              f"max={wrapper.scaler_inp.max().item():.6f}, "
                              f"mean={wrapper.scaler_inp.mean().item():.6f}")
            
            logger.info(f"Processed {batch_idx + 1} batches for statistics collection")
                
        except Exception as e:
            logger.error(f"Error initializing wrapped layers: {str(e)}")
            raise

    def compute_importance(self, pruning_unit: PruningUnit) -> float:
        """Compute importance score for a pruning unit"""
        try:
            print(f"\nComputing importance for unit {pruning_unit.id}")
            
            if not self.wrapped_layers:
                self._initialize_wrapped_layers(pruning_unit)
            
            importance = 0.0
            for name, weight in pruning_unit.parameters.items():
                print(f"\nProcessing parameter: {name}")
                
                if name not in self.wrapped_layers:
                    logger.warning(f"Parameter {name} not found in wrapped layers")
                    continue
                
                wrapper = self.wrapped_layers[name]
                
                if self.metric_type == 'WIFN':
                    # Debug intermediate computations
                    abs_weights = torch.abs(weight)
                    scaler = wrapper.scaler_inp
                    sqrt_scaler = torch.sqrt(scaler.clamp(min=1e-10))
                    product = abs_weights * sqrt_scaler.view(1, -1)
                    importance_score = product.mean().item()
                    
                    print(f"\nWIFN computation for {name}:")
                    print(f"Weight stats: min={abs_weights.min().item():.6f}, "
                          f"max={abs_weights.max().item():.6f}")
                    print(f"Scaler stats: min={sqrt_scaler.min().item():.6f}, "
                          f"max={sqrt_scaler.max().item():.6f}")
                    print(f"Product stats: min={product.min().item():.6f}, "
                          f"max={product.max().item():.6f}")
                    print(f"Importance score: {importance_score:.6f}")
                    
                    importance += importance_score
                    
                elif self.metric_type == 'WIFV':
                    fluc = wrapper.fluc_inp
                    weight_sq = torch.sum(weight.pow(2), dim=0)
                    importance_score = (fluc * weight_sq).mean().item()
                    importance += importance_score
                else:  # IFV
                    importance += wrapper.fluc_inp.mean().item()
            
            print(f"\nFinal importance for unit {pruning_unit.id}: {importance:.6f}")
            pruning_unit.importance_score = importance
            return importance
            
        except Exception as e:
            logger.error(f"Error computing importance for unit {pruning_unit.id}: {str(e)}")
            raise
    
    def compute_group_importances(self, pruning_units: List[PruningUnit]) -> List[PruningUnit]:
        """Compute importance scores for all units"""
        logger.info(f"Computing importance scores for {len(pruning_units)} units "
                   f"using {self.metric_type}")
        
        for unit in tqdm(pruning_units, desc="Calculating importance scores"):
            score = self.compute_importance(unit)
            logger.info(f"Unit {unit.id} importance: {score:.4f}")
        
        # Clean up
        for wrapper in self.wrapped_layers.values():
            wrapper.free()
        self.wrapped_layers = {}
        torch.cuda.empty_cache()
        
        # Sort by importance score
        pruning_units.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Log sorted scores
        print("\nFinal sorted importance scores:")
        for unit in pruning_units[:10]:
            print(f"{unit.id}: {unit.importance_score:.6f}")
        
        return pruning_units