import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import logging
from .utils import setup_device, create_dataloader
from .dependency_graph import PruningGroup
import gc

logger = logging.getLogger(__name__)

@dataclass
class ImportanceMetrics:
    """Detailed metrics for importance scoring"""
    gradient_norm: float
    activation_impact: float
    fisher_info: Optional[float] = None
    combined_score: float = 0.0
    confidence: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class ImportanceScorer:
    def __init__(self,
                 model: torch.nn.Module,
                 tokenizer,
                 config: Dict[str, Any],
                 calibration_data: torch.Tensor,
                 device: Optional[str] = None):
        """
        Initialize importance scorer with model and configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = setup_device(device)
        self.score_cache = {}
        
        # Setup data
        self.dataloader = create_dataloader(
            calibration_data,
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        # Validation
        self._validate_config()
        logger.info(f"Initialized ImportanceScorer with methods: {config['methods']}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_methods = set(['gradient', 'activation', 'fisher'])
        if not all(method in required_methods for method in self.config['methods']):
            raise ValueError(f"Invalid importance methods. Must be in {required_methods}")
        
        if sum(self.config['weights']) != 1.0:
            raise ValueError("Importance weights must sum to 1.0")
    
    def compute_group_importance(self,
                            group: PruningGroup,
                            use_cache: bool = True) -> ImportanceMetrics:
        """Compute importance metrics for parameter group with memory optimization"""
        # Check cache first
        if use_cache and group.id in self.score_cache:
            return self.score_cache[group.id]
        
        # Clear GPU cache before computation
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # Use automatic mixed precision for better memory efficiency
            with torch.cuda.amp.autocast():
                metrics = ImportanceMetrics(
                    gradient_norm=0.0,
                    activation_impact=0.0,
                    fisher_info=0.0,
                    additional_metrics={
                        'memory_usage': torch.cuda.memory_allocated() / 1024**2
                    }
                )
                
                # Compute different importance metrics
                methods_map = {
                    'gradient': self._compute_gradient_importance,
                    'activation': self._compute_activation_importance,
                    'fisher': self._compute_fisher_importance
                }
                
                # Prepare weights and methods
                methods = self.config['methods']
                weights = self.config['weights']
                
                # Compute each importance metric with memory management
                for method, weight in zip(methods, weights):
                    if method in methods_map:
                        # Clear cache before each computation
                        torch.cuda.empty_cache()
                        
                        # Compute importance with smaller batch if needed
                        try:
                            value = methods_map[method](group)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                logger.warning(f"OOM in {method}, trying with reduced batch")
                                # Reduce batch size temporarily
                                original_batch_size = self.config.get('batch_size', 32)
                                self.config['batch_size'] = max(1, original_batch_size // 2)
                                value = methods_map[method](group)
                                self.config['batch_size'] = original_batch_size
                        
                        # Set the computed value
                        setattr(metrics, f"{method}_importance", value)
                        metrics.combined_score += weight * value
                        
                        # Track memory usage
                        metrics.additional_metrics[f'{method}_memory'] = (
                            torch.cuda.memory_allocated() / 1024**2
                        )
                
                # Compute confidence based on metric agreement
                metrics.confidence = self._compute_confidence(metrics)
                
                # Cache the results
                if use_cache:
                    self.score_cache[group.id] = metrics
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error computing importance for group {group.id}: {str(e)}")
            raise
        finally:
            # Final cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    def _compute_gradient_importance(self, group: PruningGroup) -> float:
        """Compute gradient-based importance with memory optimization"""
        try:
            with torch.cuda.amp.autocast():
                self.model.zero_grad()
                importance = 0.0
                
                # Process in smaller batches if needed
                batch_size = self.config.get('batch_size', 32)
                for batch in self._get_batches(batch_size):
                    outputs = self.model(batch)
                    loss = outputs.loss / batch_size  # Normalize by batch size
                    loss.backward()
                    
                    for param in group.parameters.values():
                        if param.grad is not None:
                            importance += torch.abs(param.grad * param).sum().item()
                    
                    # Clear gradients after each batch
                    self.model.zero_grad()
                    
                return importance
                
        finally:
            # Ensure gradients are cleared
            self.model.zero_grad()
            torch.cuda.empty_cache()

    def _get_batches(self, batch_size: int):
        """Helper to get data batches with memory consideration"""
        dataset_size = len(self.calibration_data)
        indices = torch.randperm(dataset_size)
        
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Move batch to GPU only when needed
            batch = self.calibration_data[batch_indices].to(self.device)
            yield batch
            # Move batch back to CPU
            batch = batch.cpu()
            torch.cuda.empty_cache()
    
    
    def _compute_activation_importance(self, group: PruningGroup) -> float:
        """Compute activation-based importance"""
        activation_values = []
        handles = []
        
        def hook_fn(module, input, output):
            activation_values.append(output.abs().mean().item())
        
        # Register hooks
        for param_name in group.parameters:
            if '.' in param_name:
                module_path = param_name.split('.')[:-1]
                module = self.model
                for comp in module_path:
                    module = getattr(module, comp)
                handles.append(module.register_forward_hook(hook_fn))
        
        # Compute activations
        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                inputs = batch[0].to(self.device)
                self.model(inputs)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return np.mean(activation_values) if activation_values else 0.0
    
    def _compute_fisher_importance(self, group: PruningGroup) -> float:
        """Compute Fisher Information based importance"""
        if 'fisher' not in self.config['methods']:
            return None
        
        fisher_values = []
        num_samples = self.config.get('num_samples', 100)
        
        for _ in range(num_samples):
            self.model.zero_grad()
            batch = next(iter(self.dataloader))[0].to(self.device)
            
            outputs = self.model(batch)
            logits = outputs.logits
            
            # Sample from output distribution
            probs = F.softmax(logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze()
            
            # Compute gradients
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs[range(len(batch)), sampled_indices]
            loss = -selected_log_probs.mean()
            loss.backward()
            
            # Compute Fisher values
            for param in group.parameters.values():
                if param.grad is not None:
                    fisher_values.append(param.grad.pow(2).mean().item())
        
        return np.mean(fisher_values) if fisher_values else 0.0
    
    def _compute_confidence(self, metrics: ImportanceMetrics) -> float:
        """Compute confidence score based on metric agreement"""
        values = []
        for method in self.config['methods']:
            value = getattr(metrics, f"{method}_importance")
            if value is not None:
                values.append(value)
        
        if not values:
            return 0.0
        
        # Normalize values
        values = np.array(values)
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        
        # Compute agreement as inverse of standard deviation
        return 1.0 - np.std(values)
    
    def get_importance_ranking(self, groups: List[PruningGroup]) -> List[Tuple[str, float]]:
        """Get groups ranked by importance"""
        scores = [(group.id, self.compute_group_importance(group).combined_score)
                 for group in groups]
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def save_scores(self, path: str):
        """Save importance scores to file"""
        torch.save(self.score_cache, path)
    
    def load_scores(self, path: str):
        """Load importance scores from file"""
        self.score_cache = torch.load(path)