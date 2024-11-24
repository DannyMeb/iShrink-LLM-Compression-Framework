import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import gc
import warnings

logger = logging.getLogger(__name__)

@dataclass
class ImportanceMetrics:
    """Detailed metrics for importance scoring"""
    gradient_norm: float = 0.0
    activation_impact: float = 0.0
    fisher_info: float = 0.0
    combined_score: float = 0.0
    confidence: float = 0.0
    additional_metrics: Dict[str, float] = field(default_factory=dict)

class ImportanceScorer:
    def __init__(self,
             model: torch.nn.Module,
             tokenizer,
             config: Dict[str, Any],
             calibration_dataloader: torch.utils.data.DataLoader,
             device: Optional[str] = None):
        """Initialize importance scorer"""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataloader = calibration_dataloader
        self.score_cache = {}
        
        # Set model to eval mode but enable gradients
        self.model.eval()
        # Enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
            
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
         #Get importance methods and weights from config
        importance_config = config.get('pruning', {}).get('importance', {})
        self.methods = importance_config.get('methods', ['gradient', 'activation', 'fisher'])
        self.weights = importance_config.get('weights', [0.4, 0.3, 0.3])
        
        # Validate configuration
        self._validate_config()
        logger.info(f"Initialized ImportanceScorer with methods: {self.methods}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        valid_methods = {'gradient', 'activation', 'fisher'}
        if not all(method in valid_methods for method in self.methods):
            raise ValueError(f"Invalid importance methods. Must be in {valid_methods}")
        
        if not np.isclose(sum(self.weights), 1.0, rtol=1e-5):
            raise ValueError(f"Importance weights must sum to 1.0, got {sum(self.weights)}")
        
        if len(self.methods) != len(self.weights):
            raise ValueError("Number of methods must match number of weights")

    @torch.no_grad()
    def compute_group_importance(self,
                               group: 'PruningGroup',
                               use_cache: bool = True) -> ImportanceMetrics:
        """
        Compute importance metrics for parameter group
        
        Args:
            group: The parameter group to analyze
            use_cache: Whether to use cached results
            
        Returns:
            ImportanceMetrics containing all computed metrics
        """
        # Check cache
        if use_cache and group.id in self.score_cache:
            return self.score_cache[group.id]
        
        try:
            # Initialize metrics
            metrics = ImportanceMetrics()
            metrics.additional_metrics['start_memory'] = torch.cuda.memory_allocated() / 1024**2
            
            # Compute each importance metric
            methods_map = {
                'gradient': self._compute_gradient_importance,
                'activation': self._compute_activation_importance,
                'fisher': self._compute_fisher_importance
            }
            
            # Track computation time and memory for each method
            for method, weight in zip(self.methods, self.weights):
                if method not in methods_map:
                    continue
                    
                try:
                    torch.cuda.empty_cache()
                    start_mem = torch.cuda.memory_allocated()
                    
                    # Compute importance with memory optimization
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        value = methods_map[method](group)
                    
                    # Store results
                    setattr(metrics, f"{method}_norm", value)
                    metrics.combined_score += weight * value
                    
                    # Track memory usage
                    end_mem = torch.cuda.memory_allocated()
                    metrics.additional_metrics[f'{method}_memory'] = (end_mem - start_mem) / 1024**2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"OOM in {method}, trying with reduced precision")
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Retry with reduced precision
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            value = methods_map[method](group)
                            setattr(metrics, f"{method}_norm", value)
                            metrics.combined_score += weight * value
                    else:
                        raise
            
            # Compute confidence
            metrics.confidence = self._compute_confidence(metrics)
            
            # Cache results
            if use_cache:
                self.score_cache[group.id] = metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error computing importance for group {group.id}: {str(e)}")
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def _compute_gradient_importance(self, group: 'PruningGroup') -> float:
        """Compute gradient-based importance"""
        try:
            self.model.zero_grad()
            importance = 0.0
            num_batches = 0
            
            # Enable gradients for group parameters
            for param in group.parameters.values():
                if param is not None:
                    param.requires_grad = True
            
            for batch in self.dataloader:
                # Get input tensors and create labels
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                with torch.set_grad_enabled(True):
                    # Forward pass with gradient computation
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    
                    loss = outputs.loss
                    loss.backward()
                    
                    # Compute importance
                    for param in group.parameters.values():
                        if param.grad is not None:
                            importance += torch.abs(param.grad * param).sum().item()
                    
                    self.model.zero_grad()
                    num_batches += 1
                    
                    # Memory management
                    del outputs, loss
                    torch.cuda.empty_cache()
                
                # Move tensors to CPU to free GPU memory
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
            
            return importance / max(num_batches, 1)
            
        finally:
            self.model.zero_grad()
            torch.cuda.empty_cache()

    def _compute_activation_importance(self, group: 'PruningGroup') -> float:
        """Compute activation-based importance"""
        activation_values = []
        handles = []
        
        try:
            # Setup forward hooks
            def hook_fn(module, input, output):
                activation_values.append(output.abs().mean().item())
            
            # Register hooks for relevant modules
            for param_name in group.parameters:
                module = self.model
                for comp in param_name.split('.')[:-1]:
                    module = getattr(module, comp)
                handles.append(module.register_forward_hook(hook_fn))
            
            # Compute activations
            self.model.eval()
            with torch.no_grad():
                for batch in self.dataloader:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0].to(self.device)
                    else:
                        inputs = batch.to(self.device)
                        
                    self.model(inputs)
                    inputs = inputs.cpu()
                    torch.cuda.empty_cache()
            
            return np.mean(activation_values) if activation_values else 0.0
            
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
            torch.cuda.empty_cache()

    def _compute_fisher_importance(self, group: 'PruningGroup') -> float:
        """Compute Fisher Information based importance"""
        fisher_values = []
        num_samples = min(self.config.get('num_samples', 10), len(self.dataloader))
        
        try:
            for i, batch in enumerate(self.dataloader):
                if i >= num_samples:
                    break
                    
                self.model.zero_grad()
                
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                logits = outputs.logits
                
                # Sample from output distribution
                probs = F.softmax(logits, dim=-1)
                sampled_indices = torch.multinomial(probs, 1).squeeze()
                
                # Compute gradients
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = log_probs[range(len(inputs)), sampled_indices]
                loss = -selected_log_probs.mean()
                loss.backward()
                
                # Compute Fisher values
                for param in group.parameters.values():
                    if param.grad is not None:
                        fisher_values.append(param.grad.pow(2).mean().item())
                
                self.model.zero_grad()
                inputs = inputs.cpu()
                del outputs, logits, probs, log_probs
                torch.cuda.empty_cache()
            
            return np.mean(fisher_values) if fisher_values else 0.0
            
        finally:
            self.model.zero_grad()
            torch.cuda.empty_cache()

    def _compute_confidence(self, metrics: ImportanceMetrics) -> float:
        """Compute confidence score based on metric agreement"""
        values = []
        for method in self.methods:
            value = getattr(metrics, f"{method}_norm", None)
            if value is not None:
                values.append(value)
        
        if not values:
            return 0.0
        
        # Normalize values
        values = np.array(values)
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val > 1e-8:
            values = (values - min_val) / (max_val - min_val)
        
        # Compute agreement as inverse of standard deviation
        return 1.0 - np.std(values)

    def get_importance_ranking(self, groups: List['PruningGroup']) -> List[Tuple[str, float]]:
        """Get groups ranked by importance"""
        scores = []
        for group in tqdm(groups, desc="Computing importance scores"):
            try:
                score = self.compute_group_importance(group).combined_score
                scores.append((group.id, score))
            except Exception as e:
                logger.error(f"Error computing score for group {group.id}: {str(e)}")
                scores.append((group.id, float('-inf')))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def save_scores(self, path: str):
        """Save importance scores to file"""
        save_data = {
            'scores': self.score_cache,
            'config': self.config,
            'methods': self.methods,
            'weights': self.weights
        }
        torch.save(save_data, path)
        logger.info(f"Saved importance scores to {path}")

    def load_scores(self, path: str):
        """Load importance scores from file"""
        load_data = torch.load(path)
        self.score_cache = load_data['scores']
        self.config = load_data['config']
        self.methods = load_data['methods']
        self.weights = load_data['weights']
        logger.info(f"Loaded importance scores from {path}")

    def clear_cache(self):
        """Clear the score cache"""
        self.score_cache.clear()
        torch.cuda.empty_cache()
        gc.collect()