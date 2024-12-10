#pruning_env.py

import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from .dependency_graph import PruningUnit
from .metrics import MetricsTracker, ModelMetrics

logger = logging.getLogger(__name__)

@dataclass
class PruningState:
    """Tracks the state of pruning process"""
    current_step: int = 0
    total_pruned: int = 0
    pruned_groups: List[str] = None
    initial_metrics: Optional[ModelMetrics] = None
    current_metrics: Optional[ModelMetrics] = None
    
    def __post_init__(self):
        if self.pruned_groups is None:
            self.pruned_groups = []

class PruningEnvironment(gym.Env):
    """Environment for learning pruning strategies"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 pruning_units: List['PruningUnit'],
                 eval_dataloader: torch.utils.data.DataLoader,
                 metrics_tracker: MetricsTracker,
                 config: Dict[str, Any],
                 device: torch.device,
                 initial_metrics: ModelMetrics):
        super().__init__()
        
        self.model = model
        self.pruning_units = pruning_units
        self.eval_dataloader = eval_dataloader
        self.metrics_tracker = metrics_tracker
        self.config = config
        self.device = device
        
        # Get number of layers for features
        num_layers = len(set(unit.layer_idx for unit in pruning_units))
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'global_features': spaces.Box(
                low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
            ),
            'unit_features': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(len(pruning_units), 4), dtype=np.float32
            ),
            'layer_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(num_layers, 2), dtype=np.float32
            )
        })
        
        # Action space: can prune multiple units per step
        self.max_prune_per_step = self.config['pruning']['env'].get('max_prune_per_step', 5)
        self.action_space = spaces.MultiBinary(len(pruning_units))
        
        # Initialize state with provided metrics
        self.state = PruningState(
            initial_metrics=initial_metrics,
            current_metrics=initial_metrics
        )
        
        # Save original model state
        self.original_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        
        # Create layer mapping for efficient access
        self.layer_indices = {
            idx: [i for i, unit in enumerate(pruning_units) if unit.layer_idx == idx]
            for idx in set(unit.layer_idx for unit in pruning_units)
        }
        
        logger.info(f"Initialized PruningEnvironment with {len(pruning_units)} prunable units "
                   f"across {len(self.layer_indices)} layers")
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Restore model
        self.model.load_state_dict(self.original_state)
        
        # Reset state keeping initial metrics
        self.state = PruningState(
            initial_metrics=self.state.initial_metrics,
            current_metrics=self.state.initial_metrics
        )
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute pruning actions"""
        try:
            # Convert action to numpy if it's a tensor
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            
            # Flatten if needed
            action = action.reshape(-1)
            
            # Make sure action is boolean array
            action = action.astype(bool)
            
            # Count number of True values
            num_actions = int(np.sum(action))
            logger.debug(f"Number of actions: {num_actions}")
            
            # Track pruned units
            newly_pruned = []
            
            # Execute pruning for each True value in action
            true_indices = np.where(action)[0]
            for idx in true_indices:
                if not self._is_pruned(idx):  # Check if not already pruned
                    unit = self.pruning_units[idx]
                    self._prune_unit(unit)
                    newly_pruned.append(unit.id)
                    self.state.pruned_groups.append(unit.id)
                    self.state.total_pruned += 1
                    logger.debug(f"Pruned unit {unit.id} at index {idx}")
            
            self.state.current_step += 1
            logger.debug(f"Step {self.state.current_step}: Pruned {len(newly_pruned)} units")
            
            # Evaluate if any units were pruned
            if newly_pruned:  # List is non-empty
                try:
                    self.state.current_metrics = self.metrics_tracker.evaluate_model(
                        self.model,
                        self.eval_dataloader
                    )
                except Exception as e:
                    logger.error(f"Error evaluating model: {str(e)}")
                    # Use previous metrics if evaluation fails
                    pass
            
            # Calculate reward
            reward = self._calculate_reward(newly_pruned)
            
            # Check termination
            terminated = self._is_terminated()
            truncated = False
            
            # Info dict
            info = {
                'newly_pruned': newly_pruned,
                'accuracy': float(self.state.current_metrics.accuracy),
                'compression_ratio': float(self.state.total_pruned / len(self.pruning_units)),
                'termination_reason': self._get_termination_reason() if terminated else None
            }
            
            return self._get_observation(), reward, terminated, truncated, info
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            logger.error(f"Action shape before flatten: {action.shape}, dtype: {action.dtype}")
            logger.error(f"Action values: {action}")
            # Return observation with error info
            error_info = {'error': str(e), 'action_shape': action.shape, 'action_dtype': str(action.dtype)}
            return self._get_observation(), -1.0, True, True, error_info
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current state observation"""
        # Global features
        accuracy_ratio = self.state.current_metrics.accuracy / self.state.initial_metrics.accuracy
        compression_ratio = self.state.total_pruned / len(self.pruning_units)
        progress = min(1.0, compression_ratio / self.config['pruning']['targets']['compression_target'])
        
        global_features = np.array([
            accuracy_ratio,
            compression_ratio,
            progress
        ], dtype=np.float32)
        
        # Unit features
        unit_features = []
        for idx, unit in enumerate(self.pruning_units):
            layer_units = self.layer_indices[unit.layer_idx]
            neighbors_pruned = sum(1 for i in layer_units 
                                 if i != idx and self._is_pruned(i))
            
            unit_features.append([
                unit.importance_score,
                1.0 if self._is_pruned(idx) else 0.0,
                unit.layer_idx / len(self.layer_indices),
                neighbors_pruned / len(layer_units)
            ])
        
        # Layer features
        layer_features = []
        for layer_idx, units in self.layer_indices.items():
            pruned_ratio = sum(1 for i in units if self._is_pruned(i)) / len(units)
            avg_importance = np.mean([self.pruning_units[i].importance_score for i in units])
            layer_features.append([pruned_ratio, avg_importance])
            
        return {
            'global_features': global_features,
            'unit_features': np.array(unit_features, dtype=np.float32),
            'layer_features': np.array(layer_features, dtype=np.float32)
        }
    
    def _calculate_reward(self, newly_pruned: List[str]) -> float:
        """Calculate reward based on pruning results"""
        if not newly_pruned:
            return 0.0
            
        # Get current metrics
        accuracy_ratio = self.state.current_metrics.accuracy / self.state.initial_metrics.accuracy
        accuracy_delta = accuracy_ratio - self.last_accuracy_ratio if hasattr(self, 'last_accuracy_ratio') else 0
        self.last_accuracy_ratio = accuracy_ratio
        
        compression_ratio = self.state.total_pruned / len(self.pruning_units)
        target_compression = self.config['pruning']['targets']['compression_target']
        compression_progress = compression_ratio / target_compression
        
        # Layer balance reward
        layer_pruned_ratios = []
        for layer_units in self.layer_indices.values():
            ratio = sum(1 for i in layer_units if self._is_pruned(i)) / len(layer_units)
            layer_pruned_ratios.append(ratio)
        layer_balance = -np.std(layer_pruned_ratios)
        
        # Combine rewards using weights from config
        weights = self.config['pruning']['env']['reward_weights']
        reward = (
            weights['accuracy'] * accuracy_delta +
            weights['compression'] * (compression_progress / len(newly_pruned)) +
            weights['balance'] * layer_balance
        )
        
        # Add penalties for violations
        if accuracy_ratio < self.config['pruning']['targets']['min_accuracy_ratio']:
            reward -= weights.get('violation_penalty', 2.0)
            
        return reward
    
    def _is_pruned(self, idx: int) -> bool:
        """Check if unit is pruned"""
        return self.pruning_units[idx].id in self.state.pruned_groups
    
    def _prune_unit(self, unit: 'PruningUnit'):
        """Prune unit by zeroing parameters"""
        with torch.no_grad():
            for param in unit.parameters.values():
                param.zero_()
    
    def _is_terminated(self) -> bool:
        """Check if pruning should terminate based on config targets"""
        accuracy = self.state.current_metrics.accuracy
        accuracy_ratio = self.state.current_metrics.accuracy / self.state.initial_metrics.accuracy
        pruned_ratio = self.state.total_pruned / len(self.pruning_units)
        
        targets = self.config['pruning']['targets']
        return (
            accuracy < targets['min_accuracy'] or
            accuracy_ratio < targets['min_accuracy_ratio'] or
            self.state.total_pruned >= targets['max_pruned_heads'] or
            pruned_ratio >= targets['compression_target']
        )
    
    def _get_termination_reason(self) -> str:
        """Get reason for termination"""
        accuracy = self.state.current_metrics.accuracy
        accuracy_ratio = self.state.current_metrics.accuracy / self.state.initial_metrics.accuracy
        pruned_ratio = self.state.total_pruned / len(self.pruning_units)
        
        targets = self.config['pruning']['targets']
        
        if accuracy < targets['min_accuracy']:
            return 'min_accuracy'
        elif accuracy_ratio < targets['min_accuracy_ratio']:
            return 'min_accuracy_ratio'
        elif self.state.total_pruned >= targets['max_pruned_heads']:
            return 'max_pruned_heads'
        elif pruned_ratio >= targets['compression_target']:
            return 'compression_target'
        else:
            return 'unknown'
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of pruning state"""
        layer_stats = {}
        for layer_idx, units in self.layer_indices.items():
            pruned = sum(1 for i in units if self._is_pruned(i))
            layer_stats[f'layer_{layer_idx}'] = {
                'pruned': pruned,
                'total': len(units),
                'ratio': pruned / len(units)
            }
            
        return {
            'pruned_units': self.state.total_pruned,
            'total_units': len(self.pruning_units),
            'compression_ratio': self.state.total_pruned / len(self.pruning_units),
            'accuracy': self.state.current_metrics.accuracy,
            'accuracy_ratio': self.state.current_metrics.accuracy / self.state.initial_metrics.accuracy,
            'steps': self.state.current_step,
            'layer_statistics': layer_stats
        }