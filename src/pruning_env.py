# src/pruning_env.py

import gym
from gym import spaces
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from .metrics import MetricsTracker, ModelMetrics

logger = logging.getLogger(__name__)

@dataclass
class PruningState:
    """Tracks the state of pruning process"""
    current_step: int = 0
    total_pruned: int = 0
    pruned_groups: List[str] = None  # IDs of pruned heads
    initial_metrics: Optional[ModelMetrics] = None
    current_metrics: Optional[ModelMetrics] = None
    
    def __post_init__(self):
        if self.pruned_groups is None:
            self.pruned_groups = []

class PruningEnvironment(gym.Env):
    """Environment for learning to prune transformer heads"""
    
    def __init__(self,
                 model: torch.nn.Module,
                 pruning_units: List['PruningUnit'],
                 eval_dataloader: torch.utils.data.DataLoader,
                 metrics_tracker: MetricsTracker,
                 config: Dict[str, Any],
                 device: torch.device):
        """Initialize pruning environment"""
        super().__init__()
        
        self.model = model
        self.pruning_units = pruning_units
        self.eval_dataloader = eval_dataloader
        self.metrics_tracker = metrics_tracker
        self.config = config
        self.device = device
        
        # Setup action and observation spaces
        self.action_space = spaces.Discrete(len(pruning_units))  # Choose which head to prune
        
        # State space: [importance_scores, pruning_status, global_metrics]
        state_dim = len(pruning_units) * 2 + 3  # 2 features per head + 3 global metrics
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Initialize state
        self.state = PruningState()
        
        # Evaluate and store initial metrics
        initial_metrics = self.metrics_tracker.evaluate_model(
            self.model,
            self.eval_dataloader
        )
        self.state.initial_metrics = initial_metrics
        self.state.current_metrics = initial_metrics
        
        # Save original model state
        self.original_state = {
            k: v.clone() for k, v in self.model.state_dict().items()
        }
        
        logger.info(f"Initial model metrics: accuracy={initial_metrics.accuracy:.4f}, "
                   f"perplexity={initial_metrics.perplexity:.4f}")
        logger.info(f"Initialized PruningEnvironment with {len(pruning_units)} prunable heads")
    
    def reset(self):
        """Reset environment to initial state"""
        # Restore original model
        self.model.load_state_dict(self.original_state)
        
        # Reset state
        self.state = PruningState()
        
        # Reset metrics to initial values
        self.state.initial_metrics = self.metrics_tracker.evaluate_model(
            self.model,
            self.eval_dataloader
        )
        self.state.current_metrics = self.state.initial_metrics
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute pruning action"""
        # Validate action
        if not self._is_valid_action(action):
            return self._get_state(), -1.0, True, {'invalid_action': True}
        
        try:
            # Get pruning unit to prune
            unit = self.pruning_units[action]
            
            # Prune head
            self._prune_head(unit)
            self.state.pruned_groups.append(unit.id)
            self.state.total_pruned += 1
            self.state.current_step += 1
            
            # Evaluate new model state
            self.state.current_metrics = self.metrics_tracker.evaluate_model(
                self.model,
                self.eval_dataloader
            )
            
            # Calculate reward
            reward = self._calculate_reward()
            
            # Check if done
            done = self._is_done()
            
            # Prepare info dict
            info = {
                'pruned_head': unit.id,
                'accuracy': self.state.current_metrics.accuracy,
                'perplexity': self.state.current_metrics.perplexity,
                'compression_ratio': len(self.state.pruned_groups) / len(self.pruning_units)
            }
            
            return self._get_state(), reward, done, info
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            return self._get_state(), -1.0, True, {'error': str(e)}
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Features for each head
        for unit in self.pruning_units:
            state.extend([
                unit.importance_score,  # Importance of head
                1.0 if unit.id in self.state.pruned_groups else 0.0  # Whether head is pruned
            ])
        
        # Global metrics
        accuracy_ratio = (self.state.current_metrics.accuracy / 
                         self.state.initial_metrics.accuracy)
        compression_ratio = len(self.state.pruned_groups) / len(self.pruning_units)
        progress = self.state.current_step / self.config['max_steps']
        
        state.extend([
            accuracy_ratio,
            compression_ratio,
            progress
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid"""
        if action >= len(self.pruning_units):
            return False
            
        # Can't prune already pruned head
        unit = self.pruning_units[action]
        return unit.id not in self.state.pruned_groups
    
    def _prune_head(self, unit: 'PruningUnit'):
        """Prune attention head by zeroing its parameters"""
        with torch.no_grad():
            for param in unit.parameters.values():
                param.zero_()
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on pruning results"""
        # Get metrics
        accuracy_ratio = (self.state.current_metrics.accuracy / 
                         self.state.initial_metrics.accuracy)
        compression_ratio = len(self.state.pruned_groups) / len(self.pruning_units)
        
        # Get recently pruned unit
        last_pruned_id = self.state.pruned_groups[-1]
        last_pruned = next(unit for unit in self.pruning_units if unit.id == last_pruned_id)
        importance_penalty = last_pruned.importance_score
        
        # Calculate reward components
        accuracy_reward = max(0, accuracy_ratio - self.config['min_accuracy_ratio'])
        compression_reward = compression_ratio
        importance_reward = -importance_penalty  # Penalty for pruning important heads
        
        # Combine rewards using weights from config
        reward = (
            self.config['reward_weights']['accuracy'] * accuracy_reward +
            self.config['reward_weights']['compression'] * compression_reward +
            self.config['reward_weights']['importance'] * importance_reward
        )
        
        # Add penalties for constraint violations
        if accuracy_ratio < self.config['min_accuracy_ratio']:
            reward -= 1.0
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if pruning should stop"""
        # Current metrics
        accuracy_ratio = (self.state.current_metrics.accuracy / 
                         self.state.initial_metrics.accuracy)
        compression_ratio = len(self.state.pruned_groups) / len(self.pruning_units)
        
        # Check terminal conditions
        return (
            accuracy_ratio < self.config['min_accuracy_ratio'] or
            compression_ratio >= self.config['target_compression'] or
            self.state.current_step >= self.config['max_steps'] or
            len(self.state.pruned_groups) >= len(self.pruning_units)
        )
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of current pruning state"""
        return {
            'pruned_heads': len(self.state.pruned_groups),
            'total_heads': len(self.pruning_units),
            'compression_ratio': len(self.state.pruned_groups) / len(self.pruning_units),
            'accuracy': self.state.current_metrics.accuracy,
            'accuracy_ratio': (self.state.current_metrics.accuracy / 
                             self.state.initial_metrics.accuracy),
            'perplexity': self.state.current_metrics.perplexity,
            'steps': self.state.current_step
        }