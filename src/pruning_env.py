import gym
from gym import spaces
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from .utils import setup_device, create_dataloader, calculate_model_size
from .dependency_graph import PruningGroup
from .metrics import MetricsTracker

logger = logging.getLogger(__name__)

@dataclass
class PruningState:
    """Track the state of the pruning process"""
    current_step: int = 0
    total_pruned: int = 0
    pruned_groups: List[str] = field(default_factory=list)
    original_size: Optional[float] = None
    current_size: Optional[float] = None
    accuracy: Optional[float] = None
    latency: Optional[float] = None
    
    def compression_ratio(self) -> float:
        """Calculate current compression ratio"""
        if self.original_size and self.current_size:
            return self.current_size / self.original_size
        return 1.0

class PruningEnvironment(gym.Env):
    def __init__(self,
                 model: torch.nn.Module,
                 groups: List[PruningGroup],
                 eval_dataloader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 metrics_tracker: MetricsTracker,
                 device: Optional[str] = None):
        """
        Initialize pruning environment
        """
        super().__init__()
        self.model = model
        self.groups = groups
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.metrics_tracker = metrics_tracker
        self.device = setup_device(device)
        
        # Initialize state
        self.state = PruningState()
        self._initialize_state()
        
        # Setup spaces
        self._setup_spaces()
        
        # Save original model state
        self.original_state = self.model.state_dict()
        
        logger.info(f"Initialized PruningEnvironment with {len(groups)} groups")
    
    def _initialize_state(self):
        """Initialize environment state"""
        num_params, memory_size = calculate_model_size(self.model)
        self.state.original_size = memory_size
        self.state.current_size = memory_size
        
        # Initial evaluation
        metrics = self.metrics_tracker.measure_pruning_metrics(
            self.model,
            self.eval_dataloader,
            [],
            0
        )
        self.state.accuracy = metrics.accuracy.accuracy
        self.state.latency = metrics.latency.avg_latency
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Action space: which group to prune
        self.action_space = spaces.Discrete(len(self.groups))
        
        # Observation space
        group_features = 6  # importance, pruned, deps count, size, deps_pruned_ratio, neighbor_importance
        global_features = 5  # accuracy, latency, memory ratios, steps_remaining, total_pruned
        state_dim = len(self.groups) * group_features + global_features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Restore model
        self.model.load_state_dict(self.original_state)
        
        # Reset state
        self.state = PruningState()
        self._initialize_state()
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute pruning action"""
        # Validate action
        if not self._is_valid_action(action):
            return self._get_state(), -1.0, True, {'invalid_action': True}
        
        # Get group to prune
        group = self.groups[action]
        
        try:
            # Prune group
            self._prune_group(group)
            self.state.pruned_groups.append(group.id)
            self.state.total_pruned += 1
            self.state.current_step += 1
            
            # Measure metrics
            metrics = self.metrics_tracker.measure_pruning_metrics(
                self.model,
                self.eval_dataloader,
                self.state.pruned_groups,
                self.state.current_step
            )
            
            # Update state
            self.state.accuracy = metrics.accuracy.accuracy
            self.state.latency = metrics.latency.avg_latency
            self.state.current_size = metrics.model_size.total_memory
            
            # Calculate reward
            reward = self._calculate_reward(metrics)
            
            # Check if done
            done = self._is_done(metrics)
            
            # Prepare info dict
            info = {
                'metrics': metrics,
                'pruned_group': group.id,
                'compression_ratio': self.state.compression_ratio(),
                'accuracy_drop': self.state.accuracy - metrics.accuracy.accuracy
            }
            
            return self._get_state(), reward, done, info
            
        except Exception as e:
            logger.error(f"Error during step: {str(e)}")
            return self._get_state(), -1.0, True, {'error': str(e)}
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []
        
        # Group features
        for group in self.groups:
            pruned = 1.0 if group.id in self.state.pruned_groups else 0.0
            deps_pruned = sum(1 for dep in group.dependencies 
                            if dep in self.state.pruned_groups)
            deps_pruned_ratio = deps_pruned / len(group.dependencies) if group.dependencies else 1.0
            
            neighbor_importance = np.mean([
                g.importance_score for g in self.groups
                if g.id in group.dependencies or group.id in g.dependencies
            ]) if group.dependencies else 0.0
            
            state.extend([
                group.importance_score,
                pruned,
                len(group.dependencies),
                group.memory_size,
                deps_pruned_ratio,
                neighbor_importance
            ])
        
        # Global features
        steps_remaining = (self.config['max_steps'] - self.state.current_step) / self.config['max_steps']
        
        state.extend([
            self.state.accuracy,
            self.state.latency,
            self.state.compression_ratio(),
            steps_remaining,
            self.state.total_pruned / len(self.groups)
        ])
        
        return np.array(state, dtype=np.float32)
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid"""
        if action >= len(self.groups):
            return False
            
        group = self.groups[action]
        if group.id in self.state.pruned_groups:
            return False
            
        # Check dependencies
        return all(dep not in group.dependencies or dep in self.state.pruned_groups 
                  for dep in group.dependencies)
    
    def _prune_group(self, group: PruningGroup):
        """Prune parameter group"""
        with torch.no_grad():
            for param in group.parameters.values():
                param.zero_()
    
    def _calculate_reward(self, metrics) -> float:
        """Calculate reward based on metrics"""
        weights = self.config['reward_weights']
        
        # Calculate components
        accuracy_reward = max(0, self.state.accuracy - self.config['min_accuracy'])
        memory_reward = max(0, 1 - self.state.compression_ratio())
        latency_reward = max(0, 1 - (metrics.latency.avg_latency / metrics.latency.p90_latency))
        
        # Combine rewards
        reward = (weights['accuracy'] * accuracy_reward +
                 weights['memory'] * memory_reward +
                 weights['latency'] * latency_reward)
        
        # Apply penalties
        if self.state.accuracy < self.config['min_accuracy']:
            reward -= 1.0
        
        return reward
    
    def _is_done(self, metrics) -> bool:
        """Check if pruning should stop"""
        return (self.state.accuracy < self.config['min_accuracy'] or
                self.state.compression_ratio() <= self.config['target_memory'] or
                self.state.current_step >= self.config['max_steps'])
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """Get summary of pruning process"""
        return {
            'pruned_groups': self.state.pruned_groups,
            'compression_ratio': self.state.compression_ratio(),
            'final_accuracy': self.state.accuracy,
            'final_latency': self.state.latency,
            'total_steps': self.state.current_step,
            'total_pruned': self.state.total_pruned
        }