import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import wandb
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PPOStats:
    """Training statistics for PPO"""
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropy_losses: List[float] = field(default_factory=list)
    total_losses: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    ratios: List[float] = field(default_factory=list)
    
    def clear(self):
        """Clear all statistics"""
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, [])
    
    def get_means(self) -> Dict[str, float]:
        """Get mean values of all statistics"""
        return {
            field_name: np.mean(getattr(self, field_name))
            for field_name in self.__dataclass_fields__
            if len(getattr(self, field_name)) > 0
        }

class ActorCritic(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 dropout: float = 0.1):
        super().__init__()
        
        # Build shared layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head for policy
        self.actor = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head for value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

class PPOMemory:
    def __init__(self, batch_size: int, device: str = 'cuda'):
        """Initialize replay memory"""
        self.batch_size = batch_size
        self.device = device
        self.clear()
    
    def store(self, 
              state: np.ndarray,
              action: int,
              reward: float,
              value: float,
              log_prob: float,
              done: bool):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def clear(self):
        """Clear memory"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def get_batches(self) -> List[Dict[str, torch.Tensor]]:
        """Get all batches"""
        batch_size = min(self.batch_size, len(self.states))
        indices = np.random.permutation(len(self.states))
        
        batches = []
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batches.append(self._get_batch(batch_indices))
        
        return batches
    
    def _get_batch(self, indices: np.ndarray) -> Dict[str, torch.Tensor]:
        """Get single batch"""
        return {
            'states': torch.FloatTensor(np.array(self.states)[indices]).to(self.device),
            'actions': torch.LongTensor(np.array(self.actions)[indices]).to(self.device),
            'rewards': torch.FloatTensor(np.array(self.rewards)[indices]).to(self.device),
            'values': torch.FloatTensor(np.array(self.values)[indices]).to(self.device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)[indices]).to(self.device),
            'dones': torch.FloatTensor(np.array(self.dones)[indices]).to(self.device)
        }

class PPOAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """Initialize PPO agent"""
        self.config = config
        self.device = device
        
        # Initialize networks
        self.actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['hidden_dims']
        ).to(device)
        
        # Initialize optimizer with learning rate schedule
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=config['learning_rate']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Initialize memory and statistics
        self.memory = PPOMemory(config['batch_size'], device)
        self.stats = PPOStats()
        
        # Training tracking
        self.current_episode = 0
        self.best_reward = float('-inf')
        
        logger.info(f"Initialized PPO agent with state dim: {state_dim}, "
                   f"action dim: {action_dim}")
    
    def select_action(self, 
                     state: np.ndarray, 
                     deterministic: bool = False) -> Tuple[int, float, float]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            
            if deterministic:
                action = torch.argmax(action_probs)
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
            
            log_prob = torch.log(action_probs.squeeze(0)[action])
        
        return action.item(), value.item(), log_prob.item()
    
    def train_episode(self, env) -> Dict[str, float]:
        """Train for one episode"""
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_metrics = defaultdict(list)
        
        while not done:
            # Select action
            action, value, log_prob = self.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.memory.store(state, action, reward, value, log_prob, done)
            
            # Update statistics
            episode_reward += reward
            episode_steps += 1
            episode_metrics['values'].append(value)
            episode_metrics['rewards'].append(reward)
            
            # Update state
            state = next_state
            
            # Update policy if enough steps
            if len(self.memory.states) >= self.memory.batch_size:
                update_metrics = self.update()
                for k, v in update_metrics.items():
                    episode_metrics[k].append(v)
        
        # Update learning rate
        self.scheduler.step(episode_reward)
        
        # Track best reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
        
        self.current_episode += 1
        
        # Compute episode statistics
        statistics = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'average_value': np.mean(episode_metrics['values']),
            'average_reward': np.mean(episode_metrics['rewards']),
            'best_reward': self.best_reward
        }
        
        # Add policy update statistics if any
        if 'policy_loss' in episode_metrics:
            statistics.update({
                'average_policy_loss': np.mean(episode_metrics['policy_loss']),
                'average_value_loss': np.mean(episode_metrics['value_loss']),
                'average_total_loss': np.mean(episode_metrics['total_loss'])
            })
        
        return statistics
    
    def update(self) -> Dict[str, float]:
        """Update policy and value networks"""
        # Get all batches
        batches = self.memory.get_batches()
        
        # Compute advantages and returns for all data
        advantages = self._compute_advantages(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones
        )
        returns = advantages + torch.FloatTensor(self.memory.values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for n epochs
        for _ in range(self.config['n_epochs']):
            for batch in batches:
                # Get current predictions
                action_probs, values = self.actor_critic(batch['states'])
                dist = torch.distributions.Categorical(action_probs)
                
                # Get current log probs and entropy
                curr_log_probs = dist.log_prob(batch['actions'])
                entropy = dist.entropy().mean()
                
                # Compute ratios and policy loss
                ratios = (curr_log_probs - batch['log_probs']).exp()
                surr1 = ratios * advantages
                surr2 = torch.clamp(
                    ratios,
                    1 - self.config['clip_ratio'],
                    1 + self.config['clip_ratio']
                ) * advantages
                
                # Compute losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                entropy_loss = -self.config['entropy_coef'] * entropy
                
                # Compute total loss
                total_loss = (policy_loss + 
                            self.config['value_coef'] * value_loss +
                            entropy_loss)
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                
                # Store statistics
                self.stats.policy_losses.append(policy_loss.item())
                self.stats.value_losses.append(value_loss.item())
                self.stats.entropy_losses.append(entropy_loss.item())
                self.stats.total_losses.append(total_loss.item())
                self.stats.ratios.extend(ratios.detach().cpu().numpy())
                self.stats.advantages.extend(advantages.detach().cpu().numpy())
        
        # Clear memory
        self.memory.clear()
        
        # Return statistics
        return self.stats.get_means()
    
    def _compute_advantages(self,
                          rewards: List[float],
                          values: List[float],
                          dones: List[bool]) -> torch.Tensor:
        """Compute advantages using GAE"""
        advantages = torch.zeros_like(torch.FloatTensor(rewards)).to(self.device)
        last_gae = 0
        
        for t in reversed(range(len(rewards) - 1)):
            non_terminal = 1 - dones[t]
            delta = (rewards[t] + 
                    self.config['gamma'] * values[t + 1] * non_terminal - 
                    values[t])
            last_gae = (delta + 
                       self.config['gamma'] * 
                       self.config['gae_lambda'] * 
                       non_terminal * 
                       last_gae)
            advantages[t] = last_gae
        
        return advantages
    
    def save(self, path: str):
        """Save agent state"""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'actor_critic_state': self.actor_critic.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'stats': self.stats,
            'config': self.config,
            'best_reward': self.best_reward,
            'current_episode': self.current_episode
        }, save_path)
        
        logger.info(f"Saved agent state to {save_path}")
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.stats = checkpoint['stats']
        self.config = checkpoint['config']
        self.best_reward = checkpoint['best_reward']
        self.current_episode = checkpoint['current_episode']
        
        logger.info(f"Loaded agent state from {path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress"""
        return {
            'current_episode': self.current_episode,
            'best_reward': self.best_reward,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'policy_loss': np.mean(self.stats.policy_losses[-100:]) 
                          if self.stats.policy_losses else None,
            'value_loss': np.mean(self.stats.value_losses[-100:])
                         if self.stats.value_losses else None,
            'average_ratio': np.mean(self.stats.ratios[-100:])
                           if self.stats.ratios else None
        }