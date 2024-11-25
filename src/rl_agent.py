# src/rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class Transition:
    """Store a single transition in the buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_log_prob: float
    value: float

class Buffer:
    """Experience replay buffer"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.action_log_probs = []
        self.values = []
    
    def add(self, transition: Transition):
        """Add transition to buffer"""
        self.states.append(transition.state)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.next_states.append(transition.next_state)
        self.dones.append(transition.done)
        self.action_log_probs.append(transition.action_log_prob)
        self.values.append(transition.value)
    
    def clear(self):
        """Clear buffer"""
        self.__init__()
    
    def get(self) -> Tuple:
        """Get all data as tensors"""
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(np.array(self.next_states)),
            torch.BoolTensor(self.dones),
            torch.FloatTensor(self.action_log_probs),
            torch.FloatTensor(self.values)
        )

class ActorCritic(nn.Module):
    """Actor-Critic network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Actor network (policy)
        actor_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Critic network (value function)
        critic_layers = []
        prev_dim = state_dim
        for dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        action_probs = torch.softmax(self.actor(state), dim=-1)
        value = self.critic(state)
        return action_probs, value

class PPOAgent:
    """PPO agent for learning pruning policy"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: Dict,
                 device: torch.device):
        """Initialize PPO agent"""
        self.config = config
        self.device = device
        
        # Create actor-critic network
        self.ac = ActorCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config['hidden_dims']
        ).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.ac.parameters(),
            lr=config['learning_rate']
        )
        
        # Initialize buffer
        self.buffer = Buffer()
        
        # Training parameters
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.clip_ratio = config['clip_ratio']
        self.value_coef = config['value_coef']
        self.entropy_coef = config['entropy_coef']
        self.max_grad_norm = config['max_grad_norm']
        
        logger.info(f"Initialized PPO agent with state dim: {state_dim}, action dim: {action_dim}")
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Select action given state"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, value = self.ac(state)
            
            if deterministic:
                action = action_probs.argmax().item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
                action_log_prob = dist.log_prob(torch.tensor([action])).item()
            
            return action, action_log_prob, value.item()
    
    def train_episode(self, env) -> Dict[str, float]:
        """Train for one episode"""
        stats = {
            'episode_reward': 0,
            'episode_length': 0,
            'value_loss': 0,
            'policy_loss': 0,
            'entropy_loss': 0
        }
        
        # Collect experience
        state = env.reset()
        done = False
        
        while not done:
            # Select action
            action, action_log_prob, value = self.select_action(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.buffer.add(Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                action_log_prob=action_log_prob,
                value=value
            ))
            
            # Update statistics
            stats['episode_reward'] += reward
            stats['episode_length'] += 1
            
            state = next_state
            
            # Update policy if enough steps
            if len(self.buffer.states) >= self.config['batch_size']:
                loss_stats = self._update_policy()
                stats.update(loss_stats)
        
        return stats
    
    def _update_policy(self) -> Dict[str, float]:
        """Update policy using PPO"""
        # Get data from buffer
        states, actions, rewards, next_states, dones, old_log_probs, values = self.buffer.get()
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        values = values.to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            next_values = self.ac(next_states)[1].squeeze(-1)
            advantages = self._compute_gae(rewards, values, next_values, dones)
            returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config['n_epochs']):
            # Get action probabilities and values
            action_probs, current_values = self.ac(states)
            dist = torch.distributions.Categorical(action_probs)
            current_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio and surrogate loss
            ratios = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = 0.5 * (returns - current_values.squeeze(-1)).pow(2).mean()
            
            # Compute total loss
            loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': -entropy.item()
        }
    
    def _compute_gae(self,
                    rewards: torch.Tensor,
                    values: torch.Tensor,
                    next_values: torch.Tensor,
                    dones: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * gae
            advantages[t] = gae
        
        return advantages
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.ac.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.ac.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])