# rl_agent.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        logger.info(f"Initializing ActorCritic with state_dim={state_dim}, action_dim={action_dim}")
        
        # Use smaller initial weights to prevent exploding gradients
        self.weight_gain = 0.1
        
        # Input preprocessing
        self.input_norm = nn.LayerNorm(state_dim)
        
        # Shared network
        shared_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),  # Changed from ReLU to Tanh for better stability
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*shared_layers)
        
        # Policy head (actor)
        self.policy = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 2, action_dim),
            nn.Sigmoid()
        )
        
        # Value head (critic)
        self.value = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.LayerNorm(hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        logger.info("ActorCritic network initialized")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=self.weight_gain)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Ensure proper dimensions and dtype
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.float()
            
            # Add small noise to prevent identical inputs
            if self.training:
                state = state + torch.randn_like(state) * 1e-4
            
            # Forward pass with gradient scaling
            with torch.set_grad_enabled(self.training):
                # Normalize input
                x = self.input_norm(state)
                
                # Shared features
                features = self.shared(x)
                
                # Get action probabilities and value
                action_probs = self.policy(features)
                value = self.value(features)
                
                # Ensure valid probabilities
                action_probs = torch.clamp(action_probs, 1e-6, 1-1e-6)
                
                # Check for NaN values
                if torch.isnan(action_probs).any() or torch.isnan(value).any():
                    logger.error(f"NaN detected - action_probs: {torch.isnan(action_probs).sum()}, value: {torch.isnan(value).sum()}")
                    logger.error(f"State range: [{state.min():.4f}, {state.max():.4f}]")
                    logger.error(f"Features range: [{features.min():.4f}, {features.max():.4f}]")
                
                return action_probs, value
                
        except Exception as e:
            logger.error(f"Error in ActorCritic forward pass: {str(e)}")
            logger.error(f"Input state shape: {state.shape}")
            raise

class PPOAgent:
    """PPO agent for pruning decisions"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        logger.info(f"Initializing PPO agent with state_dim={state_dim}, action_dim={action_dim}")
        
        try:
            # Set default dtype to float32
            torch.set_default_dtype(torch.float32)
            
            # Initialize actor-critic
            self.ac = ActorCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config['hidden_dims']
            ).to(device).float()
            
            # Initialize optimizer
            self.optimizer = torch.optim.Adam(
                self.ac.parameters(),
                lr=config['learning_rate']
            )
            
            # Training parameters
            self.clip_ratio = config['clip_ratio']
            self.value_coef = config['value_coef']
            self.entropy_coef = config['entropy_coef']
            self.max_grad_norm = config['max_grad_norm']
            
            # Initialize memory buffers
            self.reset_memory()
            
        except Exception as e:
            logger.error(f"Error initializing PPO agent: {str(e)}")
            raise
    
    def reset_memory(self):
        """Reset experience memory"""
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        logger.debug("Reset agent memory buffers")
    
    def train_episode(self, env) -> Dict[str, float]:
        """Train for one episode"""
        try:
            state, _ = env.reset()
            logger.debug(f"Initial state keys: {state.keys()}")
            logger.debug(f"State shapes - global: {state['global_features'].shape}, "
                        f"unit: {state['unit_features'].shape}, "
                        f"layer: {state['layer_features'].shape}")
            
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                # Select and perform action
                action, action_log_prob, value = self.select_action(state)
                
                # Convert action to numpy and execute
                action_np = action.cpu().numpy()
                next_state, reward, terminated, truncated, info = env.step(action_np)
                
                done = terminated or truncated
                
                # Store transition
                self.rewards.append(reward)
                self.masks.append(1 - done)
                episode_reward += reward
                episode_length += 1
                
                # Update state
                state = next_state
                
                # Log episode progress
                if episode_length % 10 == 0:
                    logger.debug(f"Episode step {episode_length}, Reward: {reward:.4f}, "
                              f"Total reward: {episode_reward:.4f}")
                
                # Update policy if enough steps
                if episode_length % self.config['update_interval'] == 0:
                    logger.debug("Performing policy update")
                    losses = self.update()
            
            # Final update
            if len(self.states) > 0:
                losses = self.update()
            else:
                losses = {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
            
            stats = {
                'episode_reward': float(episode_reward),
                'episode_length': episode_length,
                **losses
            }
            
            logger.info(f"Episode complete - Length: {episode_length}, "
                      f"Total reward: {episode_reward:.4f}")
            return stats
            
        except Exception as e:
            logger.error(f"Error in training episode: {str(e)}")
            logger.error(f"Current state keys: {state.keys() if isinstance(state, dict) else 'Not a dict'}")
            raise
    
    def select_action(self, state: Dict[str, np.ndarray], training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Select pruning action using current policy"""
        try:
            # Debug prints for state components
            logger.debug("Processing state components:")
            logger.debug(f"global_features shape: {state['global_features'].shape}")
            logger.debug(f"unit_features shape: {state['unit_features'].shape}")
            logger.debug(f"layer_features shape: {state['layer_features'].shape}")
            
            # Convert state components to float32
            global_features = torch.FloatTensor(state['global_features'].astype(np.float32))
            unit_features = torch.FloatTensor(state['unit_features'].astype(np.float32))
            layer_features = torch.FloatTensor(state['layer_features'].astype(np.float32))
            
            # Flatten and concatenate
            state_tensor = torch.cat([
                global_features,
                unit_features.view(-1),
                layer_features.view(-1)
            ]).to(self.device)
            
            # Add batch dimension if needed
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            logger.debug(f"Concatenated state tensor shape: {state_tensor.shape}")
            logger.debug(f"State tensor range: [{state_tensor.min():.4f}, {state_tensor.max():.4f}]")
            
            with torch.no_grad():
                # Get action probabilities and value
                action_probs, value = self.ac(state_tensor)
                
                # Ensure proper shape for action_probs
                action_probs = action_probs.view(1, -1)  # [1, num_actions]
                
                logger.debug(f"Action probs range: [{action_probs.min():.6f}, {action_probs.max():.6f}]")
                
                # Create distribution
                dist = Bernoulli(action_probs)
                
                if training:
                    # Sample action in training
                    action = dist.sample()
                    action_log_prob = dist.log_prob(action)
                    
                    # Store experience
                    self.states.append(state_tensor)
                    self.actions.append(action)
                    self.action_log_probs.append(action_log_prob)
                    self.values.append(value)
                    
                    logger.debug(f"Selected action shape: {action.shape}")
                    return action, action_log_prob, value
                else:
                    # Deterministic actions in evaluation
                    action = (action_probs > 0.5).float()
                    return action, None, None
                    
        except Exception as e:
            logger.error(f"Error in action selection: {str(e)}")
            logger.error(f"State tensor shape: {state_tensor.shape if 'state_tensor' in locals() else 'Not created'}")
            raise
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO"""
        try:
            if len(self.states) == 0:
                logger.warning("No experiences to update from")
                return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
            
            # Stack experiences and ensure float32
            states = torch.cat(self.states).float()
            actions = torch.cat(self.actions).float()
            old_action_log_probs = torch.cat(self.action_log_probs).float()
            
            # Calculate returns and advantages
            returns, advantages = self._compute_gae(
                torch.tensor(self.rewards, device=self.device, dtype=torch.float32),
                torch.cat(self.values).float(),
                torch.tensor(self.masks, device=self.device, dtype=torch.float32)
            )
            
            # Expand advantages to match action dimensions
            advantages = advantages.unsqueeze(-1).expand(-1, actions.size(-1))
            
            logger.debug(f"Update batch size: {states.shape[0]}")
            
            # PPO update loop
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy_loss = 0
            
            for _ in range(self.config['n_epochs']):
                # Get current action probabilities and values
                action_probs, values = self.ac(states)
                dist = Bernoulli(action_probs)
                
                # Calculate probability ratio
                action_log_probs = dist.log_prob(actions)
                ratios = torch.exp(action_log_probs - old_action_log_probs)
                
                # Calculate surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with proper dimensions
                value_loss = F.mse_loss(values, returns.view(-1, 1))
                
                # Entropy loss for exploration
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
            
            # Average losses
            n_epochs = self.config['n_epochs']
            losses = {
                'policy_loss': total_policy_loss / n_epochs,
                'value_loss': total_value_loss / n_epochs,
                'entropy_loss': total_entropy_loss / n_epochs
            }
            
            logger.debug(f"Update losses: {losses}")
            
            # Reset memory
            self.reset_memory()
            
            return losses
            
        except Exception as e:
            logger.error(f"Error in policy update: {str(e)}")
            logger.error(f"Shapes - States: {states.shape if 'states' in locals() else 'Not created'}, "
                        f"Actions: {actions.shape if 'actions' in locals() else 'Not created'}, "
                        f"Advantages: {advantages.shape if 'advantages' in locals() else 'Not created'}")
            raise
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute returns and advantages using GAE"""
        try:
            # Ensure all inputs are float32
            rewards = rewards.float()
            values = values.float()
            masks = masks.float()
            
            # Get GAE parameters
            gamma = self.config['gamma']
            gae_lambda = self.config['gae_lambda']
            
            # Calculate advantages
            advantages = torch.zeros_like(rewards, dtype=torch.float32)
            last_advantage = 0
            last_value = values[-1]
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = last_value
                else:
                    next_value = values[t + 1]
                
                delta = rewards[t] + gamma * next_value * masks[t] - values[t]
                advantages[t] = delta + gamma * gae_lambda * masks[t] * last_advantage
                last_advantage = advantages[t]
            
            # Calculate returns
            returns = advantages + values
            
            # Normalize advantages if more than one sample
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            return returns, advantages
            
        except Exception as e:
            logger.error(f"Error computing GAE: {str(e)}")
            raise

    def save(self, path: str):
        """Save model"""
        try:
            torch.save({
                'model_state_dict': self.ac.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config
            }, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load model"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.ac.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise