# trainer.py

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class RLTrainer:
    """Handles training of the RL agent for model pruning"""
    
    def __init__(self, 
                 agent,
                 env,
                 config: Dict[str, Any],
                 save_dir: Path):
        """
        Initialize trainer
        
        Args:
            agent: PPO agent instance
            env: Pruning environment instance
            config: Training configuration
            save_dir: Directory for saving results
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.save_dir = save_dir
        
        # Create checkpoint directory
        self.checkpoint_dir = save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.num_episodes = config['training']['optimization']['num_episodes']
        self.patience = config['training']['optimization']['early_stopping']['patience']
        self.min_delta = config['training']['optimization']['early_stopping']['min_delta']
        
        # Logging parameters
        self.use_wandb = config['training']['logging']['use_wandb']
        self.log_freq = config['training']['logging']['log_freq']
        
    def train(self) -> Dict[str, Any]:
        """
        Run complete training process
        
        Returns:
            Dict containing training results and best model info
        """
        try:
            logger.info("Starting RL training...")
            best_reward = float('-inf')
            patience_counter = 0
            best_checkpoint = None
            training_history = []
            
            for episode in tqdm(range(self.num_episodes), desc="Training"):
                # Run training episode
                stats = self.agent.train_episode(self.env)
                training_history.append(stats)
                
                # Log progress
                if (episode + 1) % self.log_freq == 0:
                    self._log_progress(episode, stats)
                
                # Check for improvement
                if stats['episode_reward'] > best_reward + self.min_delta:
                    best_reward = stats['episode_reward']
                    best_checkpoint = self._save_checkpoint(episode, stats)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {episode + 1} episodes")
                    break
            
            # Save training history
            self._save_training_history(training_history)
            
            return {
                'best_checkpoint': best_checkpoint,
                'best_reward': best_reward,
                'episodes_trained': episode + 1,
                'early_stopped': patience_counter >= self.patience
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _log_progress(self, episode: int, stats: Dict[str, float]):
        """Log training progress"""
        log_msg = f"Episode {episode + 1}: "
        log_msg += f"Reward: {stats['episode_reward']:.4f}, "
        log_msg += f"Length: {stats['episode_length']}, "
        log_msg += f"Policy Loss: {stats['policy_loss']:.4f}, "
        log_msg += f"Value Loss: {stats['value_loss']:.4f}"
        
        logger.info(log_msg)
        
        if self.use_wandb:
            wandb.log({
                'episode': episode + 1,
                **stats
            })
    
    def _save_checkpoint(self, episode: int, stats: Dict[str, float]) -> str:
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'agent_state_dict': self.agent.ac.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'stats': stats,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def _save_training_history(self, history: list):
        """Save complete training history"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)
        self.agent.ac.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['stats']
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate current agent
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dict containing evaluation metrics
        """
        self.agent.ac.eval()
        total_reward = 0
        total_length = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                with torch.no_grad():
                    action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
            
            total_reward += episode_reward
            total_length += self.env.state.current_step
        
        self.agent.ac.train()
        
        return {
            'mean_reward': total_reward / num_episodes,
            'mean_length': total_length / num_episodes
        }