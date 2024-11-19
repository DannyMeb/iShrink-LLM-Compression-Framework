import os
import torch
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from src.model_loader import ModelLoader
from src.dependency_graph import DependencyGraphBuilder
from src.importance_scorer import ImportanceScorer
from src.pruning_env import PruningEnvironment
from src.rl_agent import EnhancedPPOAgent
from src.metrics import MetricsTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PruningPipeline:
    """Main pipeline for model pruning using RL"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        self.save_dir = Path(self.config['system']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_seed(self.config['system']['seed'])
        
        logger.info(f"Initialized pruning pipeline with device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration parameters"""
        required_sections = ['model', 'pruning', 'rl', 'training', 'evaluation', 'system']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
    
    def _setup_device(self) -> str:
        """Setup computation device"""
        if torch.cuda.is_available() and self.config['model']['device'] == 'cuda':
            return 'cuda'
        return 'cpu'
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    def run(self):
        """Execute the complete pruning pipeline"""
        try:
            # 1. Load Model
            logger.info("Loading model...")
            model, tokenizer = self._load_model()
            
            # 2. Build Dependency Graph
            logger.info("Building dependency graph...")
            groups, graph = self._build_dependency_graph(model)
            
            # 3. Setup Importance Scorer
            logger.info("Setting up importance scorer...")
            importance_scorer = self._setup_importance_scorer(model, tokenizer)
            
            # 4. Calculate importance scores
            logger.info("Calculating importance scores...")
            self._calculate_importance_scores(groups, importance_scorer)
            
            # 5. Setup Environment
            logger.info("Setting up pruning environment...")
            env = self._setup_environment(model, groups)
            
            # 6. Setup RL Agent
            logger.info("Setting up RL agent...")
            agent = self._setup_agent(env)
            
            # 7. Setup Metrics Tracker
            logger.info("Setting up metrics tracker...")
            metrics_tracker = MetricsTracker(
                save_dir=self.save_dir / 'metrics',
                original_model=model,
                device=self.device
            )
            
            # 8. Training Loop
            logger.info("Starting training loop...")
            self._training_loop(env, agent, metrics_tracker)
            
            # 9. Final Evaluation
            logger.info("Performing final evaluation...")
            self._final_evaluation(env, metrics_tracker)
            
            # 10. Save Results
            logger.info("Saving results...")
            self._save_results(metrics_tracker)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
    
    def _load_model(self):
        """Load the model and tokenizer"""
        model_config = self.config['model']
        loader = ModelLoader(
            model_name=model_config['name'],
            local_path=model_config['local_path'],
            device=self.device,
            precision=model_config['precision']
        )
        return loader.load()
    
    def _build_dependency_graph(self, model):
        """Build dependency graph from model"""
        graph_builder = DependencyGraphBuilder(model)
        return graph_builder.build()
    
    def _setup_importance_scorer(self, model, tokenizer):
        """Setup importance scorer with calibration data"""
        importance_config = self.config['pruning']['importance']
        return ImportanceScorer(
            model=model,
            calibration_data=self._get_calibration_data(tokenizer),
            num_samples=importance_config['num_samples'],
            batch_size=importance_config['batch_size'],
            device=self.device
        )
    
    def _calculate_importance_scores(self, groups, scorer):
        """Calculate importance scores for all groups"""
        for group in groups:
            score = scorer.compute_group_importance(group)
            group.importance_score = score.combined_score
    
    def _setup_environment(self, model, groups):
        """Setup pruning environment"""
        env_config = self.config['rl']['env']
        return PruningEnvironment(
            model=model,
            groups=groups,
            evaluation_data=self._get_evaluation_data(),
            target_metrics=self._get_target_metrics(),
            min_accuracy=self.config['pruning']['targets']['min_accuracy'],
            max_steps=env_config['max_steps'],
            device=self.device
        )
    
    def _setup_agent(self, env):
        """Setup RL agent"""
        ppo_config = self.config['rl']['ppo']
        return EnhancedPPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=ppo_config['learning_rate'],
            gamma=ppo_config['gamma'],
            gae_lambda=ppo_config['gae_lambda'],
            clip_ratio=ppo_config['clip_ratio'],
            batch_size=ppo_config['batch_size'],
            n_epochs=ppo_config['n_epochs'],
            device=self.device
        )
    
    def _training_loop(self, env, agent, metrics_tracker):
        """Execute training loop"""
        train_config = self.config['training']
        best_reward = float('-inf')
        patience_counter = 0
        
        for episode in range(train_config['optimization']['num_episodes']):
            state = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action
                action, value, log_prob = agent.select_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                agent.memory.store(state, action, reward, value, log_prob, done, info)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
                
                # Update agent if enough steps
                if len(agent.memory.states) >= agent.memory.batch_size:
                    agent.update()
            
            # Checkpoint if improved
            if episode_reward > best_reward:
                best_reward = episode_reward
                self._save_checkpoint(agent, env, episode)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= train_config['optimization']['early_stopping']['patience']:
                logger.info("Early stopping triggered")
                break
            
            # Log progress
            if episode % train_config['logging']['log_freq'] == 0:
                self._log_progress(episode, episode_reward, metrics_tracker)
    
    def _final_evaluation(self, env, metrics_tracker):
        """Perform final evaluation"""
        eval_config = self.config['evaluation']
        
        for _ in range(eval_config['num_runs']):
            state = env.reset()
            done = False
            
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, _, done, _ = env.step(action)
            
            metrics_tracker.measure_pruning_metrics(
                model=env.model,
                eval_dataloader=self._get_eval_dataloader(),
                pruned_groups=env.pruned_groups,
                step=env.state.current_step
            )
    
    def _save_results(self, metrics_tracker):
        """Save final results and plots"""
        metrics_tracker.plot_metrics()
        metrics_tracker.export_results('final_results.json')
        
        logger.info("Final Results:")
        for key, value in metrics_tracker.get_summary().items():
            logger.info(f"{key}: {value}")
    
    def _save_checkpoint(self, agent, env, episode):
        """Save training checkpoint"""
        checkpoint = {
            'agent_state': agent.state_dict(),
            'env_state': env.state_dict(),
            'episode': episode,
            'config': self.config
        }
        torch.save(checkpoint, self.save_dir / 'checkpoints' / f'checkpoint_{episode}.pt')
    
    def _log_progress(self, episode, reward, metrics_tracker):
        """Log training progress"""
        metrics = metrics_tracker.get_summary()
        logger.info(f"Episode {episode}")
        logger.info(f"Reward: {reward:.4f}")
        logger.info(f"Compression Ratio: {metrics['compression_ratio']:.4f}")
        logger.info(f"Accuracy: {metrics['final_metrics']['accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Run pruning pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    pipeline = PruningPipeline(args.config)
    pipeline.run()

if __name__ == '__main__':
    main()