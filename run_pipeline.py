#run_pipeline.py

import os
import json
import torch
import yaml
import logging
import argparse
from pathlib import Path

from typing import Dict, Any, Tuple, List, Optional
import wandb
from tqdm import tqdm

from src.model_loader import ModelLoader
from src.dependency_graph import DependencyGraphBuilder, PruningUnit
from src.importance_scorer import ImportanceScorer
from src.pruning_env import PruningEnvironment
from src.rl_agent import PPOAgent
from src.metrics import MetricsTracker
from src.data import create_mmlu_dataloader

# Setup logging
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
        self.device = torch.device(self.config['model']['device'])
        self.save_dir = Path(self.config['system']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if self.config['training']['logging']['use_wandb']:
            wandb.init(
                project=self.config['training']['logging']['project_name'],
                config=self.config
            )
        
        # Set random seeds
        torch.manual_seed(self.config['system']['seed'])
        torch.cuda.manual_seed_all(self.config['system']['seed'])
        
        logger.info(f"Initialized pruning pipeline with device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration parameters"""
        required_sections = ['model', 'pruning', 'rl', 'training', 'system']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
    
    def run(self):
        """Execute the complete pruning pipeline"""
        try:
            # 1. Load Model and Data
            logger.info("Loading model and data...")
            model, tokenizer,eval_dataloader = self._setup_model_and_data()
            
            # 2. Initial Model Evaluation
            metrics_tracker = MetricsTracker(
                save_dir=self.save_dir, device=self.device,
                tokenizer=tokenizer, config=self.config,
                use_wandb=self.config['training']['logging']['use_wandb'])
            
            initial_metrics_path = self.save_dir / 'metrics' / 'initial_metrics.json'
            if initial_metrics_path.exists():
                logger.info("Loading existing initial metrics...")
                initial_metrics = metrics_tracker.load_metrics('initial_metrics.json')
            else:
                logger.info("Evaluating initial model...")
                initial_metrics = metrics_tracker.evaluate_model(model, tokenizer)
                metrics_tracker.save_metrics(initial_metrics, 'initial_metrics.json')
                
            logger.info(f"Initial model accuracy: {initial_metrics.accuracy:.4f}")

            # 3. Build Pruning Units
            logger.info("Creating pruning units...")
            pruning_units = self._create_pruning_units(model)
            
            # 4. Calculate Importance Scores
            logger.info("Calculating importance scores...")
            pruning_units = self._handle_importance_scores(model, tokenizer, eval_dataloader, pruning_units)
            
            # # 5. Setup Environment
            logger.info("Setting up pruning environment...")
            env = PruningEnvironment(
                model=model,
                pruning_units=pruning_units,
                eval_dataloader=eval_dataloader,
                metrics_tracker=metrics_tracker,
                config=self.config,  # Pass the entire config
                device=self.device,
                initial_metrics=initial_metrics
            )
            
            # # 6. Setup RL Agent
            # logger.info("Setting up RL agent...")
            # agent = PPOAgent(
            #     state_dim=env.observation_space.shape[0],
            #     action_dim=env.action_space.n,
            #     config=self.config['rl']['ppo'],
            #     device=self.device
            # )
            
            # # 7. Train Agent
            # logger.info("Starting RL training...")
            # self._train_agent(env, agent)
            
            # # 8. Final Evaluation and Saving
            # logger.info("Performing final evaluation...")
            # self._save_final_results(model, env
            
            logger.info("Pruning pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _setup_model_and_data(self) -> Tuple[torch.nn.Module, Any, torch.utils.data.DataLoader]:
        """Setup model and data"""
        # Load model
        model_loader = ModelLoader(
            config=self.config['model']
        )
        model, tokenizer = model_loader.load()
        
        # Create evaluation dataloader
        eval_dataloader, _ = create_mmlu_dataloader(
            tokenizer=tokenizer,
            config=self.config,
            split="validation"
        )
        
        return model, tokenizer, eval_dataloader
    
    def _create_pruning_units(self, model) -> List[PruningUnit]:
        """Create pruning units for attention heads"""
        graph_builder = DependencyGraphBuilder(
            model=model,
            config=self.config['pruning']['dependency']
        )
        pruning_units, _ = graph_builder.build()
        return pruning_units
    
    def _handle_importance_scores(self, model, tokenizer, eval_dataloader, pruning_units):
        """Calculate or load importance scores"""
        scores_path = self.save_dir / 'importance_scores' / 'importance_scores.json'
        
        if scores_path.exists():
            logger.info(f"Loading existing importance scores from {scores_path}")
            with open(scores_path) as f:
                scores_data = json.load(f)
                for unit in pruning_units:
                    if unit.id in scores_data:
                        unit.importance_score = scores_data[unit.id]['importance_score']
        else:
            logger.info("Calculating importance scores...")
            scorer = ImportanceScorer(
                model=model,
                tokenizer=tokenizer,
                config=self.config['pruning']['importance'],  # Pass FLAP config
                calibration_dataloader=eval_dataloader,
                device=self.device
            )
            
            # Use FLAP's importance computation
            pruning_units = scorer.compute_group_importances(pruning_units)
            
            # Save computed scores
            scores_data = {
                unit.id: {
                    'importance_score': float(unit.importance_score),
                    'layer_idx': unit.layer_idx,
                    'head_idx': unit.head_idx
                }
                for unit in pruning_units
            }
            
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scores_path, 'w') as f:
                json.dump(scores_data, f, indent=2)
        
        return pruning_units
        
    def _train_agent(self, env, agent):
        """Train the RL agent"""
        config = self.config['training']['optimization']
        best_reward = float('-inf')
        patience = 0
        
        for episode in range(config['num_episodes']):
            # Train episode
            stats = agent.train_episode(env)
            
            # Log progress
            if (episode + 1) % self.config['training']['logging']['log_freq'] == 0:
                self._log_progress(episode, stats)
            
            # Save checkpoint if improved
            if stats['episode_reward'] > best_reward:
                best_reward = stats['episode_reward']
                self._save_checkpoint(agent, env, episode, stats['episode_reward'])
                patience = 0
            else:
                patience += 1
            
            # Early stopping
            if patience >= config['early_stopping']['patience']:
                logger.info("Early stopping triggered!")
                break
    
    def _log_progress(self, episode: int, stats: Dict[str, float]):
        """Log training progress"""
        logger.info(f"Episode {episode + 1}:")
        logger.info(f"  Reward: {stats['episode_reward']:.4f}")
        logger.info(f"  Episode Length: {stats['episode_length']}")
        logger.info(f"  Value Loss: {stats['value_loss']:.4f}")
        logger.info(f"  Policy Loss: {stats['policy_loss']:.4f}")
        logger.info(f"  Entropy Loss: {stats['entropy_loss']:.4f}")
        
        if self.config['training']['logging']['use_wandb']:
            wandb.log({
                'episode': episode + 1,
                **stats
            })
    
    def _save_checkpoint(self, agent, env, episode: int, reward: float):
        """Save training checkpoint"""
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'reward': reward,
            'agent_state_dict': agent.ac.state_dict(),
            'agent_optimizer_state_dict': agent.optimizer.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = checkpoint_dir / f'checkpoint_episode_{episode}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load best checkpoint based on rewards"""
        checkpoint_dir = self.save_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            return None
        
        checkpoints = list(checkpoint_dir.glob('checkpoint_episode_*.pt'))
        if not checkpoints:
            return None
        
        # Load all checkpoints and find best
        best_reward = float('-inf')
        best_checkpoint = None
        
        for checkpoint_path in checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if checkpoint['reward'] > best_reward:
                best_reward = checkpoint['reward']
                best_checkpoint = checkpoint
        
        return best_checkpoint
    
    def _save_final_results(self, model, env):
        """Save final results and generate report"""
        # Save final model
        model_save_dir = self.save_dir / 'final_model'
        model_save_dir.mkdir(exist_ok=True)
        
        # Save model
        model.save_pretrained(model_save_dir)
        
        # Get final metrics
        metrics = env.metrics_tracker.evaluate_model(model, env.eval_dataloader)
        env.metrics_tracker.save_metrics(metrics, 'final_metrics.json')
        
        # Generate summary report
        summary = {
            'initial_metrics': self._load_json(self.save_dir / 'metrics' / 'initial_metrics.json'),
            'final_metrics': self._load_json(self.save_dir / 'metrics' / 'final_metrics.json'),
            'pruning_summary': env.get_pruning_summary(),
            'config': self.config
        }
        
        # Save summary
        summary_path = self.save_dir / 'final_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved final results to {self.save_dir}")
        
        if self.config['training']['logging']['use_wandb']:
            wandb.log({'final_metrics': metrics})
            wandb.save(str(summary_path))
            wandb.finish()
    
    @staticmethod
    def _load_json(path: Path) -> Dict:
        """Load JSON file"""
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Run pruning pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    pipeline = PruningPipeline(args.config)
    pipeline.run()

if __name__ == '__main__':
    main()