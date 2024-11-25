import os
import json
import torch
import yaml
import numpy as np
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import wandb
from datasets import load_dataset
from tqdm import tqdm

import networkx as nx
from torch.utils.data import DataLoader, Dataset

# Import our components
from src.model_loader import ModelLoader
from src.dependency_graph import DependencyGraphBuilder, PruningUnit
from src.importance_scorer import ImportanceScorer
from src.pruning_env import PruningEnvironment
from src.rl_agent import PPOAgent
from src.metrics import MetricsTracker
from src.utils import setup_device, create_dataloader
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
        self.device = setup_device(self.config['model']['device'])
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
                
        # Add validation for pruning config
        required_pruning_configs = ['dependency', 'importance']
        for config_type in required_pruning_configs:
            if config_type not in config['pruning']:
                raise ValueError(f"Missing required pruning config: {config_type}")
    
    def run(self):
        """Execute the complete pruning pipeline"""
        try:
            # 1. Load Model and Data
            logger.info("Loading model and data...")
            model, tokenizer, eval_dataloader = self._setup_model_and_data()
            
            # 2. Initial Model Evaluation
            logger.info("Evaluating initial model...")
            metrics_tracker = MetricsTracker(
                save_dir=self.save_dir,
                device=self.device,
                use_wandb=self.config['training']['logging']['use_wandb']
            )
            initial_metrics = metrics_tracker.evaluate_model(model, eval_dataloader)
            
            # 3. Build Pruning Units (Attention Heads)
            logger.info("Creating pruning units...")
            pruning_units = self._create_pruning_units(model)
            
            # 4. Calculate or Load Importance Scores
            scores_path = self.save_dir / 'importance_scores' / 'head_importance_scores.json'
            if scores_path.exists():
                logger.info(f"Loading existing importance scores from {scores_path}")
                self._load_importance_scores(pruning_units, scores_path)
            else:
                logger.info("Calculating importance scores...")
                scorer = self._setup_importance_scorer(model, tokenizer, eval_dataloader)
                self._calculate_importance_scores(pruning_units, scorer)
            
            # 5. Setup Environment with Initial Metrics
            logger.info("Setting up pruning environment...")
            env = PruningEnvironment(
                model=model,
                pruning_units=pruning_units,
                eval_dataloader=eval_dataloader,
                metrics_tracker=metrics_tracker,
                config=self.config['pruning']['env'],
                device=self.device
            )
            
            # 6. Setup RL Agent
            logger.info("Setting up RL agent...")
            agent = PPOAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                config=self.config['rl']['ppo'],
                device=self.device
            )
            
            # 7. Train Agent
            logger.info("Starting RL training...")
            self._train_agent(env, agent)
            
            # 8. Save Final Results
            logger.info("Saving results...")
            self._save_results(model, env, agent)
            
            logger.info("Pruning pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _setup_model_and_data(self) -> Tuple[torch.nn.Module, Any, DataLoader]:
        """Setup model and data"""
        # Load model
        model_loader = ModelLoader(
            config=self.config['model']
        )
        model, tokenizer = model_loader.load()
        
        # Create evaluation dataloader
        eval_dataloader = self._create_eval_dataloader(tokenizer)
        
        return model, tokenizer, eval_dataloader
    
    def _create_eval_dataloader(self, tokenizer) -> DataLoader:
        # Load WikiText dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        def tokenize_function(examples):
            # Remove return_tensors="pt" from here
            return tokenizer(examples["text"],
                            truncation=True,
                            max_length=self.config["model"]["max_seq_length"],
                            padding="max_length")
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Convert to torch format
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        # Create dataloader
        eval_dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.config["training"]["data"]["batch_size"],
            shuffle=False,
            num_workers=self.config["system"]["num_workers"]
        )
        
        return eval_dataloader
    
    def _create_pruning_units(self, model) -> List[PruningUnit]:
        """Create pruning units for attention heads"""
        graph_builder = DependencyGraphBuilder(
            model=model,
            config=self.config['pruning']['dependency']
        )
        pruning_units, _ = graph_builder.build()  # We only need the units, not the graph
        return pruning_units
    
    def _setup_importance_scorer(self, model, tokenizer, eval_dataloader) -> ImportanceScorer:
        """Setup importance scorer"""
        return ImportanceScorer(
            model=model,
            tokenizer=tokenizer,
            config=self.config['pruning']['importance'],
            calibration_dataloader=eval_dataloader,
            device=self.device
        )
    
    # def _calculate_importance_scores(self, groups, scorer):
    #     """Calculate importance scores for all groups"""
    #     for group in tqdm(groups, desc="Calculating importance scores"):
    #         group.importance_score = scorer.compute_importance(group)

    def _calculate_importance_scores(self, groups, scorer):
        """Calculate importance scores for all groups and save them"""
        # Create directory for importance scores
        scores_dir = self.save_dir / 'importance_scores'
        scores_dir.mkdir(exist_ok=True)

        try:
            # Calculate scores
            for group in tqdm(groups, desc="Calculating importance scores"):
                group.importance_score = scorer.compute_importance(group)

            # Prepare data for saving
            scores_data = {
                group.id: {
                    'importance_score': float(group.importance_score),  # Convert to float for JSON
                    'layer_idx': group.layer_idx,
                    'head_idx': group.head_idx,
                    'metadata': {
                        'parameters': list(group.parameters.keys())  # Save parameter names
                    }
                }
                for group in groups
            }

            # Save as JSON for readability and easy loading
            json_path = scores_dir / 'head_importance_scores.json'
            with open(json_path, 'w') as f:
                json.dump(scores_data, f, indent=2)

            # Also save as numpy array for numerical operations
            scores_array = np.array([group.importance_score for group in groups])
            np.save(scores_dir / 'importance_scores.npy', scores_array)

            logger.info(f"Saved importance scores for {len(groups)} heads to {scores_dir}")

        except Exception as e:
            logger.error(f"Error saving importance scores: {str(e)}")
            raise
    
    def _load_importance_scores(self, pruning_units: List[PruningUnit], scores_path: Path):
        """Load pre-computed importance scores"""
        try:
            with open(scores_path) as f:
                scores_data = json.load(f)
                
            # Assign scores to pruning units
            for unit in pruning_units:
                if unit.id in scores_data:
                    unit.importance_score = scores_data[unit.id]['importance_score']
                else:
                    logger.warning(f"No saved importance score for unit {unit.id}")
                    unit.importance_score = 0.0  # Default score
                    
            logger.info(f"Successfully loaded importance scores for {len(pruning_units)} units")
            
        except Exception as e:
            logger.error(f"Error loading importance scores: {str(e)}")
            raise RuntimeError(f"Failed to load importance scores: {str(e)}")


    def _setup_environment(self, model, groups, eval_dataloader) -> PruningEnvironment:
        """Setup pruning environment"""
        env = PruningEnvironment(
            model=model,
            groups=groups,
            eval_dataloader=eval_dataloader,
            config=self.config['rl']['env'],
            metrics_tracker=MetricsTracker(
                save_dir=self.save_dir / 'metrics',
                original_model=model,
                device=self.device,
                use_wandb=self.config['training']['logging']['use_wandb']
            ),
            device=self.device
        )
        return env
    
    def _setup_agent(self, env) -> PPOAgent:
        """Setup RL agent"""
        return PPOAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            config=self.config['rl']['ppo'],
            device=self.device
        )
    
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
                self._save_checkpoint(agent, env, episode)
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
        logger.info(f"  Steps: {stats['episode_steps']}")
        
        if self.config['training']['logging']['use_wandb']:
            wandb.log(stats)
    
    def _save_checkpoint(self, agent, env, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = self.save_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'agent_state': agent.state_dict(),
            'env_state': env.state_dict(),
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / f'checkpoint_episode_{episode}.pt')
    
    def _final_evaluation(self, env, agent):
        """Perform final evaluation"""
        # Load best checkpoint
        best_checkpoint = self._load_best_checkpoint()
        if best_checkpoint:
            agent.load_state_dict(best_checkpoint['agent_state'])
        
        # Evaluate
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic=True)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        logger.info(f"Final evaluation reward: {total_reward:.4f}")
        return env.model
    
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
    
    def _save_results(self, best_model):
        """Save final results and generate report"""
        # Save final model
        model_save_dir = self.save_dir / 'final_model'
        model_save_dir.mkdir(exist_ok=True)
        best_model.save_pretrained(model_save_dir)
        
        # Generate and save final report
        self.metrics_tracker.generate_report('final_report.json')
        
        if self.config['training']['logging']['use_wandb']:
            wandb.save(str(self.save_dir / 'final_report.json'))
            wandb.finish()

def main():
    parser = argparse.ArgumentParser(description='Run pruning pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    pipeline = PruningPipeline(args.config)
    pipeline.run()

if __name__ == '__main__':
    main()