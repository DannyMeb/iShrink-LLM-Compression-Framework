# run_pipeline.py

import os
import json
import torch
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import wandb

from src.model_loader import ModelLoader
from src.dependency_graph import DependencyGraphBuilder
from src.importance_scorer import ImportanceScorer
from src.pruning_env import PruningEnvironment
from src.rl_agent import PPOAgent
from src.rl_trainer import RLTrainer
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
            model, tokenizer, eval_dataloader = self._setup_model_and_data()
            
            # 2. Initial Model Evaluation
            logger.info("Setting up metrics tracker...")
            metrics_tracker = MetricsTracker(
                save_dir=self.save_dir,
                device=self.device,
                tokenizer=tokenizer,
                config=self.config,
                use_wandb=self.config['training']['logging']['use_wandb']
            )
            
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
            logger.info("Handling importance scores...")
            pruning_units = self._handle_importance_scores(
                model, tokenizer, eval_dataloader, pruning_units
            )
            
            # 5. Setup Environment
            logger.info("Setting up pruning environment...")
            env = PruningEnvironment(
                model=model,
                pruning_units=pruning_units,
                eval_dataloader=eval_dataloader,
                metrics_tracker=metrics_tracker,
                config=self.config,
                device=self.device,
                initial_metrics=initial_metrics
            )
            
            # 6. Setup RL Agent
            logger.info("Setting up RL agent...")
            agent = self._setup_agent(env)
            
            # 7. Setup and Run Training
            logger.info("Setting up RL trainer...")
            trainer = RLTrainer(
                agent=agent,
                env=env,
                config=self.config,
                save_dir=self.save_dir
            )
            
            logger.info("Starting RL training...")
            training_results = trainer.train()
            
            # 8. Load Best Model and Evaluate
            if training_results['best_checkpoint']:
                logger.info("Loading best checkpoint...")
                trainer.load_checkpoint(training_results['best_checkpoint'])
            
            logger.info("Evaluating final model...")
            eval_results = trainer.evaluate(num_episodes=5)
            
            # 9. Save Final Results
            logger.info("Saving final results...")
            self._save_final_results(model, env, training_results, eval_results)
            
            logger.info("Pruning pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def _setup_model_and_data(self):
        """Setup model and data loaders"""
        model_loader = ModelLoader(config=self.config['model'])
        model, tokenizer = model_loader.load()
        
        eval_dataloader, _ = create_mmlu_dataloader(
            tokenizer=tokenizer,
            config=self.config,
            split="validation"
        )
        
        return model, tokenizer, eval_dataloader
    
    def _create_pruning_units(self, model):
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
        temp_scores_path = scores_path.with_suffix('.tmp.json')
        
        try:
            if self._are_importance_scores_valid(scores_path, pruning_units):
                logger.info(f"Loading existing importance scores from {scores_path}")
                return self._load_importance_scores(scores_path, pruning_units)
            
            logger.info("Computing importance scores from scratch...")
            return self._compute_importance_scores(
                model, tokenizer, eval_dataloader, pruning_units,
                scores_path, temp_scores_path
            )
            
        except Exception as e:
            logger.error(f"Error handling importance scores: {str(e)}")
            if temp_scores_path.exists():
                temp_scores_path.unlink()
            raise
    
    def _are_importance_scores_valid(self, scores_path: Path, pruning_units) -> bool:
        """Validate existing importance scores"""
        if not scores_path.exists():
            return False
            
        try:
            with open(scores_path) as f:
                scores_data = json.load(f)
                
            unit_ids = {unit.id for unit in pruning_units}
            saved_ids = set(scores_data.keys())
            
            if not unit_ids.issubset(saved_ids):
                logger.warning("Cached importance scores are incomplete")
                return False
                
            for unit_id, data in scores_data.items():
                required_keys = ['importance_score', 'layer_idx', 'head_idx']
                if not all(key in data for key in required_keys):
                    logger.warning(f"Invalid score format for unit {unit_id}")
                    return False
            
            return True
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error validating importance scores: {str(e)}")
            return False
    
    def _compute_importance_scores(self, model, tokenizer, eval_dataloader, 
                                 pruning_units, scores_path: Path, temp_path: Path):
        """Compute importance scores from scratch"""
        try:
            scorer = ImportanceScorer(
                model=model,
                tokenizer=tokenizer,
                config=self.config['pruning']['importance'],
                calibration_dataloader=eval_dataloader,
                device=self.device
            )
            
            pruning_units = scorer.compute_group_importances(pruning_units)
            
            scores_data = {
                unit.id: {
                    'importance_score': float(unit.importance_score),
                    'layer_idx': unit.layer_idx,
                    'head_idx': unit.head_idx
                }
                for unit in pruning_units
            }
            
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, 'w') as f:
                json.dump(scores_data, f, indent=2)
            
            temp_path.replace(scores_path)
            
            logger.info(f"Successfully computed and saved importance scores to {scores_path}")
            return pruning_units
            
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def _load_importance_scores(self, scores_path: Path, pruning_units):
        """Load and apply cached importance scores"""
        with open(scores_path) as f:
            scores_data = json.load(f)
        
        for unit in pruning_units:
            if unit.id in scores_data:
                unit.importance_score = scores_data[unit.id]['importance_score']
            else:
                raise ValueError(f"Missing importance score for unit {unit.id}")
        
        logger.info(f"Successfully loaded importance scores for {len(pruning_units)} units")
        return pruning_units
    
    def _setup_agent(self, env):
        """Setup RL agent"""
        obs_dim = (
            env.observation_space['global_features'].shape[0] +
            env.observation_space['unit_features'].shape[1] +
            env.observation_space['layer_features'].shape[1]
        )
        
        return PPOAgent(
            state_dim=obs_dim,
            action_dim=len(env.pruning_units),
            config=self.config['rl']['ppo'],
            device=self.device
        )
    
    def _save_final_results(self, model, env, training_results, eval_results):
        """Save final model and results"""
        try:
            final_model_dir = self.save_dir / 'final_model'
            final_model_dir.mkdir(exist_ok=True)
            model.save_pretrained(final_model_dir)
            
            final_metrics = env.metrics_tracker.evaluate_model(
                model,
                env.eval_dataloader
            )
            
            summary = {
                'initial_metrics': self._load_json(self.save_dir / 'metrics' / 'initial_metrics.json'),
                'final_metrics': vars(final_metrics),
                'pruning_summary': env.get_pruning_summary(),
                'training_results': training_results,
                'evaluation_results': eval_results,
                'config': self.config
            }
            
            summary_path = self.save_dir / 'final_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Saved final results to {self.save_dir}")
            
            if self.config['training']['logging']['use_wandb']:
                wandb.log({
                    'final_metrics': vars(final_metrics),
                    'training_results': training_results,
                    'evaluation_results': eval_results
                })
                wandb.save(str(summary_path))
                wandb.finish()
                
        except Exception as e:
            logger.error(f"Error saving final results: {str(e)}")
            raise
    
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