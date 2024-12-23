# run_pipeline.py

import os
import numpy as np
import json
from src.verify import ModelVerifier
import torch
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple
import wandb
from dataclasses import dataclass
import torch.nn as nn
import time
from datetime import datetime, timedelta

from src.model_loader import ModelLoader
from src.zero_out import ProgressiveSparsifier
from src.pruning_units import DependencyGraphBuilder
from src.importance_scorer import ImportanceScorer
from src.adaptive_pruner import StructuralPruner, PruningResult
from src.metrics import MetricsTracker, ModelMetrics
from src.data import create_mmlu_dataloader


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure specific loggers
logger = logging.getLogger(__name__)
logging.getLogger('lm-eval').setLevel(logging.ERROR)  # Suppress lm-eval warnings
logging.getLogger('transformers').setLevel(logging.WARNING)  # Suppress transformer warnings if needed
logging.getLogger('torch').setLevel(logging.WARNING)  # Suppress torch warnings if needed

@dataclass
class PipelineState:
    """Tracks the state of the pruning pipeline"""
    model: torch.nn.Module
    tokenizer: Any
    eval_dataloader: torch.utils.data.DataLoader
    initial_metrics: ModelMetrics
    metrics_tracker: MetricsTracker
    pruning_units: list

class PruningPipeline:
    """Main pipeline for model pruning"""
    
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
        self._set_random_seeds()
        
        logger.info(f"Initialized pruning pipeline with device: {self.device}")
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        seed = self.config['system']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration"""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self._validate_config(config)
        return config
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration parameters"""
        required_sections = ['model', 'pruning', 'training', 'system']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
    
    def _setup_metrics_tracker(self, tokenizer) -> MetricsTracker:
        """Setup metrics tracking with appropriate configuration"""
        logger.info("Setting up metrics tracker...")
        return MetricsTracker(
            save_dir=self.save_dir,
            device=self.device,
            tokenizer=tokenizer,
            config=self.config,
            use_wandb=self.config['training']['logging']['use_wandb']
        )
    
    def _setup_model_and_data(self) -> Tuple[torch.nn.Module, Any, torch.utils.data.DataLoader]:
        """Setup model and data loaders"""
        try:
            model_loader = ModelLoader(config=self.config['model'])
            model, tokenizer = model_loader.load()
            
            # eval_dataloader, _ = create_mmlu_dataloader(
            #     tokenizer=tokenizer,
            #     config=self.config,
            #     split="validation"
            # )
            
            return model, tokenizer, None
            
        except Exception as e:
            logger.error(f"Error setting up model and data: {str(e)}")
            raise
    
    def _handle_initial_evaluation(self, model: torch.nn.Module, tokenizer: Any) -> Tuple[ModelMetrics, MetricsTracker]:
        metrics_tracker = self._setup_metrics_tracker(tokenizer)
        initial_metrics_path = self.save_dir / 'metrics' / 'initial_metrics.json'
        
        try:
            if initial_metrics_path.exists():
                logger.info("Found cached initial metrics, loading...")
                initial_metrics = metrics_tracker.load_metrics('initial_metrics.json')
            else:
                logger.info("No cached metrics found. Evaluating initial model...")
                initial_metrics = metrics_tracker.evaluate_model(model, tokenizer, verbose=True)  # Only log here
                metrics_tracker.save_metrics(initial_metrics, 'initial_metrics.json')
            
            return initial_metrics, metrics_tracker
                
        except Exception as e:
            logger.error(f"Error during initial model evaluation: {str(e)}")
            if initial_metrics_path.exists():
                logger.warning("Error with cached metrics. Removing cache and retrying...")
                initial_metrics_path.unlink()
                return self._handle_initial_evaluation(model, tokenizer)
            raise
    
    def _create_pruning_units(self, model: torch.nn.Module, layer_percent: float) -> list:
        """Create pruning units for attention heads"""
        graph_builder = DependencyGraphBuilder(
            model=model,
            config=self.config['pruning']['dependency']
        )
        pruning_units, _ = graph_builder.build(percent=layer_percent)
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


    def save_results(self, model: torch.nn.Module, tokenizer: Any, pruning_result: PruningResult, initial_metrics: ModelMetrics):
        """Save pruning results based on current PruningResult structure"""
        try:
            # Save the final model
            # final_model_dir = self.save_dir / 'final_model'
            # final_model_dir.mkdir(exist_ok=True)
            # model.save_pretrained(final_model_dir)
            # tokenizer.save_pretrained(final_model_dir)

            # Create summary using available attributes from PruningResult
            summary = {
                'initial_size': {
                    'num_heads': pruning_result.original_size['num_heads'],
                    'num_kv_heads': pruning_result.original_size['num_kv_heads'],
                    'intermediate_size': pruning_result.original_size['intermediate_size'],
                    'hidden_size': pruning_result.original_size['hidden_size']
                },
                'pruned_size': {
                    'num_heads': pruning_result.pruned_size['num_heads'],
                    'num_kv_heads': pruning_result.pruned_size['num_kv_heads'],
                    'intermediate_size': pruning_result.pruned_size['intermediate_size'],
                    'hidden_size': pruning_result.pruned_size['hidden_size']
                },
                'compression_metrics': {
                    'compression_ratio': pruning_result.compression_ratio,
                    'parameters_before': pruning_result.params_before,
                    'parameters_after': pruning_result.params_after,
                    'actual_compression': pruning_result.actual_compression
                },
                'initial_metrics': {
                    'accuracy': initial_metrics.accuracy,
                    'latency': initial_metrics.latency,
                    'throughput': initial_metrics.throughput,
                    'parameter_count': initial_metrics.parameter_count,
                    'active_parameter_count': initial_metrics.active_parameter_count,
                    'compute_metrics': vars(initial_metrics.compute_metrics),
                    'cost_metrics': vars(initial_metrics.cost_metrics),
                    'memory_footprint': initial_metrics.memory_footprint
                }
            }

            # Save masks for attention and MLP layers
            summary['pruning_masks'] = {
                'attention_masks': {
                    layer_idx: mask.tolist() 
                    for layer_idx, mask in pruning_result.attention_mask.items()
                },
                'mlp_masks': {
                    layer_idx: mask.tolist() 
                    for layer_idx, mask in pruning_result.mlp_mask.items()
                }
            }

            # Log compression results
            logger.info("\n=== Pruning Results ===")
            logger.info(f"Original Parameters: {pruning_result.params_before:,}")
            logger.info(f"Pruned Parameters: {pruning_result.params_after:,}")
            logger.info(f"Compression Ratio: {pruning_result.compression_ratio:.2%}")
            logger.info(f"Actual Compression: {pruning_result.actual_compression:.2%}")
            
            # Save dimensions changes
            logger.info("\n=== Model Dimensions ===")
            logger.info(f"Original Attention Heads: {pruning_result.original_size['num_heads']}")
            logger.info(f"Pruned Attention Heads: {pruning_result.pruned_size['num_heads']}")
            logger.info(f"Original KV Heads: {pruning_result.original_size['num_kv_heads']}")
            logger.info(f"Pruned KV Heads: {pruning_result.pruned_size['num_kv_heads']}")
            logger.info(f"Original Intermediate Size: {pruning_result.original_size['intermediate_size']}")
            logger.info(f"Pruned Intermediate Size: {pruning_result.pruned_size['intermediate_size']}")

            # Save summary to a JSON file
            summary_path = self.save_dir / 'pruning_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            logger.info(f"\nResults saved to {summary_path}")
            
            # Log to wandb if enabled
            if self.config['training']['logging']['use_wandb']:
                wandb.log(summary)

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def _log_timing_results(self, total_time: float, stage_times: Dict[str, float]):
        """Log detailed timing results"""
        logger.info("\n=== Experiment Timing ===")
        logger.info(f"Total experiment time: {timedelta(seconds=int(total_time))}")
        logger.info("\nStage-wise timing:")
        for stage, duration in stage_times.items():
            percentage = (duration / total_time) * 100
            logger.info(f"{stage}: {timedelta(seconds=int(duration))} ({percentage:.1f}%)")


    def run(self):
        """Execute the complete pruning pipeline"""
        start_time = time.time()
        stage_times = {}
        try:
            # 1. Load Model and Data
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Loading model and data...")
            model, tokenizer, eval_dataloader = self._setup_model_and_data()
            stage_times['model_loading'] = time.time() - stage_start
        
            # 2. Initial Model Evaluation
            logger.info("\n" + "="*50)
            logger.info("Initial evaluation....")
            stage_start = time.time()
            initial_metrics, metrics_tracker = self._handle_initial_evaluation(model, tokenizer)
            stage_times['initial_evaluation'] = time.time() - stage_start
            
            # 3. Build Pruning Units
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Creating pruning units...")
            layer_percent = self.config['pruning']['dependency'].get('layer_percentage', 100.0)
            pruning_units = self._create_pruning_units(model, layer_percent)
            stage_times['build_units'] = time.time() - stage_start
            
            # 4. Calculate Importance Scores
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Handling importance scores...")
            pruning_units = self._handle_importance_scores(
                model, tokenizer, eval_dataloader, pruning_units
            )
            stage_times['importance_scoring'] = time.time() - stage_start
            
            # 4.5 Progressive Sparsification Analysis
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Starting Progressive Sparsification Analysis")
            
            sparsifier = ProgressiveSparsifier(
                model=model,
                tokenizer=tokenizer,
                save_dir=self.save_dir / 'sparsified_models'
            )
            
            try:
                sparsify_timing = sparsifier.sparsify(pruning_units)
                stage_times['progressive_sparsification'] = time.time() - stage_start
                # Add detailed timings
                for timing_key, timing_value in sparsify_timing.items():
                    stage_times[f'sparsify_{timing_key}'] = timing_value
                    
                logger.info("\nProgressive sparsification completed successfully")
                
            except Exception as e:
                logger.error(f"Error during progressive sparsification: {str(e)}")
                logger.warning("Continuing with main pruning pipeline...")

            # 5. Setup Pruner
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Setting up progressive pruner...")
            pruner = StructuralPruner(
                model=model,
                config=self.config,
                attention_sparsity=0, 
                mlp_sparsity=0.35
            )
            stage_times['setup_pruner'] = time.time() - stage_start
            
            # 6. Run Pruning
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Starting progressive pruning...")
            pruning_result = pruner.optimize_pruning(pruning_units)
            stage_times['pruning'] = time.time() - stage_start
            
            # 7. Save Results
            stage_start = time.time()
            logger.info("\n" + "="*50)
            logger.info("Saving final results...")
            save_path = self.save_dir / 'final_model'
            pruner.save_model(pruning_result, tokenizer, save_path)
            stage_times['save_results'] = time.time() - stage_start
            del model
            del pruning_result.pruned_model
            torch.cuda.empty_cache()
            

            # 8.  Run verification
            config_path="config/config.yaml"
            logger.info("Starting model verification...")
            verifier = ModelVerifier(config_path)
            # final_metrics = verifier.verify()
    
            total_time = time.time() - start_time
            self._log_timing_results(total_time, stage_times)
            
            return pruning_result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            if self.config['training']['logging']['use_wandb']:
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