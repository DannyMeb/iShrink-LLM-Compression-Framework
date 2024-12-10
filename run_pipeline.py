# run_pipeline.py

import os
import numpy as np
import json
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
from src.dependency_graph import DependencyGraphBuilder
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
            
            eval_dataloader, _ = create_mmlu_dataloader(
                tokenizer=tokenizer,
                config=self.config,
                split="validation"
            )
            
            return model, tokenizer, eval_dataloader
            
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
        try:
            # Save the final model
            final_model_dir = self.save_dir / 'final_model'
            final_model_dir.mkdir(exist_ok=True)
            model.save_pretrained(final_model_dir)
            tokenizer.save_pretrained(final_model_dir)

            # Convert layer statistics to dictionary
            layer_stats_dict = {
                layer_idx: {
                    'attention_units': {
                        'pruned': stats.attention_units_pruned,
                        'total': stats.total_attention_units,
                        'remaining': stats.total_attention_units - stats.attention_units_pruned,
                        'prune_ratio': float(stats.attention_units_pruned / stats.total_attention_units)
                    },
                    'mlp_units': {
                        'pruned': stats.mlp_units_pruned,
                        'total': stats.total_mlp_units,
                        'remaining': stats.total_mlp_units - stats.mlp_units_pruned,
                        'prune_ratio': float(stats.mlp_units_pruned / stats.total_mlp_units)
                    },
                    'memory_saved': float(stats.memory_saved)
                }
                for layer_idx, stats in pruning_result.layer_stats.items()
            }

            # Convert batch statistics to a list of dictionaries
            batch_stats_list = [
                {
                    'batch_number': batch.batch_number,
                    'units_pruned': batch.units_pruned,
                    'accuracy': float(batch.accuracy),
                    'accuracy_drop': float(batch.accuracy_drop),
                    'memory_reduction': float(batch.memory_reduction),
                    'remaining_memory': float(batch.remaining_memory)
                }
                for batch in pruning_result.batch_stats
            ]

            # Calculate computational savings
            computational_savings = {
                'flops_reduction': float(
                    (initial_metrics.flops - pruning_result.final_metrics.flops) / initial_metrics.flops * 100
                ) if initial_metrics.flops > 0 else 0.0,
                'active_parameter_reduction': float(
                    (initial_metrics.active_parameter_count - pruning_result.final_metrics.active_parameter_count) /
                    initial_metrics.active_parameter_count * 100
                ) if initial_metrics.active_parameter_count > 0 else 0.0,
                'latency_reduction': float(
                    (initial_metrics.latency - pruning_result.final_metrics.latency) / initial_metrics.latency * 100
                ) if initial_metrics.latency > 0 else 0.0,
                'throughput_improvement': float(
                    (pruning_result.final_metrics.throughput - initial_metrics.throughput) / initial_metrics.throughput * 100
                ) if initial_metrics.throughput > 0 else 0.0,
                'cost_reduction': float(
                    (initial_metrics.cost_metrics.inference_cost_usd - pruning_result.final_metrics.cost_metrics.inference_cost_usd) /
                    initial_metrics.cost_metrics.inference_cost_usd * 100
                ) if initial_metrics.cost_metrics.inference_cost_usd > 0 else 0.0,
                'co2_reduction': float(
                    (initial_metrics.co2_emissions - pruning_result.final_metrics.co2_emissions) /
                    initial_metrics.co2_emissions * 100
                ) if initial_metrics.co2_emissions > 0 else 0.0
            }

            # Create summary
            summary = {
                'initial_metrics': {
                    'accuracy': initial_metrics.accuracy,
                    'latency': initial_metrics.latency,
                    'throughput': initial_metrics.throughput,
                    'parameter_count': initial_metrics.parameter_count,
                    'active_parameter_count': initial_metrics.active_parameter_count,
                    'compute_metrics': vars(initial_metrics.compute_metrics),
                    'cost_metrics': vars(initial_metrics.cost_metrics),
                    'memory_footprint': initial_metrics.memory_footprint
                },
                'final_metrics': {
                    'accuracy': pruning_result.final_metrics.accuracy,
                    'latency': pruning_result.final_metrics.latency,
                    'throughput': pruning_result.final_metrics.throughput,
                    'parameter_count': pruning_result.final_metrics.parameter_count,
                    'active_parameter_count': pruning_result.final_metrics.active_parameter_count,
                    'compute_metrics': vars(pruning_result.final_metrics.compute_metrics),
                    'cost_metrics': vars(pruning_result.final_metrics.cost_metrics),
                    'memory_footprint': pruning_result.final_metrics.memory_footprint
                },
                'pruning_summary': {
                    'total_units_pruned': len(pruning_result.pruned_units),
                    'memory_reduction_mb': pruning_result.memory_reduction,
                    'performance_impact': pruning_result.performance_impact,
                    'computational_savings': computational_savings,
                    'layer_statistics': layer_stats_dict,
                    'batch_statistics': batch_stats_list
                }
            }

            # Save summary to a JSON file
            summary_path = self.save_dir / 'pruning_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            # Log final results
            logger.info("\n=== Final Pruning Results ===")
            logger.info(f"Total Memory Reduction: {pruning_result.memory_reduction:.2f} MB")
            logger.info(f"Final Accuracy: {pruning_result.final_metrics.accuracy:.4f}")
            logger.info(f"Performance Impact: {pruning_result.performance_impact * 100:.2f}%")
            logger.info("\n=== Computational Savings ===")
            for key, value in computational_savings.items():
                logger.info(f"{key.replace('_', ' ').title()}: {value:.2f}%")

            # Log statistics
            logger.info("\n=== Layer-Wise Statistics ===")
            for layer_idx, stats in sorted(layer_stats_dict.items()):
                logger.info(
                    f"Layer {layer_idx}: "
                    f"Attention Units Pruned {stats['attention_units']['pruned']} / {stats['attention_units']['total']} "
                    f"({stats['attention_units']['prune_ratio'] * 100:.2f}%), "
                    f"MLP Units Pruned {stats['mlp_units']['pruned']} / {stats['mlp_units']['total']} "
                    f"({stats['mlp_units']['prune_ratio'] * 100:.2f}%)"
                )

            # Log batch progress
            logger.info("\n=== Batch-Wise Statistics ===")
            for batch in batch_stats_list:
                logger.info(
                    f"Batch {batch['batch_number']}: "
                    f"Units Pruned: {batch['units_pruned']}, "
                    f"Accuracy: {batch['accuracy']:.4f} (Drop: {batch['accuracy_drop']:.4f}), "
                    f"Memory Reduction: {batch['memory_reduction']:.2f} MB"
                )

            # Log summary to WandB
            wandb.log(summary)

        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def run(self):
        """Execute the complete pruning pipeline"""
        start_time = time.time()
        stage_times = {}
        try:
            # 1. Load Model and Data
            stage_start = time.time()
            logger.info("Loading model and data...")
            model, tokenizer, eval_dataloader = self._setup_model_and_data()
            stage_times['model_loading'] = time.time() - stage_start
        
            # 2. Initial Model Evaluation
            stage_start = time.time()
            initial_metrics, metrics_tracker = self._handle_initial_evaluation(model, tokenizer)
            stage_times['initial_evaluation'] = time.time() - stage_start
            
            # 3. Build Pruning Units
            stage_start = time.time()
            logger.info("Creating pruning units...")
            layer_percent = self.config['pruning']['dependency'].get('layer_percentage', 100.0)
            pruning_units = self._create_pruning_units(model, layer_percent)
            stage_times['build_units'] = time.time() - stage_start
            
            # 4. Calculate Importance Scores
            stage_start = time.time()
            logger.info("Handling importance scores...")
            pruning_units = self._handle_importance_scores(
                model, tokenizer, eval_dataloader, pruning_units
            )
            stage_times['importance_scoring'] = time.time() - stage_start
            
            # 5. Setup Pruner
            stage_start = time.time()
            logger.info("Setting up progressive pruner...")
            pruner = StructuralPruner(
                model=model,
                config=self.config,
                target_sparsity=0.5
            )
            stage_times['setup_pruner'] = time.time() - stage_start
            
            # 6. Run Pruning
            stage_start = time.time()
            logger.info("Starting progressive pruning...")
            pruning_result = pruner.optimize_pruning(pruning_units)
            stage_times['pruning'] = time.time() - stage_start
            
            # 7. Save Results
            stage_start = time.time()
            logger.info("Saving final results...")
            save_path = self.save_dir / 'final_model'
            pruner.save_model(pruning_result, tokenizer, save_path)
            stage_times['save_results'] = time.time() - stage_start
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # final_metrics = self.metrics_tracker.evaluate_model(
            #     pruning_result.pruned_model,
            #     tokenizer
            # )
            
            # self.save_results(
            #     pruning_result.pruned_model, 
            #     tokenizer,
            #     pruning_result,
            #     initial_metrics
            # )

            # Log timing results
            logger.info("\n=== Experiment Timing ===")
            logger.info(f"Total experiment time: {timedelta(seconds=int(total_time))}")
            logger.info("\nStage-wise timing:")
            for stage, duration in stage_times.items():
                percentage = (duration / total_time) * 100
                logger.info(f"{stage}: {timedelta(seconds=int(duration))} ({percentage:.1f}%)")
                
            
            logger.info("Pruning pipeline completed successfully!")
            
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