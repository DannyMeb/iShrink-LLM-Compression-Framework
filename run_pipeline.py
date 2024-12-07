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

from src.model_loader import ModelLoader
from src.dependency_graph import DependencyGraphBuilder
from src.importance_scorer import ImportanceScorer
from src.adaptive_pruner import LayerProgressivePruner, PruningResult
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

   

    def save_results(self, model: torch.nn.Module, pruning_result: PruningResult, initial_metrics: ModelMetrics):
        try:
            # Save the final model
            final_model_dir = self.save_dir / 'final_model'
            final_model_dir.mkdir(exist_ok=True)
            model.save_pretrained(final_model_dir)

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
        
        try:
            # 1. Load Model and Data
            logger.info("Loading model and data...")
            model, tokenizer, eval_dataloader = self._setup_model_and_data()
            
            # def calculate_flops_and_params(model: nn.Module, max_seq_length: int) -> Dict[str, float]:
            #     """
            #     Calculate FLOPS and parameters for a LLaMA model
            #     Args:
            #         model: The LLaMA model
            #         max_seq_length: Maximum sequence length
            #     Returns:
            #         Dictionary containing FLOPS and parameter counts
            #     """
            #     config = model.config
                
            #     # Core architecture parameters
            #     hidden_size = config.hidden_size  # 2048
            #     num_layers = config.num_hidden_layers  # 16
            #     num_attention_heads = config.num_attention_heads  # 32
            #     num_kv_heads = config.num_key_value_heads  # 8
            #     head_dim = config.head_dim  # 64
            #     intermediate_size = config.intermediate_size  # 8192
            #     vocab_size = config.vocab_size  # 128256
                
            #     # Log configuration
            #     logger.info(f"\nModel Configuration:")
            #     logger.info(f"hidden_size: {hidden_size}")
            #     logger.info(f"num_layers: {num_layers}")
            #     logger.info(f"num_attention_heads: {num_attention_heads}")
            #     logger.info(f"num_key_value_heads: {num_kv_heads}")
            #     logger.info(f"head_dim: {head_dim}")
            #     logger.info(f"intermediate_size: {intermediate_size}")
            #     logger.info(f"vocab_size: {vocab_size}")

            #     # Embedding parameters
            #     embedding_params = vocab_size * hidden_size
            #     embedding_nonzero = torch.count_nonzero(model.model.embed_tokens.weight).item()
                
            #     # Per layer parameter calculations
            #     # For attention: Q projection (hidden x hidden), K and V projections (hidden x hidden/4 each due to GQA)
            #     # and output projection (hidden x hidden)
            #     qkv_params_per_layer = (hidden_size * hidden_size +  # Q projection
            #                         2 * hidden_size * (hidden_size // 4))  # K and V projections (grouped)
            #     attention_out_params_per_layer = hidden_size * hidden_size
            #     attention_params_per_layer = qkv_params_per_layer + attention_out_params_per_layer
                
            #     # MLP parameters per layer: gate_proj, up_proj, down_proj
            #     mlp_params_per_layer = 3 * hidden_size * intermediate_size
                
            #     # Total parameters
            #     total_attention_params = attention_params_per_layer * num_layers
            #     total_mlp_params = mlp_params_per_layer * num_layers
            #     total_params = embedding_params + total_attention_params + total_mlp_params
                
            #     # Count nonzero parameters
            #     total_attention_nonzero = 0
            #     total_mlp_nonzero = 0
            #     total_flops = 0
                
            #     for layer in model.model.layers:
            #         # Count nonzeros
            #         attention_nonzero = sum(torch.count_nonzero(p).item() for p in layer.self_attn.parameters())
            #         mlp_nonzero = sum(torch.count_nonzero(p).item() for p in layer.mlp.parameters())
                    
            #         total_attention_nonzero += attention_nonzero
            #         total_mlp_nonzero += mlp_nonzero
                    
            #         # Calculate ratios
            #         attn_ratio = attention_nonzero / attention_params_per_layer
            #         mlp_ratio = mlp_nonzero / mlp_params_per_layer
                    
            #         # FLOPS calculation
            #         # 1. Attention
            #         # Q projection for all heads
            #         q_proj_flops = max_seq_length * hidden_size * hidden_size * attn_ratio
            #         # K,V projections for kv_heads
            #         kv_proj_flops = 2 * max_seq_length * hidden_size * (hidden_size // 4) * attn_ratio
            #         # Attention scores and softmax (per head)
            #         attn_scores_flops = num_attention_heads * max_seq_length * max_seq_length * head_dim
            #         attn_softmax_flops = num_attention_heads * max_seq_length * max_seq_length
            #         # Attention output
            #         attn_output_flops = max_seq_length * hidden_size * hidden_size * attn_ratio
                    
            #         # 2. MLP with SwiGLU
            #         mlp1_flops = 2 * max_seq_length * hidden_size * intermediate_size * mlp_ratio  # gate and up
            #         mlp2_flops = max_seq_length * intermediate_size * hidden_size * mlp_ratio  # down
                    
            #         # 3. RMSNorm (2 per layer)
            #         ln_flops = 4 * max_seq_length * hidden_size
                    
            #         layer_flops = (q_proj_flops + kv_proj_flops + attn_scores_flops + 
            #                     attn_softmax_flops + attn_output_flops + mlp1_flops + 
            #                     mlp2_flops + ln_flops)
            #         total_flops += layer_flops
                    
            #         # logger.info(f"\nLayer Analysis:")
            #         # logger.info(f"Attention - params: {attention_params_per_layer}, nonzero: {attention_nonzero}")
            #         # logger.info(f"MLP - params: {mlp_params_per_layer}, nonzero: {mlp_nonzero}")
            #         # logger.info(f"Layer FLOPS: {layer_flops:,.0f}")

            #     # Add embedding lookup FLOPS
            #     total_flops += max_seq_length
                
            #     nonzero_params = embedding_nonzero + total_attention_nonzero + total_mlp_nonzero
                
            #     return {
            #         'total_flops': float(total_flops),
            #         'total_params': int(total_params),
            #         'nonzero_params': int(nonzero_params),
            #         'sparsity': float(1 - (nonzero_params / total_params)) if total_params > 0 else 0.0,
            #         'attention_params': int(total_attention_params),
            #         'attention_nonzero': int(total_attention_nonzero),
            #         'attention_sparsity': float(1 - (total_attention_nonzero / total_attention_params)),
            #         'mlp_params': int(total_mlp_params),
            #         'mlp_nonzero': int(total_mlp_nonzero),
            #         'mlp_sparsity': float(1 - (total_mlp_nonzero / total_mlp_params)),
            #         'layers': num_layers,
            #         'hidden_size': hidden_size,
            #         'intermediate_size': intermediate_size,
            #         'num_attention_heads': num_attention_heads,
            #         'num_kv_heads': num_kv_heads,
            #         'embedding_params': embedding_params
            #     }

            # def apply_random_pruning(model: nn.Module, prune_ratio: float = 0.5) -> nn.Module:
            #     """
            #     Randomly prune model parameters
            #     Args:
            #         model: The transformer model
            #         prune_ratio: Ratio of parameters to prune (0 to 1)
            #     Returns:
            #         Pruned model
            #     """
            #     logger.info(f"Applying random {prune_ratio:.1%} pruning to model...")
            #     with torch.no_grad():
            #         total_params = 0
            #         pruned_params = 0
            #         for name, param in model.named_parameters():
            #             if 'weight' in name:  # Only prune weights
            #                 mask = (torch.rand_like(param) > prune_ratio).to(param.dtype)
            #                 param.data.mul_(mask)
            #                 total_params += param.numel()
            #                 pruned_params += (mask == 0).sum().item()
                            
            #     logger.info(f"Pruned {pruned_params}/{total_params} parameters ({pruned_params/total_params:.2%})")
            #     return model       
            
            # initial_stats = calculate_flops_and_params(
            # model, 
            # self.config['model']['max_seq_length'])
            # logger.info("\nInitial model statistics:")
            # print(initial_stats)
            
            # # Apply random pruning
            # model = apply_random_pruning(model, prune_ratio=0.5)
            
            # logger.info("Calculating stats after random pruning...")
            # pruned_stats = calculate_flops_and_params(model, self.config['model']['max_seq_length'])
            # print(pruned_stats)
        
            # 2. Initial Model Evaluation
            initial_metrics, metrics_tracker = self._handle_initial_evaluation(model, tokenizer)
            
            # 3. Build Pruning Units
            logger.info("Creating pruning units...")
            layer_percent = self.config['pruning']['dependency'].get('layer_percentage', 100.0)
            pruning_units = self._create_pruning_units(model, layer_percent)
            
            # 4. Calculate Importance Scores
            logger.info("Handling importance scores...")
            pruning_units = self._handle_importance_scores(
                model, tokenizer, eval_dataloader, pruning_units
            )
            
            # 5. Setup Pruner
            logger.info("Setting up progressive pruner...")
            pruner = LayerProgressivePruner(
                model=model,
                config=self.config,
                device=self.device,
                initial_metrics=initial_metrics,
                metrics_tracker=metrics_tracker,
                eval_dataloader=eval_dataloader,
                tokenizer=tokenizer
            )
            
            # 6. Run Pruning
            logger.info("Starting progressive pruning...")
            pruning_result = pruner.optimize_pruning(pruning_units)
            
            # 7. Save Results
            logger.info("Saving final results...")
            self.save_results(model, pruning_result, initial_metrics)
            
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