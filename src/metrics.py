# metrics.py

import torch
import time
import json
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import psutil
from pathlib import Path
import numpy as np
import subprocess
from thop import profile
import traceback
import wandb
import torch
import torch.nn as nn
from thop import profile
logger = logging.getLogger(__name__)

@dataclass
class ComputeMetrics:
    """Stores compute-related metrics"""
    flops: int                  # Total FLOPs
    macs: int                  # Multiply-accumulate operations
    parameter_count: int        # Total parameters
    active_parameter_count: int # Non-zero parameters
    sparsity: float            # Parameter sparsity ratio
    bandwidth_usage: float      # Memory bandwidth usage in GB/s
    cache_hits: float          # Cache hit ratio
    activation_memory: float    # Peak activation memory in MB
    attention_params: int       # Total attention parameters
    attention_nonzero: int     # Non-zero attention parameters
    attention_sparsity: float  # Attention sparsity ratio
    mlp_params: int            # Total MLP parameters
    mlp_nonzero: int          # Non-zero MLP parameters
    mlp_sparsity: float       # MLP sparsity ratio
    embedding_params: int      # Embedding parameters

@dataclass
class EnvironmentalMetrics:
    """Stores environmental impact metrics"""
    co2_emissions: float       # CO2 emissions in grams
    energy_consumption: float  # Energy consumption in joules
    power_usage: float        # Average power usage in watts


@dataclass
class CostMetrics:
    """Stores cost-related metrics"""
    inference_cost_usd: float   # Cost per inference in USD
    gpu_time_cost: float        # GPU time cost
    memory_cost: float          # Memory usage cost
    total_operation_cost: float # Total operational cost

@dataclass
class ModelMetrics:
    """Stores comprehensive model metrics."""
    accuracy: float
    latency: float  # ms
    throughput: float  # samples/second
    memory_footprint: Dict[str, float] 
    flops: int
    macs: int
    parameter_count: int
    active_parameter_count: int
    sparsity: float
    bandwidth_usage: float
    cache_hits: float
    activation_memory_mb: float
    attention_params: int
    attention_nonzero: int
    attention_sparsity: float
    mlp_params: int
    mlp_nonzero: int
    mlp_sparsity: float
    embedding_params: int
    power_watts: float
    co2_emissions: float
    cost_per_inference: float
    gpu_memory_mb: float
    compute_metrics: ComputeMetrics
    cost_metrics: CostMetrics


class GPUPowerMetrics:
    """Handles GPU power and energy measurements"""
    
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
    
    def get_gpu_power(self) -> Optional[float]:
        """Get current GPU power consumption in watts"""
        if not self.has_gpu:
            return None
            
        try:
            result = subprocess.check_output([
                'nvidia-smi',
                '--query-gpu=power.draw',
                '--format=csv,noheader,nounits'
            ]).decode('utf-8').strip()
            
            return float(result)
        except Exception as e:
            logger.warning(f"Could not measure GPU power: {str(e)}")
            return None
    
    def measure_energy_consumption(self, duration_ms: float) -> Dict[str, float]:
        """Calculate energy consumption and CO2 emissions"""
        power = self.get_gpu_power()
        if power is None:
            return {
                'energy_joules': 0.0,
                'co2_grams': 0.0,
                'power_watts': 0.0
            }
        
        # Calculate energy in joules (power * time)
        energy_joules = power * (duration_ms / 1000)  # convert ms to seconds
        
        # Estimate CO2 based on average grid carbon intensity (0.4 kg CO2/kWh)
        co2_grams = (energy_joules / 3600000) * 0.4 * 1000
        
        return {
            'energy_joules': energy_joules,
            'co2_grams': co2_grams,
            'power_watts': power
        }

class MetricsTracker:
    def __init__(
        self,
        save_dir: Path,
        device: torch.device,
        tokenizer: Any,
        config: Dict,
        use_wandb: bool = False
    ):
        """Initialize metrics tracker"""
        self.save_dir = save_dir
        self.device = device
        self.tokenizer = tokenizer
        self.config = config
        self.use_wandb = use_wandb
        self.metrics_dir = save_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize WandB
        project_name="LLM compression"
        experiment_name="Gentra"
        wandb.init(project=project_name, name=experiment_name)

        # Initialize power metrics
        self.power_metrics = GPUPowerMetrics()
        
        # Cost configuration
        self.cost_config = {
            'gpu_hour_rate': 0.50,      # USD per GPU hour
            'memory_gb_hour_rate': 0.10, # USD per GB-hour
            'operation_rate': 0.05       # USD per million FLOPs
        }
    
    def _measure_accuracy(self, model: torch.nn.Module, tokenizer) -> float:
        """Measure model accuracy using lm_eval framework"""
        try:
            from lm_eval import evaluator, tasks
            from lm_eval.models.huggingface import HFLM
            import logging
            lm_eval_logger = logging.getLogger('lm-eval')
            lm_eval_logger.setLevel(logging.ERROR)  # This will suppress warnings, only show errors
            
            model_args = {
                "pretrained": model,
                "tokenizer": tokenizer,
                "device": self.device,
                "batch_size": 8,
                "trust_remote_code": True
            }
            
            hf_model = HFLM(**model_args)
            
            results = evaluator.simple_evaluate(
                model=hf_model,
                tasks=["mmlu"],
                num_fewshot=4,
                limit=0.1,
                bootstrap_iters=10000,
                device=str(self.device)
            )
            
            if "mmlu" in results["results"]:
                mmlu_results = results["results"]["mmlu"]
                if "acc,none" in mmlu_results:
                    accuracy = mmlu_results["acc,none"]
                elif "acc" in mmlu_results:
                    accuracy = mmlu_results["acc"]
                else:
                    raise KeyError("Could not find accuracy metric in MMLU results")
            else:
                raise KeyError("Could not find MMLU results in evaluation output")
            
            return float(accuracy)
            
        except Exception as e:
            logger.error(f"Error during accuracy measurement: {str(e)}")
            raise
    
    def _measure_performance(self, model: torch.nn.Module) -> Tuple[float, float]:
        """Measure model latency and throughput with proper warmup"""
        model.eval()
        
        # Create sample input
        sample_text = "Q: What is the capital of France?\nA: Let's approach this step by step:"
        inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            padding='max_length',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Thorough warmup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(10):
                _ = model(**inputs)
                torch.cuda.synchronize()
        
        # Measure
        num_runs = 20  # Increased from 10
        latencies = []
        torch.cuda.synchronize()
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(**inputs)
                torch.cuda.synchronize()
                latencies.append((time.time() - start_time) * 1000)  # ms
        
        # Remove outliers (optional)
        latencies = sorted(latencies)[2:-2]  # Remove 2 highest and lowest
        
        avg_latency = sum(latencies) / len(latencies)
        throughput = 1000 / avg_latency  # samples/second
        
        return avg_latency, throughput
    
    def _measure_memory_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """Measure model memory footprint"""
        memory_stats = {
            'gpu_allocated': torch.cuda.memory_allocated(self.device) / 1024**2,
            'gpu_cached': torch.cuda.memory_reserved(self.device) / 1024**2,
            'cpu_memory': psutil.Process().memory_info().rss / 1024**2
        }
        return memory_stats
    
    def _measure_bandwidth_and_cache(self, model: torch.nn.Module) -> Tuple[float, float]:
        """Measure memory bandwidth and cache performance"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            start_stats = pynvml.nvmlDeviceGetMemoryInfo(handle)
            throughput_start = pynvml.nvmlDeviceGetPcieThroughput(handle, 0)
            
            self._run_inference(model)
            
            end_stats = pynvml.nvmlDeviceGetMemoryInfo(handle)
            throughput_end = pynvml.nvmlDeviceGetPcieThroughput(handle, 0)
            
            bandwidth = (throughput_end - throughput_start) / 1024  # GB/s
            cache_ratio = 1 - ((end_stats.used - start_stats.used) / start_stats.total)
            
            return bandwidth, cache_ratio
            
        except Exception as e:
            logger.warning(f"Could not measure bandwidth metrics: {e}")
            return 0.0, 0.0
    
    def _run_inference(self, model: torch.nn.Module):
        """Helper function to run a single inference pass"""
        sample_input = torch.randint(
            0, 1000,
            (1, self.config['model']['max_seq_length']),
            device=self.device
        )
        sample_mask = torch.ones_like(sample_input, device=self.device)
        
        with torch.no_grad():
            model(input_ids=sample_input, attention_mask=sample_mask)
    
    
    def _calculate_compute_metrics(self, model: nn.Module) -> ComputeMetrics:
        """Calculate compute-related metrics accounting for pruned parameters."""
        config = model.config
        
        # Core architecture parameters
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_attention_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size
        vocab_size = config.vocab_size
        max_seq_length = self.config['model']['max_seq_length']
        
        # Log configuration
        logger.info(f"\nModel Configuration:")
        logger.info(f"hidden_size: {hidden_size}")
        logger.info(f"num_layers: {num_layers}")
        logger.info(f"num_attention_heads: {num_attention_heads}")
        logger.info(f"num_key_value_heads: {num_kv_heads}")
        logger.info(f"head_dim: {head_dim}")
        logger.info(f"intermediate_size: {intermediate_size}")
        logger.info(f"vocab_size: {vocab_size}")

        # Embedding parameters
        embedding_params = vocab_size * hidden_size
        embedding_nonzero = torch.count_nonzero(model.model.embed_tokens.weight).item()

        # Per-layer calculations
        attention_params_per_layer = (
            hidden_size * hidden_size +       # Q projection
            2 * hidden_size * (hidden_size // 4) +  # K, V projections (grouped)
            hidden_size * hidden_size         # Output projection
        )
        mlp_params_per_layer = 3 * hidden_size * intermediate_size
        total_attention_params = attention_params_per_layer * num_layers
        total_mlp_params = mlp_params_per_layer * num_layers
        total_params = embedding_params + total_attention_params + total_mlp_params
        
        total_attention_nonzero = 0
        total_mlp_nonzero = 0
        total_flops = 0

        # Process each layer
        for layer in model.model.layers:
            # Count nonzeros in attention
            q_nonzero = torch.count_nonzero(layer.self_attn.q_proj.weight).item()
            k_nonzero = torch.count_nonzero(layer.self_attn.k_proj.weight).item()
            v_nonzero = torch.count_nonzero(layer.self_attn.v_proj.weight).item()
            o_nonzero = torch.count_nonzero(layer.self_attn.o_proj.weight).item()
            attention_nonzero = q_nonzero + k_nonzero + v_nonzero + o_nonzero
            
            # Count nonzeros in MLP
            gate_nonzero = torch.count_nonzero(layer.mlp.gate_proj.weight).item()
            up_nonzero = torch.count_nonzero(layer.mlp.up_proj.weight).item()
            down_nonzero = torch.count_nonzero(layer.mlp.down_proj.weight).item()
            mlp_nonzero = gate_nonzero + up_nonzero + down_nonzero
            
            total_attention_nonzero += attention_nonzero
            total_mlp_nonzero += mlp_nonzero
            
            # Ratios
            attn_ratio = attention_nonzero / attention_params_per_layer
            mlp_ratio = mlp_nonzero / mlp_params_per_layer
            
            # FLOPS calculation
            q_proj_flops = max_seq_length * hidden_size * hidden_size * attn_ratio
            kv_proj_flops = 2 * max_seq_length * hidden_size * (hidden_size // 4) * attn_ratio
            attn_scores_flops = num_attention_heads * max_seq_length * max_seq_length * head_dim
            attn_softmax_flops = num_attention_heads * max_seq_length * max_seq_length
            attn_output_flops = max_seq_length * hidden_size * hidden_size * attn_ratio
            
            mlp1_flops = 2 * max_seq_length * hidden_size * intermediate_size * mlp_ratio
            mlp2_flops = max_seq_length * intermediate_size * hidden_size * mlp_ratio
            
            ln_flops = 4 * max_seq_length * hidden_size

            layer_flops = (
                q_proj_flops + kv_proj_flops + attn_scores_flops +
                attn_softmax_flops + attn_output_flops + mlp1_flops +
                mlp2_flops + ln_flops
            )
            total_flops += layer_flops

        # Add embedding lookup FLOPS
        total_flops += max_seq_length

        # Count nonzero parameters
        nonzero_params = embedding_nonzero + total_attention_nonzero + total_mlp_nonzero

        # Calculate sparsity
        sparsity = 1 - (nonzero_params / total_params) if total_params > 0 else 0.0

        # Measure bandwidth and cache performance
        bandwidth_usage, cache_hits = self._measure_bandwidth_and_cache(model)

        # Measure activation memory
        torch.cuda.reset_peak_memory_stats()
        self._run_inference(model)
        activation_memory = torch.cuda.max_memory_allocated() / 1024**2

        return ComputeMetrics(
            flops=int(total_flops),
            macs=int(total_flops // 2),  # FLOPs divided by 2 for MACs
            parameter_count=int(total_params),
            active_parameter_count=int(nonzero_params),
            sparsity=sparsity,
            bandwidth_usage=bandwidth_usage,
            cache_hits=cache_hits,
            activation_memory=activation_memory,
            attention_params=int(total_attention_params),
            attention_nonzero=int(total_attention_nonzero),
            attention_sparsity=float(1 - (total_attention_nonzero / total_attention_params)) if total_attention_params > 0 else 0,
            mlp_params=int(total_mlp_params),
            mlp_nonzero=int(total_mlp_nonzero),
            mlp_sparsity=float(1 - (total_mlp_nonzero / total_mlp_params)) if total_mlp_params > 0 else 0,
            embedding_params=int(embedding_params)
        )

    def _calculate_cost_metrics(self, compute_metrics: ComputeMetrics, latency: float, memory_footprint: Dict[str, float], power_watts: float) -> CostMetrics:
        """Calculate cost-related metrics with corrected calculation"""
        # Convert time to hours
        time_hours = latency / (1000 * 3600)  # ms to hours
        
        # GPU time cost
        gpu_time_cost = time_hours * self.cost_config['gpu_hour_rate']
        
        # Memory cost
        memory_gb = memory_footprint['gpu_allocated'] / 1024
        memory_cost = memory_gb * time_hours * self.cost_config['memory_gb_hour_rate']
        
        # Operation cost - based on active parameters instead of raw FLOPS
        million_params = compute_metrics.active_parameter_count / 1e6
        operation_cost = million_params * self.cost_config['operation_rate']
        
        # Total cost per inference
        inference_cost = gpu_time_cost + memory_cost + operation_cost
        
        return CostMetrics(
            inference_cost_usd=inference_cost,
            gpu_time_cost=gpu_time_cost,
            memory_cost=memory_cost,
            total_operation_cost=operation_cost
        )
    
    def evaluate_model(self, model: torch.nn.Module, tokenizer, verbose: bool = False) -> ModelMetrics:
        """Comprehensive model evaluation with detailed logging."""
        try:
            logger.info("Starting model evaluation...")
            start_time = time.time()
            
            # Measure accuracy
            logger.info("Measuring accuracy...")
            accuracy = self._measure_accuracy(model, tokenizer)
            
            # Measure performance
            logger.info("Measuring performance metrics...")
            latency, throughput = self._measure_performance(model)
            
            # Calculate compute metrics
            logger.info("Calculating compute metrics...")
            compute_metrics = self._calculate_compute_metrics(model)
            
            # Measure memory usage
            logger.info("Measuring memory usage...")
            memory_stats = self._measure_memory_usage(model)
            
            # Calculate energy and power metrics
            duration_ms = (time.time() - start_time) * 1000
            energy_metrics = self.power_metrics.measure_energy_consumption(duration_ms)
            
            # Calculate cost metrics
            cost_metrics = self._calculate_cost_metrics(
                compute_metrics=compute_metrics,
                latency=latency,
                memory_footprint=memory_stats,
                power_watts=energy_metrics['power_watts']
            )
            
            # Create and return the final metrics object
            metrics = ModelMetrics(
                accuracy=accuracy,
                latency=latency,
                throughput=throughput,
                memory_footprint=memory_stats,
                flops=compute_metrics.flops,
                macs=compute_metrics.macs,
                parameter_count=compute_metrics.parameter_count,
                active_parameter_count=compute_metrics.active_parameter_count,
                sparsity=compute_metrics.sparsity,
                bandwidth_usage=compute_metrics.bandwidth_usage,
                cache_hits=compute_metrics.cache_hits,
                activation_memory_mb=compute_metrics.activation_memory,
                attention_params=compute_metrics.attention_params,
                attention_nonzero=compute_metrics.attention_nonzero,
                attention_sparsity=compute_metrics.attention_sparsity,
                mlp_params=compute_metrics.mlp_params,
                mlp_nonzero=compute_metrics.mlp_nonzero,
                mlp_sparsity=compute_metrics.mlp_sparsity,
                embedding_params=compute_metrics.embedding_params,
                power_watts=energy_metrics['power_watts'],
                co2_emissions=energy_metrics['co2_grams'],
                cost_per_inference=cost_metrics.inference_cost_usd,
                gpu_memory_mb=memory_stats['gpu_allocated'],
                compute_metrics=compute_metrics,
                cost_metrics=cost_metrics 
            )
            
            # if verbose:
            #     self._log_metrics(metrics)
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log(vars(metrics))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    
    def save_metrics(self, metrics: ModelMetrics, filename: str):
        """Save metrics to a JSON file and log to WandB."""
        try:
            # Convert the ModelMetrics object into a dictionary for saving
            metrics_dict = {
                'accuracy': metrics.accuracy,
                'latency_ms': metrics.latency,
                'throughput': metrics.throughput,
                'memory_footprint': metrics.memory_footprint,
                'flops': metrics.flops,
                'macs': metrics.macs,
                'parameter_count': metrics.parameter_count,
                'active_parameter_count': metrics.active_parameter_count,
                'sparsity': metrics.sparsity,
                'bandwidth_usage': metrics.bandwidth_usage,
                'cache_hits': metrics.cache_hits,
                'activation_memory_mb': metrics.activation_memory_mb,
                'attention_params': metrics.attention_params,
                'attention_nonzero': metrics.attention_nonzero,
                'attention_sparsity': metrics.attention_sparsity,
                'mlp_params': metrics.mlp_params,
                'mlp_nonzero': metrics.mlp_nonzero,
                'mlp_sparsity': metrics.mlp_sparsity,
                'embedding_params': metrics.embedding_params,
                'power_watts': metrics.power_watts,
                'co2_emissions': metrics.co2_emissions,
                'cost_per_inference': metrics.cost_per_inference,
                'gpu_memory_mb': metrics.gpu_memory_mb,
                'compute_metrics': vars(metrics.compute_metrics),
                'cost_metrics': vars(metrics.cost_metrics)
            }

            # Save the dictionary as a JSON file
            save_path = self.metrics_dir / filename
            with open(save_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
            
            logger.info(f"Metrics successfully saved to {save_path}")

            # Log metrics to WandB
            wandb.log(metrics_dict)

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

    
    def load_metrics(self, filename: str) -> ModelMetrics:
        """Load metrics from a JSON file."""
        metrics_path = self.metrics_dir / filename
        try:
            with open(metrics_path) as f:
                metrics_dict = json.load(f)

            # Ensure all nested metrics exist; use empty defaults if missing
            compute_metrics_dict = metrics_dict.get('compute_metrics', {})
            cost_metrics_dict = metrics_dict.get('cost_metrics', {})

            compute_metrics = ComputeMetrics(**compute_metrics_dict)
            cost_metrics = CostMetrics(**cost_metrics_dict)

            # Create ModelMetrics object with fallbacks for missing attributes
            metrics = ModelMetrics(
                accuracy=metrics_dict.get('accuracy', 0.0),
                latency=metrics_dict.get('latency_ms', 0.0),
                throughput=metrics_dict.get('throughput', 0.0),
                memory_footprint=metrics_dict.get('memory_footprint', {}),
                flops=metrics_dict.get('flops', 0),
                macs=metrics_dict.get('macs', 0),
                parameter_count=metrics_dict.get('parameter_count', 0),
                active_parameter_count=metrics_dict.get('active_parameter_count', 0),
                sparsity=metrics_dict.get('sparsity', 0.0),
                bandwidth_usage=metrics_dict.get('bandwidth_usage', 0.0),
                cache_hits=metrics_dict.get('cache_hits', 0.0),
                activation_memory_mb=metrics_dict.get('activation_memory_mb', 0.0),
                attention_params=metrics_dict.get('attention_params', 0),
                attention_nonzero=metrics_dict.get('attention_nonzero', 0),
                attention_sparsity=metrics_dict.get('attention_sparsity', 0.0),
                mlp_params=metrics_dict.get('mlp_params', 0),
                mlp_nonzero=metrics_dict.get('mlp_nonzero', 0),
                mlp_sparsity=metrics_dict.get('mlp_sparsity', 0.0),
                embedding_params=metrics_dict.get('embedding_params', 0),
                power_watts=metrics_dict.get('power_watts', 0.0),
                co2_emissions=metrics_dict.get('co2_emissions', 0.0),
                cost_per_inference=metrics_dict.get('cost_per_inference', 0.0),
                gpu_memory_mb=metrics_dict.get('gpu_memory_mb', 0.0),
                compute_metrics=compute_metrics,
                cost_metrics=cost_metrics
            )

            # Log metrics to WandB
            wandb.log(metrics_dict)

            return metrics

        except Exception as e:
            logger.error(f"Failed to load metrics from {metrics_path}: {str(e)}")
            raise


    def _log_metrics(self, metrics: ModelMetrics, prefix: str = ""):
        """Log all available metrics in a structured format."""
        logger.info(f"\n{'='*20} {prefix} Model Metrics {'='*20}")
        
        # Log basic metrics
        logger.info("\n=== Basic Performance ===")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Latency: {metrics.latency:.2f} ms")
        logger.info(f"Throughput: {metrics.throughput:.2f} samples/second")
        
        # Log compute-related metrics
        logger.info("\n=== Compute Statistics ===")
        logger.info(f"FLOPs: {metrics.flops:,}")
        logger.info(f"MACs: {metrics.macs:,}")
        logger.info(f"Parameters: {metrics.parameter_count:,}")
        logger.info(f"Active Parameters: {metrics.active_parameter_count:,}")
        logger.info(f"Sparsity: {metrics.sparsity * 100:.2f}%")
        logger.info(f"Bandwidth Usage: {metrics.bandwidth_usage:.2f} GB/s")
        logger.info(f"Cache Hits: {metrics.cache_hits:.2f}")
        logger.info(f"Activation Memory: {metrics.activation_memory_mb:.2f} MB")
        
        # Log component-level metrics
        logger.info(f"Attention Params: {metrics.attention_params:,}")
        logger.info(f"Attention Nonzero Params: {metrics.attention_nonzero:,}")
        logger.info(f"Attention Sparsity: {metrics.attention_sparsity * 100:.2f}%")
        logger.info(f"MLP Params: {metrics.mlp_params:,}")
        logger.info(f"MLP Nonzero Params: {metrics.mlp_nonzero:,}")
        logger.info(f"MLP Sparsity: {metrics.mlp_sparsity * 100:.2f}%")
        logger.info(f"Embedding Params: {metrics.embedding_params:,}")
        
        # Log environmental and cost metrics
        logger.info("\n=== Environmental Impact ===")
        logger.info(f"CO2 Emissions: {metrics.co2_emissions:.2f} g")
        logger.info(f"Energy Consumption: {metrics.power_watts:.2f} W")
        
        logger.info("\n=== Cost Analysis ===")
        logger.info(f"Cost per Inference: ${metrics.cost_per_inference:.6f}")
        
        logger.info("\n" + "="*60)


