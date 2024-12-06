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
    macs: int                   # Multiply-accumulate operations
    parameter_count: int        # Total parameters
    active_parameter_count: int # Non-zero parameters
    sparsity: float            # Parameter sparsity ratio
    bandwidth_usage: float      # Memory bandwidth usage in GB/s
    cache_hits: float          # Cache hit ratio
    activation_memory: float    # Peak activation memory in MB

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
    """Stores comprehensive model metrics"""
    accuracy: float
    latency: float  # ms
    throughput: float  # samples/second
    memory_footprint: Dict[str, float]  # Memory stats in MB
    parameter_count: int
    compute_metrics: ComputeMetrics
    environmental_metrics: EnvironmentalMetrics
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
        use_wandb: bool = True
    ):
        """Initialize metrics tracker"""
        self.save_dir = save_dir
        self.device = device
        self.tokenizer = tokenizer
        self.config = config
        self.use_wandb = use_wandb
        self.metrics_dir = save_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Calculate compute-related metrics without using thop.profile"""
        # Count parameters and calculate sparsity
        total_params = 0
        nonzero_params = 0
        total_flops = 0
        total_macs = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Count parameters
                weight = module.weight
                total_params += weight.numel()
                nonzero_params += torch.count_nonzero(weight).item()
                
                # Estimate FLOPS for linear layers
                out_features, in_features = weight.shape
                # Each output requires in_features MAC operations
                macs_per_output = in_features * (torch.count_nonzero(weight).item() / weight.numel())
                total_macs += out_features * macs_per_output
                # Each MAC is 2 FLOPS (multiply + add)
                total_flops += 2 * out_features * macs_per_output
                
                if module.bias is not None:
                    total_params += module.bias.numel()
                    nonzero_params += torch.count_nonzero(module.bias).item()
                    total_flops += out_features  # One add per output for bias
        
        # Measure memory bandwidth and cache performance
        bandwidth_usage, cache_hits = self._measure_bandwidth_and_cache(model)
        
        # Measure activation memory
        torch.cuda.reset_peak_memory_stats()
        self._run_inference(model)
        activation_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        return ComputeMetrics(
            flops=int(total_flops),
            macs=int(total_macs),
            parameter_count=nonzero_params,
            active_parameter_count=nonzero_params,
            sparsity=1 - (nonzero_params / total_params),
            bandwidth_usage=bandwidth_usage,
            cache_hits=cache_hits,
            activation_memory=activation_memory
        )
    
    def _calculate_cost_metrics(
        self,
        compute_metrics: ComputeMetrics,
        latency: float,
        memory_footprint: Dict[str, float],
        power_watts: float
    ) -> CostMetrics:
        """Calculate cost-related metrics"""
        time_hours = latency / (1000 * 3600)  # ms to hours
        
        # GPU time cost
        gpu_time_cost = time_hours * self.cost_config['gpu_hour_rate']
        
        # Memory cost
        memory_gb = memory_footprint['gpu_allocated'] / 1024
        memory_cost = memory_gb * time_hours * self.cost_config['memory_gb_hour_rate']
        
        # Operation cost
        million_flops = compute_metrics.flops / 1e6
        operation_cost = million_flops * self.cost_config['operation_rate']
        
        # Total cost per inference
        inference_cost = gpu_time_cost + memory_cost + operation_cost
        
        return CostMetrics(
            inference_cost_usd=inference_cost,
            gpu_time_cost=gpu_time_cost,
            memory_cost=memory_cost,
            total_operation_cost=operation_cost
        )
    
    def evaluate_model(self, model: torch.nn.Module, tokenizer, verbose: bool = False) -> ModelMetrics:
        """Comprehensive model evaluation with detailed logging"""
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
            
            # Create environmental metrics
            environmental_metrics = EnvironmentalMetrics(
                co2_emissions=energy_metrics['co2_grams'],
                energy_consumption=energy_metrics['energy_joules'],
                power_usage=energy_metrics['power_watts']
            )
            
            # Calculate cost metrics
            cost_metrics = self._calculate_cost_metrics(
                compute_metrics=compute_metrics,
                latency=latency,
                memory_footprint=memory_stats,
                power_watts=energy_metrics['power_watts']
            )
            
            # Create final metrics object
            metrics = ModelMetrics(
                accuracy=accuracy,
                latency=latency,
                throughput=throughput,
                memory_footprint=memory_stats,
                parameter_count=compute_metrics.parameter_count,
                compute_metrics=compute_metrics,
                environmental_metrics=environmental_metrics,
                cost_metrics=cost_metrics
            )
            
            if verbose:
               self._log_metrics(metrics)
            
            # Log to wandb if enabled
            if self.use_wandb:
                wandb.log({
                        'accuracy': accuracy,
                        'latency_ms': latency,
                        'throughput': throughput,
                        'flops': compute_metrics.flops,
                        'sparsity': compute_metrics.sparsity,
                        'power_watts': energy_metrics['power_watts'],
                        'co2_emissions': energy_metrics['co2_grams'],
                        'cost_per_inference': cost_metrics.inference_cost_usd,
                        'gpu_memory_mb': memory_stats['gpu_allocated']
                    })
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def save_metrics(self, metrics: ModelMetrics, filename: str):
        """Save metrics to file"""
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'latency_ms': metrics.latency,
            'throughput_samples_per_sec': metrics.throughput,
            'memory_footprint_mb': metrics.memory_footprint,
            'parameter_count': metrics.parameter_count,
            'compute_metrics': vars(metrics.compute_metrics),
            'environmental_metrics': vars(metrics.environmental_metrics),
            'cost_metrics': vars(metrics.cost_metrics)
        }
        
        save_path = self.metrics_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics_dict)
    
    def load_metrics(self, filename: str) -> ModelMetrics:
        """Load metrics from file"""
        metrics_path = self.metrics_dir / filename
        try:
            with open(metrics_path) as f:
                metrics_dict = json.load(f)
            
            compute_metrics = ComputeMetrics(**metrics_dict['compute_metrics'])
            environmental_metrics = EnvironmentalMetrics(**metrics_dict['environmental_metrics'])
            cost_metrics = CostMetrics(**metrics_dict['cost_metrics'])
            
            return ModelMetrics(
                accuracy=metrics_dict['accuracy'],
                latency=metrics_dict['latency_ms'],
                throughput=metrics_dict['throughput_samples_per_sec'],
                memory_footprint=metrics_dict['memory_footprint_mb'],
                parameter_count=metrics_dict['parameter_count'],
                compute_metrics=compute_metrics,
                environmental_metrics=environmental_metrics,
                cost_metrics=cost_metrics
            )
            
        except Exception as e:
            logger.error(f"Failed to load metrics from {metrics_path}: {str(e)}")
            raise

    def _log_metrics(self, metrics: ModelMetrics, prefix: str = ""):
        """Log all available metrics in a structured format"""
        logger.info(f"\n{'='*20} {prefix} Model Metrics {'='*20}")
        
        # Basic Performance Metrics
        logger.info("\n=== Basic Performance ===")
        logger.info(f"Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"Latency: {metrics.latency:.2f} ms")
        logger.info(f"Throughput: {metrics.throughput:.2f} samples/second")
        
        # Memory Metrics
        logger.info("\n=== Memory Usage ===")
        for key, value in metrics.memory_footprint.items():
            logger.info(f"{key}: {value:.2f} MB")
        
        # Compute Metrics
        logger.info("\n=== Compute Statistics ===")
        compute = metrics.compute_metrics
        logger.info(f"Total FLOPs: {compute.flops/1e9:.2f}G")
        logger.info(f"MACs: {compute.macs/1e9:.2f}G")
        logger.info(f"Total Parameters: {compute.parameter_count:,}")
        logger.info(f"Active Parameters: {compute.active_parameter_count:,}")
        logger.info(f"Model Sparsity: {compute.sparsity*100:.2f}%")
        logger.info(f"Memory Bandwidth Usage: {compute.bandwidth_usage:.2f} GB/s")
        logger.info(f"Cache Hit Ratio: {compute.cache_hits*100:.2f}%")
        logger.info(f"Peak Activation Memory: {compute.activation_memory:.2f} MB")
        
        # Environmental Impact
        logger.info("\n=== Environmental Impact ===")
        env = metrics.environmental_metrics
        logger.info(f"CO2 Emissions: {env.co2_emissions:.4f}g")
        logger.info(f"Energy Consumption: {env.energy_consumption:.2f} joules")
        logger.info(f"Power Usage: {env.power_usage:.2f} watts")
        
        # Cost Analysis
        logger.info("\n=== Cost Analysis ===")
        cost = metrics.cost_metrics
        logger.info(f"Cost per Inference: ${cost.inference_cost_usd:.6f}")
        logger.info(f"GPU Time Cost: ${cost.gpu_time_cost:.6f}")
        logger.info(f"Memory Cost: ${cost.memory_cost:.6f}")
        logger.info(f"Operation Cost: ${cost.total_operation_cost:.6f}")
        
        logger.info("\n" + "="*60)

