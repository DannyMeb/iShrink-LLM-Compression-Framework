import torch
import time
import json
from typing import Dict, Optional
from dataclasses import dataclass
import logging
import psutil
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Stores comprehensive model metrics"""
    accuracy: float
    latency: float  # ms
    throughput: float  # samples/second
    memory_footprint: Dict[str, float]  # Memory stats in MB
    parameter_count: int

class MetricsTracker:
    def __init__(self,
                 save_dir: Path,
                 device: torch.device,
                 tokenizer,
                 config: Dict,
                 use_wandb: bool = False):
        """
        Initialize the metrics tracker.
        
        Args:
            save_dir: Directory to save metrics
            device: Torch device to use
            tokenizer: Model tokenizer
            config: Configuration dictionary
            use_wandb: Whether to log to Weights & Biases
        """
        self.save_dir = save_dir
        self.device = device
        self.tokenizer = tokenizer
        self.config = config
        self.use_wandb = use_wandb
        self.metrics_dir = save_dir / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(self, model: torch.nn.Module, tokenizer) -> ModelMetrics:
        """
        Comprehensive model evaluation including accuracy, performance and memory metrics.
        
        Args:
            model: The model to evaluate
            tokenizer: The tokenizer to use with the model
            
        Returns:
            ModelMetrics containing all evaluation results
        """
        try:
            # 1. Measure accuracy using lm_eval
            accuracy = self._measure_accuracy(model, tokenizer)
            
            # 2. Measure latency and throughput
            latency, throughput = self._measure_performance(model)
            
            # 3. Measure memory footprint
            memory_stats = self._measure_memory_usage(model)
            
            # 4. Count parameters
            param_count = sum(p.numel() for p in model.parameters())
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                latency=latency,
                throughput=throughput,
                memory_footprint=memory_stats,
                parameter_count=param_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise
     # def _measure_accuracy(self, model: torch.nn.Module) -> float:
    #     """
    #     Measure model accuracy using lm_eval framework for MMLU benchmark.
    #     Uses the locally saved model from ModelLoader.
    #     """
    #     try:
    #         # Create temporary path to save results
    #         output_path = self.save_dir / 'temp_eval_results'
    #         output_path.mkdir(exist_ok=True)
            
    #         # Use the local model path where ModelLoader saved the model
    #         model_path = str(Path(self.config['model']['local_path']))
            
    #         # Run lm_eval command
    #         cmd = [
    #             "python", "-m", "lm_eval",
    #             "--model", "hf",
    #             "--model_args", (
    #                 f"pretrained={model_path},"  # Use local path where model is saved
    #                 "trust_remote_code=True"
    #             ),
    #             "--tasks", "mmlu",
    #             "--num_fewshot", "5",
    #             "--device", str(self.device),
    #             "--batch_size", "8",
    #             "--output_path", str(output_path)
    #         ]
            
    #         # Run the command and capture output
    #         process = subprocess.Popen(
    #             cmd,
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True
    #         )
            
    #         # Get output and error streams
    #         stdout, stderr = process.communicate()
            
    #         # Log both stdout and stderr
    #         logger.info(f"lm_eval stdout:\n{stdout}")
    #         if stderr:
    #             logger.error(f"lm_eval stderr:\n{stderr}")
                
    #         if process.returncode != 0:
    #             raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)

    #         # Read results from the output file
    #         results_file = output_path / 'results.json'
    #         if results_file.exists():
    #             with open(results_file) as f:
    #                 results = json.load(f)
    #                 accuracy = results["results"]["mmlu"]["acc"]
    #                 logger.info(f"MMLU Evaluation accuracy: {accuracy:.4f}")
    #                 return accuracy
    #         else:
    #             raise FileNotFoundError(f"Results file not found at {results_file}")

    #     except Exception as e:
    #         logger.error(f"Error during lm_eval accuracy measurement: {str(e)}")
    #         if hasattr(e, 'stderr'):
    #             logger.error(f"Stderr: {e.stderr}")
    #         raise




    
    def _measure_accuracy(self, model: torch.nn.Module, tokenizer) -> float:
        """
        Measure model accuracy using lm_eval framework for MMLU benchmark.
        Uses direct Python API calls.
        """
        try:
            from lm_eval import evaluator, tasks
            from lm_eval.models.huggingface import HFLM

            # Configure model for evaluation
            model_args = {
                "pretrained": model,
                "tokenizer": tokenizer,
                "device": self.device,
                "batch_size": 8,
                "trust_remote_code": True
            }

            # Create model evaluator
            hf_model = HFLM(**model_args)

            # Run evaluation
            results = evaluator.simple_evaluate(
                model=hf_model,
                tasks=["mmlu"],
                num_fewshot=5,
                device=str(self.device)
            )

            # Log full results structure for debugging
            logger.info(f"Full results structure: {results}")

            # Extract accuracy from the appropriate key in results
            if "mmlu" in results["results"]:
                accuracy_dict = results["results"]["mmlu"]
                # Try common key names for accuracy
                for key in ["acc", "accuracy", "average_accuracy", "mean_accuracy"]:
                    if key in accuracy_dict:
                        accuracy = accuracy_dict[key]
                        break
            else:
                raise KeyError("Could not find MMLU results in evaluation output")

            logger.info(f"MMLU Evaluation accuracy: {accuracy:.4f}")
            return accuracy

        except Exception as e:
            logger.error(f"Error during lm_eval accuracy measurement: {str(e)}")
            if 'results' in locals():
                logger.error(f"Results structure: {results}")
            raise

    def _measure_performance(self, model: torch.nn.Module) -> tuple[float, float]:
        """
        Measure model latency and throughput using a representative sequence length.
        
        Args:
            model: The model to evaluate
            
        Returns:
            tuple: (latency in ms, throughput in samples/second)
        """
        model.eval()
        
        # Create a representative sample (typical sequence length for MMLU)
        sample_text = "Q: What is the capital of France?\nA: Let's approach this step by step:\n1) France is a country in Western Europe\n2) The capital city has been the same since 508 CE\n3) It is located in the northern part of the country\nTherefore, the capital of France is"
        
        # Tokenize with padding to max length
        inputs = self.tokenizer(
            sample_text,
            return_tensors="pt",
            padding='max_length',
            max_length=512,  # Typical context length
            truncation=True
        )
        
        # Move to device
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Warmup runs to ensure GPU is ready
        with torch.no_grad():
            for _ in range(3):
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        # Measure latency over multiple runs
        num_runs = 10
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        latency = (total_time / num_runs) * 1000  # Convert to ms
        throughput = num_runs / total_time  # samples/second
        
        # Log performance metrics
        logger.info(f"Average latency: {latency:.2f}ms")
        logger.info(f"Throughput: {throughput:.2f} samples/second")
        logger.info(f"Input sequence length: {input_ids.size(1)} tokens")
        
        return latency, throughput

    def _measure_memory_usage(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Measure model memory footprint.
        
        Args:
            model: The model to evaluate
            
        Returns:
            dict: Memory statistics in MB
        """
        memory_stats = {
            'gpu_allocated': torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
            'gpu_cached': torch.cuda.memory_reserved(self.device) / 1024**2,      # MB
            'cpu_memory': psutil.Process().memory_info().rss / 1024**2            # MB
        }
        return memory_stats

    def save_metrics(self, metrics: ModelMetrics, filename: str):
        """
        Save metrics to file.
        
        Args:
            metrics: ModelMetrics instance to save
            filename: Name of the output file
        """
        metrics_dict = {
            'accuracy': metrics.accuracy,
            'latency_ms': metrics.latency,
            'throughput_samples_per_sec': metrics.throughput,
            'memory_footprint_mb': metrics.memory_footprint,
            'parameter_count': metrics.parameter_count
        }
        
        save_path = self.metrics_dir / filename
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        logger.info(f"Saved metrics to {save_path}")
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics_dict)