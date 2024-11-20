from pathlib import Path
import os
import torch
from transformers import LlamaForCausalLM, AutoTokenizer 
from typing import Tuple, Optional, Dict, Any
import logging
from .utils import ModelConfig, calculate_model_size, setup_device, MemoryManager

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, 
                 config: Dict[str, Any],
                 hf_token: Optional[str] = None):
        """Initialize model loader with configuration"""
        self.config = config
        self.device = setup_device(config['device'])
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HuggingFace token required. Set HF_TOKEN environment variable or pass directly.")
        
        # Set precision
        self.dtype = torch.float16 if config['precision'] == "float16" else torch.float32
        
        # Create model directory if it doesn't exist
        if self.config.get('local_path'):
            self.local_path = Path(self.config['local_path'])
            self.local_path.mkdir(parents=True, exist_ok=True)
        
    def load(self) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
        """Load model from local path or download from HuggingFace"""
        try:
            # Check if model exists locally
            if self._check_local_model():
                logger.info("Loading model from local storage...")
                return self._load_local()
            
            # If not, download and save
            logger.info("Model not found locally. Downloading...")
            return self._download_and_save()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _check_local_model(self) -> bool:
        """Check if model exists locally"""
        if not self.config.get('local_path'):
            return False
        
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json"
        ]
        
        return all((self.local_path / file).exists() for file in required_files)

    def _load_local(self) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
        """Load model from local storage"""
        try:
            model = LlamaForCausalLM.from_pretrained(
                self.local_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map='auto'
            )
            
            if self.config.get('gradient_checkpointing', False):
                model.gradient_checkpointing_enable()
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.local_path,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info(f"Successfully loaded model from {self.local_path}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            logger.info("Attempting to download from HuggingFace...")
            return self._download_and_save()

    def _download_and_save(self) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
        """Download model and save locally"""
        try:
            logger.info(f"Downloading model {self.config['name']}...")
            
            # Download model
            model = LlamaForCausalLM.from_pretrained(
                self.config['name'],
                token=self.hf_token,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map='auto'
            )
            
            if self.config.get('gradient_checkpointing', False):
                model.gradient_checkpointing_enable()
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config['name'],
                token=self.hf_token,
                use_fast=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Save locally if path is specified
            if self.config.get('local_path'):
                logger.info(f"Saving model to {self.local_path}")
                model.save_pretrained(self.local_path)
                tokenizer.save_pretrained(self.local_path)
                logger.info("Model saved successfully")
            
            return model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    def save_model(self, 
                  model: LlamaForCausalLM, 
                  tokenizer: AutoTokenizer, 
                  save_path: Optional[str] = None) -> None:
        """Save model to specified path or default local path"""
        try:
            save_path = Path(save_path) if save_path else self.local_path
            save_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving model to {save_path}")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise