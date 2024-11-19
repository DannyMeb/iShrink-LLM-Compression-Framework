from pathlib import Path
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Tuple, Optional, Dict, Any
import logging
from .utils import ModelConfig, calculate_model_size, setup_device, MemoryManager

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, 
                 config: ModelConfig,
                 hf_token: Optional[str] = None):
        """
        Initialize model loader with configuration
        """
        self.config = config
        self.device = setup_device(config.device)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        if not self.hf_token:
            raise ValueError("HuggingFace token required. Set HF_TOKEN environment variable or pass directly.")
        
        # Set precision
        self.dtype = torch.float16 if config.precision == "float16" else torch.float32
        
    def load(self) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        """Load model with memory optimizations"""
        try:
            MemoryManager.clear_cache()
            
            model = LlamaForCausalLM.from_pretrained(
                self.config.name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                gradient_checkpointing=True,
                device_map='auto',  # Automatically handle memory mapping
                offload_folder="offload"  # Temporary storage for weight offloading
            )
            
            tokenizer = LlamaTokenizer.from_pretrained(self.config.name)
            
            logger.info(f"Model loaded. Memory usage: {MemoryManager.get_memory_stats()}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _check_local_model(self) -> bool:
        """Check if model exists locally"""
        if not self.config.local_path:
            return False
            
        paths = [
            self.config.local_path / "config.json",
            self.config.local_path / "pytorch_model.bin",
            self.config.local_path / "tokenizer.json"
        ]
        return all(p.exists() for p in paths)

    def _load_local(self) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        """Load model from local storage"""
        logger.info(f"Loading model from {self.config.local_path}")
        
        try:
            model = LlamaForCausalLM.from_pretrained(
                self.config.local_path,
                device_map=self.device,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            )
            tokenizer = LlamaTokenizer.from_pretrained(self.config.local_path)
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            logger.info("Attempting to download from HuggingFace...")
            return self._download_and_save()

    def _download_and_save(self) -> Tuple[LlamaForCausalLM, LlamaTokenizer]:
        """Download model and save locally"""
        logger.info(f"Downloading model {self.config.name}")
        
        try:
            model = LlamaForCausalLM.from_pretrained(
                self.config.name,
                token=self.hf_token,
                device_map=self.device,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                self.config.name,
                token=self.hf_token
            )
            
            if self.config.local_path:
                logger.info(f"Saving model to {self.config.local_path}")
                self.config.local_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(self.config.local_path)
                tokenizer.save_pretrained(self.config.local_path)
            
            return model, tokenizer
            
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")