from pathlib import Path
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Tuple, Optional, Dict, Any
import logging
from .utils import ModelConfig, calculate_model_size, setup_device

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
        
    def load(self) -> Tuple[LlamaForCausalLM, LlamaTokenizer, Dict[str, Any]]:
        """Load model and return with metadata"""
        if self._check_local_model():
            model, tokenizer = self._load_local()
        else:
            model, tokenizer = self._download_and_save()
        
        # Calculate model statistics
        num_params, memory_size = calculate_model_size(model)
        
        metadata = {
            'num_params': num_params,
            'memory_size': memory_size,
            'device': self.device,
            'dtype': self.dtype,
            'model_name': self.config.name
        }
        
        logger.info(f"Loaded model with {num_params:,} parameters, "
                   f"using {memory_size:.2f}MB memory")
        
        return model, tokenizer, metadata

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