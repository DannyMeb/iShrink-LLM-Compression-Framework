import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
import logging
import yaml
from torch.utils.data import Dataset, DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Shared model configuration"""
    name: str
    local_path: Optional[str]
    device: str
    precision: str
    batch_size: int
    max_seq_length: int

@dataclass
class PruningConfig:
    """Shared pruning configuration"""
    min_accuracy: float
    target_memory: float
    target_latency: float
    dependency_config: Dict[str, Any]
    importance_config: Dict[str, Any]

@dataclass
class TrainingConfig:
    """Shared training configuration"""
    num_episodes: int
    batch_size: int
    checkpoint_freq: int
    eval_freq: int
    save_dir: Path

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Convert paths to Path objects
    config['system']['save_dir'] = Path(config['system']['save_dir'])
    if config['model']['local_path']:
        config['model']['local_path'] = Path(config['model']['local_path'])
    
    return config

def setup_device(device_name: str) -> torch.device:
    """Setup computation device"""
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class PruningDataset(Dataset):
    """Dataset class for pruning"""
    def __init__(self, 
                 data: torch.Tensor,
                 labels: Optional[torch.Tensor] = None,
                 tokenizer=None):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.labels is not None:
            return item, self.labels[idx]
        return item

def create_dataloader(dataset: PruningDataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 4) -> DataLoader:
    """Create dataloader with consistent settings"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def save_checkpoint(state: Dict[str, Any], path: Path, filename: str):
    """Save checkpoint with consistent format"""
    save_path = path / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load checkpoint with consistent format"""
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    return torch.load(path, map_location=device)

def calculate_model_size(model: torch.nn.Module) -> Tuple[int, float]:
    """Calculate model size in parameters and memory"""
    num_params = sum(p.numel() for p in model.parameters())
    memory_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
    return num_params, memory_size

def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)