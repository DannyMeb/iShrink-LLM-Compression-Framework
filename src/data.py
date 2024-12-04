#data.py

from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging
import numpy as np
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class MMULExample:
    """Single MMLU example structure"""
    question: str
    choices: List[str]
    answer: str
    subject: str

class MMLUDataset(Dataset):
    """Dataset for MMLU evaluation"""
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        split: str = "validation",
        config: str = "all",
        cache_dir: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Map for both numeric and letter answers
        self.answer_map = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3,
            '0': 0, '1': 1, '2': 2, '3': 3,
            0: 0, 1: 1, 2: 2, 3: 3
        }
        self.idx_to_answer = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        try:
            logger.info(f"Loading MMLU dataset with config '{config}' and split '{split}'")
            self.dataset = load_dataset(
                "cais/mmlu",
                config,
                split=split,
                cache_dir=cache_dir
            )
            logger.info(f"Successfully loaded {len(self.dataset)} examples")
            
            # Validate dataset structure
            self._validate_dataset()
            
        except Exception as e:
            logger.error(f"Failed to load MMLU dataset: {str(e)}")
            raise RuntimeError(f"Dataset loading failed: {str(e)}")
    
    def _validate_dataset(self):
        """Validate dataset structure and content"""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Check first example for required fields
        example = self.dataset[0]
        required_fields = ['question', 'choices', 'answer']
        for field in required_fields:
            if field not in example:
                raise ValueError(f"Dataset missing required field: {field}")
        
        # Log the first few answers for debugging
        first_answers = list(self.dataset['answer'][:5])
        logger.info(f"First few answers in dataset: {first_answers}")
        
        # Validate that all answers can be mapped
        invalid_answers = []
        for ans in self.dataset['answer']:
            if not isinstance(ans, (int, str)) or ans not in self.answer_map:
                invalid_answers.append(ans)
        
        if invalid_answers:
            raise ValueError(
                f"Dataset contains invalid answers: {set(invalid_answers)}"
            )
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from dataset"""
        try:
            item = self.dataset[idx]
            
            # Format input text
            formatted_input = self._format_example(item)
            
            # Tokenize
            encoding = self.tokenizer(
                formatted_input,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get answer index directly from answer_map
            answer = item['answer']
            if isinstance(answer, str) and answer.isdigit():
                answer = int(answer)
            answer_idx = self.answer_map[answer]
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(answer_idx, dtype=torch.long),
                'subject': item.get('subject', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
            raise
    
    def _format_example(self, item: Dict) -> str:
        """Format example for model input"""
        formatted_input = f"Question: {item['question']}\n"
        for letter, choice in zip(['A', 'B', 'C', 'D'], item['choices']):
            formatted_input += f"{letter}: {choice}\n"
        return formatted_input
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get distribution of subjects in dataset"""
        subjects = {}
        for item in self.dataset:
            subject = item.get('subject', 'unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
        return subjects

def create_mmlu_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    split: str = "validation",
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, Dataset]:
    """
    Create MMLU dataloader
    
    Args:
        tokenizer: Tokenizer for text processing
        config: Configuration dictionary
        split: Dataset split
        cache_dir: Optional cache directory
    
    Returns:
        Tuple of (DataLoader, Dataset)
    """
    try:
        # Create dataset
        dataset = MMLUDataset(
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_length'],
            split=split,
            config=config['training']['data']['dataset_config'],
            cache_dir=cache_dir
        )
        
        # Log subject distribution if available
        # subject_dist = dataset.get_subject_distribution()
        # if len(subject_dist) > 1:  # Only log if multiple subjects
        #     # logger.info("Subject distribution:")
        #     for subject, count in subject_dist.items():
        #         logger.info(f"  {subject}: {count} examples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['data']['batch_size'],
            shuffle=(split == "train"),
            num_workers=config['system']['num_workers'],
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        logger.info(
            f"Created MMLU dataloader for split '{split}' with "
            f"{len(dataset)} examples and batch size {dataloader.batch_size}"
        )
        
        return dataloader, dataset
        
    except Exception as e:
        logger.error(f"Error creating MMLU dataloader: {str(e)}")
        raise

def get_available_subjects() -> List[str]:
    """Get list of available MMLU subjects"""
    try:
        # This will raise an error that contains the list of configs
        load_dataset("cais/mmlu")
    except ValueError as e:
        # Extract subjects from error message
        error_msg = str(e)
        start_idx = error_msg.find('[')
        end_idx = error_msg.find(']')
        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Could not parse available subjects")
        
        subjects = eval(error_msg[start_idx:end_idx+1])
        return subjects
    
    return []  # Should never reach here

def validate_subject(subject: str) -> bool:
    """Validate if a subject is available in MMLU"""
    available_subjects = get_available_subjects()
    return subject in available_subjects