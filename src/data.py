from typing import Dict, Any, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MMLUDataset(Dataset):
    """Dataset for MMLU evaluation"""
    def __init__(self, tokenizer, max_length: int, split: str = "validation"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load MMLU dataset
        self.dataset = load_dataset("cais/mmlu", split=split)
        
        # Map answer letters to indices
        self.answer_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get question and choices
        question = item['question']
        choices = [item['choices'][i] for i in range(4)]  # Always 4 choices in MMLU
        
        # Format as: "Question: {question}\nA: {choice_a}\nB: {choice_b}\nC: {choice_c}\nD: {choice_d}"
        formatted_input = f"Question: {question}\n"
        for letter, choice in zip(['A', 'B', 'C', 'D'], choices):
            formatted_input += f"{letter}: {choice}\n"
        
        # Tokenize
        encodings = self.tokenizer(
            formatted_input,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        label = self.answer_map[item['answer']]
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_mmlu_dataloader(
    tokenizer,
    config: Dict[str, Any],
    split: str = "validation"
) -> Tuple[DataLoader, Dataset]:
    """Create MMLU dataloader"""
    
    try:
        dataset = MMLUDataset(
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_length'],
            split=split
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['data']['batch_size'],
            shuffle=(split == "train"),
            num_workers=config['system']['num_workers'],
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created MMLU dataloader for split '{split}' with {len(dataset)} examples")
        return dataloader, dataset
        
    except Exception as e:
        logger.error(f"Error creating MMLU dataloader: {str(e)}")
        raise