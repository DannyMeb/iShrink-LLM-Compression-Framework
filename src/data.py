<<<<<<< HEAD
from typing import Dict, Any, Tuple, Optional, List
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
import logging
import numpy as np
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class MMULExample:
    """Single MMLU example structure"""
    question: str
    choices: List[str]
    answer: int
    subject: str

class MMLUDataset(Dataset):
    """Dataset for MMLU evaluation"""
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        split: str = "validation",
        config: str = "all",
        num_shots: int = 5,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize MMLU dataset
        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            split: Dataset split ('validation', 'dev', 'test')
            config: MMLU configuration/subject
            num_shots: Number of few-shot examples
            cache_dir: Optional directory for caching dataset
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.num_shots = num_shots
        self.idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        try:
            logger.info(f"Loading MMLU dataset with config '{config}' and split '{split}'")
            
            # Load evaluation dataset
            self.eval_dataset = load_dataset(
                "cais/mmlu",
                config,
                split=split,
                cache_dir=cache_dir
            )
            
            # Load dev set for few-shot examples
            if config == "all":
                # Load dev sets for each subject
                subjects = self._get_available_subjects()
                dev_datasets = []
                for subject in tqdm(subjects, desc="Loading dev sets"):
                    try:
                        dev_data = load_dataset(
                            "cais/mmlu",
                            subject,
                            split="dev",
                            cache_dir=cache_dir
                        )
                        dev_datasets.append(dev_data)
                    except Exception as e:
                        logger.warning(f"Could not load dev data for {subject}: {e}")
                        continue
                
                if not dev_datasets:
                    raise RuntimeError("Could not load any dev sets")
                
                self.dev_dataset = concatenate_datasets(dev_datasets)
            else:
                # Load dev set for specific subject
                self.dev_dataset = load_dataset(
                    "cais/mmlu",
                    config,
                    split="dev",
                    cache_dir=cache_dir
                )
            
            logger.info(f"Successfully loaded {len(self.eval_dataset)} evaluation examples")
            logger.info(f"Successfully loaded {len(self.dev_dataset)} dev examples for few-shot prompting")
            
            # Log first few answers for verification
            first_answers = [item['answer'] for item in list(self.eval_dataset)[:5]]
            logger.info(f"First few answers in dataset: {first_answers}")
            
            # Validate dataset structure
            self._validate_dataset()
            
        except Exception as e:
            logger.error(f"Failed to load MMLU dataset: {str(e)}")
            raise
    
    def _get_available_subjects(self) -> List[str]:
        """Get list of available MMLU subjects"""
        try:
            load_dataset("cais/mmlu")
        except ValueError as e:
            error_msg = str(e)
            start_idx = error_msg.find('[')
            end_idx = error_msg.find(']')
            if start_idx == -1 or end_idx == -1:
                raise RuntimeError("Could not parse available subjects")
            
            subjects = eval(error_msg[start_idx:end_idx+1])
            return [s for s in subjects if s != "all"]
        return []

    def _get_few_shot_examples(self, subject: str) -> List[Dict]:
        """Get few-shot examples for a given subject"""
        # Filter examples by subject
        subject_examples = [
            ex for ex in self.dev_dataset 
            if ex.get('subject', '') == subject
        ]
        
        if len(subject_examples) >= self.num_shots:
            # If we have enough examples from same subject, use those
            return random.sample(subject_examples, self.num_shots)
        
        # Otherwise, use all available subject examples and fill rest randomly
        examples = subject_examples.copy()
        remaining = self.num_shots - len(examples)
        if remaining > 0:
            other_examples = [
                ex for ex in self.dev_dataset 
                if ex.get('subject', '') != subject
            ]
            if other_examples:
                examples.extend(random.sample(other_examples, remaining))
        
        return examples[:self.num_shots]  # Ensure we return exactly num_shots examples

    def _format_few_shot_prompt(self, few_shot_examples: List[Dict], subject: str) -> str:
        """Format few-shot examples into a prompt"""
        prompt = (
            f"You are an expert in {subject}. For each multiple choice question, carefully analyze each option "
            f"and select the most accurate answer. First, let's look at some example questions and their answers:\n\n"
        )
        
        # Add few-shot examples with reasoning
        for i, example in enumerate(few_shot_examples, 1):
            prompt += f"Question {i}:\n{example['question']}\n"
            for idx, choice in enumerate(example['choices']):
                prompt += f"{self.idx_to_letter[idx]}: {choice}\n"
            prompt += (
                f"Let's think about this step by step:\n"
                f"1. First, understand what the question is asking about {subject}\n"
                f"2. Analyze each option:\n"
            )
            for idx, choice in enumerate(example['choices']):
                prompt += f"   - Option {self.idx_to_letter[idx]}: {choice}\n"
            prompt += f"3. The correct answer is {self.idx_to_letter[example['answer']]} for this {subject} question\n\n"
        
        prompt += (
            f"Now, let's solve the following {subject} question using the same careful approach. "
            f"Remember to consider each option thoroughly before selecting the answer.\n\n"
        )
        return prompt

    def __len__(self) -> int:
        return len(self.eval_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from dataset"""
        try:
            item = self.eval_dataset[idx]
            subject = item.get('subject', 'general knowledge')
            
            # Get few-shot examples
            few_shot_examples = self._get_few_shot_examples(subject)
            
            # Format prompt with few-shot examples
            prompt = self._format_few_shot_prompt(few_shot_examples, subject)
            
            # Add the actual question with structured thinking prompt
            prompt += f"Question:\n{item['question']}\n"
            for idx, choice in enumerate(item['choices']):
                prompt += f"{self.idx_to_letter[idx]}: {choice}\n"
            prompt += (
                "Let's solve this step by step:\n"
                "1. First, understand what the question is asking\n"
                "2. Analyze each option carefully:\n"
            )
            for idx, choice in enumerate(item['choices']):
                prompt += f"   - Option {self.idx_to_letter[idx]}: {choice}\n"
            prompt += "3. Based on the analysis, the answer is"
            
            # Log examples for inspection
            if idx < 2:
                logger.info(f"\nExample {idx} prompt:")
                logger.info(prompt)
                logger.info(f"Correct answer: {self.idx_to_letter[item['answer']]} (index {item['answer']})")
            
            # Tokenize
            encoding = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(item['answer'], dtype=torch.long),
                'subject': subject
            }
            
        except Exception as e:
            logger.error(f"Error processing example {idx}: {str(e)}")
            raise
    
    def _validate_dataset(self):
        """Validate dataset structure and content"""
        if len(self.eval_dataset) == 0:
            raise ValueError("Dataset is empty")
        
        # Check first example for required fields
        example = self.eval_dataset[0]
        required_fields = ['question', 'choices', 'answer']
        for field in required_fields:
            if field not in example:
                raise ValueError(f"Dataset missing required field: {field}")
        
        # Validate answer format
        invalid_answers = [
            ans for ans in self.eval_dataset['answer']
            if not isinstance(ans, int) or ans not in range(4)
        ]
        if invalid_answers:
            raise ValueError(f"Dataset contains invalid answers: {set(invalid_answers)}")
    
    def get_subject_distribution(self) -> Dict[str, int]:
        """Get distribution of subjects in dataset"""
        subjects = {}
        for item in self.eval_dataset:
            subject = item.get('subject', 'unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
        return subjects

def create_mmlu_dataloader(
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any],
    split: str = "validation",
    cache_dir: Optional[str] = None
=======
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
>>>>>>> origin/main
) -> Tuple[DataLoader, Dataset]:
    """Create MMLU dataloader"""
    
    try:
<<<<<<< HEAD
        # Create dataset
        dataset = MMLUDataset(
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_length'],
            split=split,
            config=config['training']['data']['dataset_config'],
            num_shots=5,  # Fixed to 5 shots to match baseline
            cache_dir=cache_dir
        )
        
        # Log subject distribution
        subject_dist = dataset.get_subject_distribution()
        if len(subject_dist) > 1:
            logger.info("Subject distribution:")
            for subject, count in sorted(subject_dist.items()):
                logger.info(f"  {subject}: {count} examples")
        
        # Create dataloader
=======
        dataset = MMLUDataset(
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_length'],
            split=split
        )
        
>>>>>>> origin/main
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['data']['batch_size'],
            shuffle=(split == "train"),
            num_workers=config['system']['num_workers'],
<<<<<<< HEAD
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
        
        logger.info(
            f"Created MMLU dataloader for split '{split}' with "
            f"{len(dataset)} examples and batch size {dataloader.batch_size}"
        )
        
=======
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"Created MMLU dataloader for split '{split}' with {len(dataset)} examples")
>>>>>>> origin/main
        return dataloader, dataset
        
    except Exception as e:
        logger.error(f"Error creating MMLU dataloader: {str(e)}")
        raise