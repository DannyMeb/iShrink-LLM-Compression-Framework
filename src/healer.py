import os
import sys
import logging
import math
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import datasets
import evaluate
import torch
import torch.nn.functional as F
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the model to finetune"}
    )
    config_name: str = field(
        default="decapoda-research/llama-7b-hf",
        metadata={"help": "Pretrained config name or path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for models and datasets"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "Alpha parameter for LoRA scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout probability for LoRA layers"}
    )

@dataclass
class DataTrainingArguments:
    """Arguments for training data"""
    dataset_name: str = field(
        default="allenai/c4",
        metadata={"help": "The name of the dataset to use"}
    )
    dataset_config_name: str = field(
        default="allenai--c4",
        metadata={"help": "The configuration name of the dataset"}
    )
    max_train_samples: Optional[int] = field(
        default=30000,
        metadata={"help": "Maximum number of training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum number of evaluation samples"}
    )
    block_size: Optional[int] = field(
        default=1024,
        metadata={"help": "Block size for training"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "Number of preprocessing workers"}
    )

def load_config(config_path: str):
    """Load configuration file"""
    config_path = PROJECT_ROOT / config_path
    logger.info(f"Loading config from: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_model_choice():
    """Prompt user for model choice"""
    while True:
        choice = input("\nWhich model would you like to finetune?\n1. Final pruned model\n2. Sparsified model\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")

def get_sparsity_level():
    """Prompt user for sparsity level"""
    sparsity_levels = [5, 15, 25, 35, 45, 50]
    while True:
        print("\nAvailable sparsity levels:", ", ".join(f"{x}%" for x in sparsity_levels))
        choice = input("Enter sparsity percentage: ").strip().replace('%', '')
        try:
            level = int(choice)
            if level in sparsity_levels:
                return level
            print("Invalid sparsity level. Please choose from the available levels.")
        except ValueError:
            print("Please enter a valid number.")

def get_model_path(config, model_choice):
    """Get path to model based on user choice"""
    save_dir = PROJECT_ROOT / config['system']['save_dir']
    
    if model_choice == 1:
        model_path = save_dir / 'final_model'
        model_type = "pruned"
        output_dir = save_dir / 'pruned_finetuned_model'
    else:
        sparsity = get_sparsity_level()
        model_path_1 = save_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
        model_path_2 = save_dir / f"sparsified_model_at_{sparsity}%"
        
        if model_path_1.exists():
            model_path = model_path_1
        elif model_path_2.exists():
            model_path = model_path_2
        else:
            available_models = []
            for path in [save_dir, save_dir / 'sparsified_models']:
                if path.exists():
                    available_models.extend(p.name for p in path.iterdir() 
                                         if p.is_dir() and 'sparsified_model_at_' in p.name)
            
            if available_models:
                logger.error(f"Available sparsified models: {', '.join(available_models)}")
            raise FileNotFoundError(f"Sparsified model at {sparsity}% not found. Checked paths:\n"
                                 f"- {model_path_1}\n"
                                 f"- {model_path_2}")
        
        model_type = f"sparsified_{sparsity}"
        output_dir = save_dir / f"sparsified_{sparsity}%_finetuned_model"
    
    logger.info(f"Using model from: {model_path}")
    logger.info(f"Will save finetuned model to: {output_dir}")
    return model_path, output_dir, model_type

def preprocess_dataset(dataset, tokenizer, block_size, num_workers):
    """Tokenize and chunk dataset"""
    # Tokenize
    tokenize_function = lambda examples: tokenizer(
        examples["text"],
        truncation=True,
        max_length=block_size,
        padding="max_length",
    )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset

def setup_training_arguments(training_args, output_dir):
    """Set default training arguments if not provided"""
    if training_args.per_device_train_batch_size is None:
        training_args.per_device_train_batch_size = 8
    if training_args.gradient_accumulation_steps is None:
        training_args.gradient_accumulation_steps = 4
    if training_args.warmup_steps is None:
        training_args.warmup_steps = 50
    if training_args.logging_steps is None:
        training_args.logging_steps = 10
    if training_args.save_steps is None:
        training_args.save_steps = 100
    if training_args.eval_steps is None:
        training_args.eval_steps = 100
    
    training_args.output_dir = str(output_dir)
    training_args.fp16 = True
    training_args.gradient_checkpointing = True
    training_args.evaluation_strategy = "steps"
    training_args.save_strategy = "steps"
    training_args.save_total_limit = 3
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "loss"
    training_args.greater_is_better = False
    
    return training_args

def get_compute_metrics():
    """Setup metrics computation"""
    metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        
        if isinstance(logits, tuple):
            logits = logits[0]
            
        predictions = logits.argmax(-1)
        
        # Calculate perplexity
        loss = F.cross_entropy(
            torch.from_numpy(logits.reshape(-1, logits.shape[-1])),
            torch.from_numpy(labels.reshape(-1))
        )
        perplexity = math.exp(loss.item())
        
        # Shift predictions and labels for accuracy
        predictions = predictions[:, :-1].reshape(-1)
        labels = labels[:, 1:].reshape(-1)
        
        accuracy = metric.compute(predictions=predictions, references=labels)
        
        return {
            "accuracy": accuracy["accuracy"],
            "perplexity": perplexity
        }
    
    return compute_metrics

def setup_model_and_tokenizer(model_path, model_args, data_args):
    """Setup model and tokenizer with proper configuration"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=model_args.cache_dir,
        padding_side="right",
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir=model_args.cache_dir,
        device_map="auto"
    )
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging based on training args
    logger.setLevel(training_args.get_process_log_level())
    datasets.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Load config and get model paths
    config = load_config(model_args.config_path)
    model_choice = get_model_choice()
    model_path, output_dir, model_type = get_model_path(config, model_choice)
    
    # Setup default training arguments
    training_args = setup_training_arguments(training_args, output_dir)
    
    # Load tokenizer and model
    logger.info(f"Loading model from {model_path}")
    tokenizer, model = setup_model_and_tokenizer(model_path, model_args, data_args)

    # Setup LoRA
    target_modules = ["gate_proj", "up_proj", "down_proj"]  # Focus on MLP layers
    logger.info(f"Setting up LoRA with target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading C4 dataset...")
    raw_datasets = load_dataset(
        'allenai/c4', 
        'allenai--c4', 
        data_files={
            'train': 'en/c4-train.00000-of-01024.json.gz',
            'validation': 'en/c4-validation.00000-of-00008.json.gz'
        },
        cache_dir=model_args.cache_dir
    )

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    tokenized_datasets = {}
    for split in ['train', 'validation']:
        tokenized_datasets[split] = preprocess_dataset(
            raw_datasets[split],
            tokenizer,
            data_args.block_size,
            data_args.preprocessing_num_workers
        )

    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=get_compute_metrics(),
    )

    # Prepare for training
    model.config.use_cache = False
    
    # Set up proper state dict handling for LoRA
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    # Optional model compilation
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Train
    logger.info("\nStarting training...")
    train_result = trainer.train()

    # Save final model and adapter
    logger.info(f"\nSaving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(trainer.model.state_dict(), os.path.join(output_dir, "adapter_model.bin"))

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Run final evaluation
    if trainer.is_world_process_zero():
        logger.info("\nRunning final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("final_eval", eval_metrics)
        
        logger.info("\nTraining complete!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Final training metrics: {metrics}")
        logger.info(f"Final evaluation metrics: {eval_metrics}")

if __name__ == "__main__":
    main()