import os
import sys
import logging
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

# Fixed paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

@dataclass
class ModelArguments:
    """Arguments for model configuration"""
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate for training"}
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    training_percentage: float = field(
        default=100.0,
        metadata={"help": "Percentage of training data to use (0-100)"}
    )
    max_train_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "Maximum number of training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=100,
        metadata={"help": "Maximum number of evaluation samples"}
    )
    block_size: int = field(
        default=2048,
        metadata={"help": "Size of the blocks the dataset is split into"}
    )
    batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU"}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation"}
    )

def load_config():
    """Load configuration file"""
    logger.info(f"Loading config from: {CONFIG_PATH}")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def enable_input_require_grads(model):
    """Enable gradient checkpointing with proper hooks"""
    def make_inputs_require_grads(module, input, output):
        output.requires_grad_(True)

    model._require_grads_hook = model.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

def get_model_choice():
    """Prompt user to choose between pruned or sparsified model"""
    while True:
        choice = input("\nWhich model would you like to finetune?\n1. Pruned model\n2. Sparsified model\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid choice. Please enter 1 or 2.")

def get_sparsity_level():
    """Prompt user for sparsity level if sparsified model is chosen"""
    sparsity_levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
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

def get_model_path():
    """Get model path based on user choice"""
    config = load_config()
    model_name = config['model']['name'].split('/')[-1]
    model_base_dir = PROJECT_ROOT / "experiments" / "results" / model_name
    
    choice = get_model_choice()
    
    if choice == 1:
        model_path = model_base_dir / 'pruned_model'
        output_dir = model_base_dir / 'finetuned_models' / 'finetuned_pruned_model'
    else:
        sparsity = get_sparsity_level()
        model_path = model_base_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
        output_dir = model_base_dir / 'finetuned_models' / f"sparsified_model_at_{sparsity}%_finetuned"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    logger.info(f"Using model from: {model_path}")
    logger.info(f"Will save finetuned model to: {output_dir}")
    
    return model_path, output_dir

def format_alpaca_prompt(example: Dict) -> str:
    """Format the Alpaca prompt according to the standard template"""
    if example.get("input"):
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

def process_dataset(tokenizer, args):
    logger.info("Loading Alpaca-cleaned dataset...")
    try:
        dataset = load_dataset("yahma/alpaca-cleaned")
        
        # Split the dataset into train and validation
        split_dataset = dataset["train"].train_test_split(
            test_size=0.1, 
            seed=42
        )
        
        def tokenize_function(examples):
            # Format prompts for all examples in the batch
            formatted_prompts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples['input'][i]
                output = examples['output'][i]
                
                if input_text and input_text.strip():
                    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                formatted_prompts.append(prompt)
            
            # Tokenize the formatted prompts
            tokenized = tokenizer(
                formatted_prompts,
                truncation=True,
                max_length=args.block_size,
                padding="max_length",
                return_tensors=None,
            )
            
            return tokenized
        
        logger.info("Processing training dataset...")
        if args.max_train_samples:
            train_dataset = split_dataset["train"].select(range(min(len(split_dataset["train"]), args.max_train_samples)))
        else:
            train_dataset = split_dataset["train"]
            
        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing training dataset",
        )
        
        # Add labels for training
        train_tokenized = train_tokenized.map(
            lambda examples: {"labels": examples["input_ids"]},
            batched=True,
            desc="Adding labels to training dataset",
        )
        
        eval_tokenized = None
        if args.do_eval:
            logger.info("Processing validation dataset...")
            if args.max_eval_samples:
                eval_dataset = split_dataset["test"].select(range(min(len(split_dataset["test"]), args.max_eval_samples)))
            else:
                eval_dataset = split_dataset["test"]
                
            eval_tokenized = eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing validation dataset",
            )
            
            # Add labels for evaluation
            eval_tokenized = eval_tokenized.map(
                lambda examples: {"labels": examples["input_ids"]},
                batched=True,
                desc="Adding labels to validation dataset",
            )

        return train_tokenized, eval_tokenized

    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def setup_model(model_path: Path, args: ModelArguments):
    logger.info("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='auto',
        use_safetensors=True,
        max_memory={0: "35GB"},
    )
    
    logger.info("Configuring LoRA...")
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    
    if model.config.use_cache:
        model.config.use_cache = False
        enable_input_require_grads(model)
    
    model.print_trainable_parameters()
    return model

def setup_training_args(args: ModelArguments, output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=100,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps" if args.do_eval else "no",
        save_strategy="steps",
        save_total_limit=10,
        load_best_model_at_end=args.do_eval,
        fp16=True,
        half_precision_backend="auto",
        optim="adamw_torch",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        group_by_length=True,
        report_to=["tensorboard"],
    )

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    logging.getLogger('codecarbon').setLevel(logging.WARNING)
    
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if model_args.training_percentage < 100:
        model_args.max_train_samples = int(model_args.max_train_samples * (model_args.training_percentage / 100))
        logger.info(f"Using {model_args.training_percentage}% of training data: {model_args.max_train_samples} samples")

    set_seed(42)

    model_path, output_dir = get_model_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint = None
    if output_dir.exists():
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint:
            logger.info(f"Found checkpoint at {last_checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = setup_model(model_path, model_args)
    train_dataset, eval_dataset = process_dataset(tokenizer, model_args)
    training_args = setup_training_args(model_args, output_dir)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    if model_args.do_eval:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()