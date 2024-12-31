import os
import sys
import logging
import torch
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
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
    def get_lowest_module(module):
        if len(list(module.children())) == 0:
            return module
        else:
            return get_lowest_module(list(module.children())[0])

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
    # Load config and extract model name
    config = load_config()
    model_name = config['model']['name'].split('/')[-1]
    model_base_dir = PROJECT_ROOT / "experiments" / "results" / model_name
    
    choice = get_model_choice()
    
    if choice == 1:
        # Load from pruned model directory
        model_path = model_base_dir / 'pruned_model'
        output_dir = model_base_dir / 'finetuned_models' / 'finetuned_pruned_model'
    else:
        # Load from sparsified model directory
        sparsity = get_sparsity_level()
        model_path = model_base_dir / 'sparsified_models' / f"sparsified_model_at_{sparsity}%"
        output_dir = model_base_dir / 'finetuned_models' / f"sparsified_model_at_{sparsity}%_finetuned"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    logger.info(f"Using model from: {model_path}")
    logger.info(f"Will save finetuned model to: {output_dir}")
    
    return model_path, output_dir

def process_dataset(tokenizer, args):
    logger.info("Loading C4 dataset...")
    try:
        dataset = load_dataset(
            'allenai/c4',
            'en',
            streaming=True,
            cache_dir='cache'
        ).shuffle(seed=42)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.block_size,
                padding="max_length",
                return_tensors=None,
            )

        logger.info("Processing training dataset...")
        train_processed = []
        chunk_size = 100
        max_train = args.max_train_samples or 1000

        for i in range(0, max_train, chunk_size):
            chunk = list(dataset["train"].shuffle(seed=42).take(min(chunk_size, max_train - i)))
            chunk_processed = [tokenize_function(example) for example in chunk]
            
            for example in chunk_processed:
                example["labels"] = example["input_ids"].copy()
            
            train_processed.extend(chunk_processed)
            torch.cuda.empty_cache()
            logger.info(f"Processed {len(train_processed)}/{max_train} training examples")

        eval_processed = None
        if args.do_eval:
            logger.info("Processing validation dataset...")
            eval_processed = []
            max_eval = args.max_eval_samples or 100

            for i in range(0, max_eval, chunk_size):
                chunk = list(dataset["validation"].take(min(chunk_size, max_eval - i)))
                chunk_processed = [tokenize_function(example) for example in chunk]
                
                for example in chunk_processed:
                    example["labels"] = example["input_ids"].copy()
                
                eval_processed.extend(chunk_processed)
                torch.cuda.empty_cache()
                logger.info(f"Processed {len(eval_processed)}/{max_eval} validation examples")

        return train_processed, eval_processed

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
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj","gate_proj", "up_proj", "down_proj"],  # "q_proj", "v_proj", "k_proj", "o_proj", q_proj", "v_proj", "k_proj", "o_proj",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, config)
    
    # Enable gradient checkpointing with proper hooks
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