import os
import sys
import logging
import torch
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
    get_peft_model_state_dict,
)
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

# Fixed paths
BASE_DIR = Path(__file__).resolve().parent.parent / "experiments/results"
PRUNED_MODEL_PATH = BASE_DIR / "final_model"
SPARSIFIED_MODELS_DIR = BASE_DIR / "sparsified_models"
OUTPUT_DIR = Path("experiments/results/finetuned_models")

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
        default=30000,
        metadata={"help": "Maximum number of training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=64,  # Reduced from 128
        metadata={"help": "Maximum number of evaluation samples"}
    )
    block_size: int = field(
        default=2048,
        metadata={"help": "Size of the blocks the dataset is split into"}
    )
    batch_size: int = field(
        default=2,  # Reduced from 8
        metadata={"help": "Batch size per GPU"}
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation"}
    )

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
    choice = get_model_choice()
    
    if choice == 1:
        model_path = PRUNED_MODEL_PATH
        output_dir = OUTPUT_DIR / "finetuned_pruned_model"
    else:
        sparsity = get_sparsity_level()
        model_path = SPARSIFIED_MODELS_DIR / f"sparsified_model_at_{sparsity}%"
        output_dir = OUTPUT_DIR / f"sparsified_model_at_{sparsity}%_finetuned"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    logger.info(f"Using model from: {model_path}")
    logger.info(f"Will save finetuned model to: {output_dir}")
    
    return model_path, output_dir

def setup_training_args(args: ModelArguments, output_dir: Path) -> TrainingArguments:
    """Setup training arguments with memory optimizations"""
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=100,  # Increased warmup steps
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        evaluation_strategy="steps" if args.do_eval else "no",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=args.do_eval,
        fp16=True,
        half_precision_backend="auto",
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        remove_unused_columns=False,  # Added to prevent column removal
        ddp_find_unused_parameters=False,  # Optimize DDP training
    )

def process_dataset(dataset, tokenizer, max_samples, block_size):
    """Process dataset in chunks to manage memory"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_tensors=None,
        )

    processed = []
    chunk_size = 1000  # Process in smaller chunks
    
    for i in range(0, max_samples, chunk_size):
        chunk = list(dataset.take(min(chunk_size, max_samples - i)))
        chunk_processed = [tokenize_function(example) for example in chunk]
        
        # Add labels for causal language modeling
        for example in chunk_processed:
            example["labels"] = example["input_ids"].copy()
        
        processed.extend(chunk_processed)
        
        # Clear memory
        torch.cuda.empty_cache()
        
        logger.info(f"Processed {len(processed)}/{max_samples} examples")
    
    return processed

def main():
    # Set CUDA memory split size
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # Suppress codecarbon logs
    logging.getLogger('codecarbon').setLevel(logging.WARNING)
    
    # Parse arguments
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Calculate actual number of samples based on percentage
    if model_args.training_percentage < 100:
        model_args.max_train_samples = int(model_args.max_train_samples * (model_args.training_percentage / 100))
        logger.info(f"Using {model_args.training_percentage}% of training data: {model_args.max_train_samples} samples")

    # Set random seed
    set_seed(42)

    # Get model paths
    model_path, output_dir = get_model_path()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing checkpoints
    last_checkpoint = None
    if output_dir.exists():
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint:
            logger.info(f"Found checkpoint at {last_checkpoint}")

    # Load tokenizer and model
    logger.info("Loading tokenizer and model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map='auto',
            use_safetensors=True,
            max_memory={0: "35GB"},  # Limit GPU memory usage
        )

        # Ensure model parameters are set up for training
        model.enable_input_require_grads()
        
        logger.info(f"Successfully loaded model from {model_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

    # Setup LoRA with modified configuration
    logger.info("Setting up LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    # Create PEFT model
    model = get_peft_model(model, lora_config)
    
    # Ensure all trainable parameters require gradients
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        'allenai/c4', 
        'en',
        streaming=True,
        cache_dir='cache'
    )

    # Process datasets
    logger.info("Processing training dataset...")
    train_dataset = process_dataset(
        dataset["train"],
        tokenizer,
        model_args.max_train_samples,
        model_args.block_size
    )

    if model_args.do_eval:
        logger.info("Processing validation dataset...")
        validation_dataset = process_dataset(
            dataset["validation"],
            tokenizer,
            model_args.max_eval_samples,
            model_args.block_size
        )
    else:
        validation_dataset = None

    logger.info(f"Loaded {len(train_dataset)} training examples")
    if validation_dataset:
        logger.info(f"Loaded {len(validation_dataset)} validation examples")

    # Setup training arguments and trainer
    training_args = setup_training_args(model_args, output_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset if model_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Setup proper state dict handling for LoRA
    model.config.use_cache = False
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
    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(trainer.model.state_dict(), output_dir / "adapter_model.bin")

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Final evaluation
    if model_args.do_eval:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)
        
        logger.info("Training complete!")
        logger.info(f"Model saved to: {output_dir}")
        logger.info(f"Final training metrics: {metrics}")
        logger.info(f"Final evaluation metrics: {eval_metrics}")

if __name__ == "__main__":
    main()