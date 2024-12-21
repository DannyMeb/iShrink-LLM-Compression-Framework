#!/bin/bash

# Ensure script exits on any error
set -e

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directories if they don't exist
mkdir -p experiments/results/finetuned_models
mkdir -p cache

# Optimized training parameters
LEARNING_RATE=2e-5
NUM_EPOCHS=4
BLOCK_SIZE=2048
BATCH_SIZE=2             # Reduced batch size for memory efficiency
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=64      # Reduced evaluation samples
TRAINING_PERCENTAGE=50   # Use 70% of training data

# Optional: Clean up old checkpoints (uncomment if needed)
# rm -rf experiments/results/finetuned_models/*

# Print training configuration
echo "Starting training with the following configuration:"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Block Size: $BLOCK_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Max Train Samples: $MAX_TRAIN_SAMPLES"
echo "Max Eval Samples: $MAX_EVAL_SAMPLES"
echo "Training Data Percentage: $TRAINING_PERCENTAGE%"

# Run the finetuning script
python3 src/healer.py \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --block_size $BLOCK_SIZE \
    --batch_size $BATCH_SIZE \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --training_percentage $TRAINING_PERCENTAGE \
    --do_eval

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Print GPU memory status after training
    echo "Final GPU memory status:"
    nvidia-smi
else
    echo "Training failed with exit code $?"
    exit 1
fi