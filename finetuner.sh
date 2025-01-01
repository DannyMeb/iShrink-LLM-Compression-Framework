#!/bin/bash

set -e

# export CUDA_VISIBLE_DEVICES=3
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create output directories
mkdir -p cache

# Training parameters
LEARNING_RATE=1e-4
NUM_EPOCHS=5
BLOCK_SIZE=2048
BATCH_SIZE=2
MAX_TRAIN_SAMPLES=2000
MAX_EVAL_SAMPLES=256
TRAINING_PERCENTAGE=75

echo "Starting training with the following configuration:"
echo "Learning Rate: $LEARNING_RATE"
echo "Number of Epochs: $NUM_EPOCHS"
echo "Block Size: $BLOCK_SIZE"
echo "Batch Size: $BATCH_SIZE"
echo "Max Train Samples: $MAX_TRAIN_SAMPLES"
echo "Max Eval Samples: $MAX_EVAL_SAMPLES"
echo "Training Data Percentage: $TRAINING_PERCENTAGE%"

python3 src/healer.py \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --block_size $BLOCK_SIZE \
    --batch_size $BATCH_SIZE \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --training_percentage $TRAINING_PERCENTAGE \
    --do_eval

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Final GPU memory status:"
    nvidia-smi
else
    echo "Training failed with exit code $?"
    exit 1
fi