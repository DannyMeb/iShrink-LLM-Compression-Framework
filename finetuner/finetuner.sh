#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

# Create output directories if they don't exist
mkdir -p experiments/results/finetuned_models

# Default parameters
LEARNING_RATE=2e-5
NUM_EPOCHS=1
BLOCK_SIZE=2048
BATCH_SIZE=8
MAX_TRAIN_SAMPLES=30000
MAX_EVAL_SAMPLES=128
TRAINING_PERCENTAGE=1

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