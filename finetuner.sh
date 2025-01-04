#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

LEARNING_RATE=3e-4
NUM_EPOCHS=5
BLOCK_SIZE=256
BATCH_SIZE=8
MAX_TRAIN_SAMPLES=40000
MAX_EVAL_SAMPLES=500
TRAINING_PERCENTAGE=100

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