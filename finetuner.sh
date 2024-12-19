#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

# Create temporary output directory
TEMP_OUTPUT_DIR="experiments/results/temp_output"
mkdir -p $TEMP_OUTPUT_DIR

python3 src/healer.py \
    --output_dir $TEMP_OUTPUT_DIR \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --block_size 2048 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --warmup_steps 50 \
    --do_train \
    --do_eval \
    --fp16 \
    --gradient_checkpointing \
    --evaluation_strategy "steps" \
    --eval_steps 100 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --preprocessing_num_workers 1 \
    --dataloader_num_workers 1 \
    --overwrite_output_dir \
    --log_level info \
    --optim "adamw_torch"