#!/bin/bash

# Define variables for retrain experiment
dataset="organamnist"
exp="exp1" # Change this to the experiment you want to retrain
repeat="1" # Change this to the repetition you want to retrain
echo "Starting $exp F13 repeat $repeat"
exp_path="exp_${dataset}/${exp}_repeat_${repeat}"

# Retrain model
CUDA_VISIBLE_DEVICES=0 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F13 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 128 \
    --eval_batch_size 128 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --optimizer "AdamW"

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Retrain model script failed."
    exit 1
fi

echo "Retrain model script completed successfully."
