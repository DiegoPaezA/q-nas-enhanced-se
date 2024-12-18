#!/bin/bash

# Define common variables for the evolution experiments
dataset="organamnist" # Change this to the dataset you want to run: "pathmnist", "octmnist", "tissuemnist", "organamnist"
exp_path_base="exp_${dataset}"
config_file="config_files_medmnist"
fitness_metric="best_accuracy"
data_path="${dataset}_data"
log_level="DEBUG"

configs=("config1.txt" "config2.txt" "config3.txt" "config4.txt")
exps=("exp1" "exp2" "exp3" "exp4")
cuda_devices=("0,1,2" "0,1,2" "0,1,2" "0,1,2") 

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"
    
    echo "Running evolution experiment with $config"

    for ((i=1; i<=3; i++)); do # Change the range to the number of repeats
        exp_path="${exp_path_base}/${exp}_repeat_$i"
        
        CUDA_VISIBLE_DEVICES="$cuda_device" python run_evolution.py \
            --experiment_path "$exp_path" \
            --config_file "${config_file}/${config}" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --fitness_metric "$fitness_metric" \
            --log_level "$log_level"
    done
done