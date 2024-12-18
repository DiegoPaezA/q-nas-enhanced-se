""" Copyright (c) 2024, Diego Páez
    * Licensed under The MIT License [see LICENSE for details]

    - Retrain CNN model generated by Q-NAS.
"""

import argparse
import os
import qnas_config as cfg
from util import check_files, init_log, save_results_file
from cnn import input
from cnn import train_resnet as train
import time

def main(**args):

    logger = init_log(args['log_level'], name=__name__)
    # Check if *experiment_path* contains all the necessary files to retrain an evolved model
    experiment_path = args['experiment_path']
    config_code = args['config_code']
    train_params = args
    
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    train_params['experiment_path'] = os.path.join(experiment_path, args['retrain_folder'])
    # Load data
    logger.info(f"Loading data ...")
    data_loader = input.GenericDataLoader(params=train_params)
    train_loader, val_loader = data_loader.get_loader(pin_memory_device=args['device'])
    test_loader = data_loader.get_loader(for_train=False, pin_memory_device=args['device'])
    
    train_params['experiment_path'] = os.path.join(experiment_path, f"retrain_{config_code}_{1}")
    
    output_dict = {}
    # Retrain model for the number of repetitions
    for i in range(1, args['num_repetitions']+1):
        logger.info(f"Retraining {experiment_path} repetition {i} ...")
        start_time = time.perf_counter()
        results_dict = train.train_and_eval(params=train_params, train_loader=train_loader, 
                                            val_loader=val_loader, test_loader=test_loader)    
        train_params['experiment_path'] = os.path.join(experiment_path, f"retrain_{config_code}_{i+1}")
        
        end_time = time.perf_counter()
        hours, rem = divmod(end_time - start_time, 3600)
        mins, _ = divmod(rem, 60)
        logger.info(f"Retraining {experiment_path} repetition {i} finished in {int(hours):02d}h:{int(mins):02d}m.")
        
        results_dict["time"] = f"{int(hours):02d}h:{int(mins):02d}m"
        output_dict[f"{train_params['lr_scheduler']}_{config_code}_retrain_{i}"] = results_dict

    # Save results
    logger.info(f"Saving results ...")
    if train_params['lr_scheduler']  != "None":
        file_name = f"retrain_results_{config_code}_{train_params['lr_scheduler']}.txt"
        save_results_file(out_path=experiment_path, results_dict=output_dict, file_name=file_name)
    else:
        save_results_file(out_path=experiment_path, results_dict=output_dict, file_name=f"retrain_results_{config_code}.txt")

    logger.info(f"Retraining finished.")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_path', type=str, required=True,
                        help='Directory where the evolved network logs are.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to input data.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.', 
                        choices=['cifar10', 'cifar100', 'pathmnist','octmnist', 'tissuemnist', 'organamnist' ,'atleta_axial', 'atleta_coronal'])
    parser.add_argument('--retrain_folder', type=str, default='retrain',
                        help='Name of the folder with retrain model files that will be saved '
                             'inside *experiment_path*.')
    parser.add_argument('--model_flag', default='resnet18',help='choose backbone from resnet18, resnet50',type=str)
    parser.add_argument('--config_code', type=str, default='config',
                        help='Name of the code of the configuration of hyperparameters.')
    # Training info
    parser.add_argument('--log_level', choices=['NONE', 'INFO', 'DEBUG'], default='NONE',
                        help='Logging information level.')
    parser.add_argument('--max_epochs', type=int, default=300,
                        help='The maximum number of epochs during training. Default = 300.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Size of the batch that will be divided into multiple GPUs.'
                             ' Default = 128.')
    parser.add_argument('--eval_batch_size', type=int, default=1000,
                        help='Size of the evaluation batch. Default = 0 (which means the '
                             'entire validation dataset at once).')
    parser.add_argument('--limit_data', action='store_true',
                        help='Limit the amount of data to be used during training. Default = False.')
    parser.add_argument('--limit_data_value', type=int, default=10000,
                        help='Amount of data to be used during training. Default = 10000.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to be used during training. Default = cuda.')
    parser.add_argument('--num_repetitions', type=int, default=1,
                        help='Number of times the training will be repeated. Default = 1.')
    parser.add_argument('--lr_scheduler', type=str, default="None", choices=['cosine', 'reduce_on_plateau', 'exponential','multistep', 'None'],
                        help='Learning rate scheduler. Default = None.')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['RMSProp', 'Adam', 'AdamW', 'SGD'],
                        help='Optimizer to be used during training. Default = AdamW.')
    parser.add_argument('--data_augmentation', action='store_true',
                    help='Disable data augmentation during training. Default = False.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers to be used during data loading. Default = 4.')
    parser.add_argument('--save_checkpoints_epochs', type=int, default=5,
                        help='Number of epochs to save the model. Default = 5.')
    arguments = parser.parse_args()

    main(**vars(arguments))