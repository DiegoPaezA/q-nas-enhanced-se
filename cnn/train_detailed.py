""" Copyright (c) 2023, Diego Páez
* Licensed under the MIT license

- Compute the fitness of a model_net using the evolved networks.


"""
import os
import time
import numpy as np
import torch
from medmnist import INFO, Evaluator
import torch.nn as nn
from tqdm.notebook import tqdm
from typing import Dict, List, Union, Any
from sklearn.metrics import confusion_matrix
from cnn import model, input
from util import create_info_file, init_log, load_yaml
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, MultiStepLR

current_directory = os.path.dirname(os.path.dirname(__file__))
log_directory = os.path.join(current_directory, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    
log_file = os.path.join(log_directory, 'retrain.log')
LOGGER = init_log("INFO", name=__name__, file_path=log_file)

def release_gpu_memory(gpu_name='cuda:0'):
    """
    Release GPU memory.
    
    Args:
        gpu_name (str): The name of the GPU device (default is 'cuda').
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. No GPU memory to release.")
        return
    
    if gpu_name == 'cuda':
        gpu_name = 'cuda:0'

    device = torch.device(gpu_name)
    torch.cuda.set_device(device)

    # Get memory usage before clearing the cache
    memory_allocated_before = torch.cuda.memory_allocated(device)
    memory_reserved_before = torch.cuda.memory_reserved(device)

    # Clear the cache
    torch.cuda.empty_cache()

    # Check if there was a significant change
    memory_allocated_after = torch.cuda.memory_allocated(device)
    memory_reserved_after = torch.cuda.memory_reserved(device)

    # Verificar si hubo un cambio significativo
    if memory_allocated_before != memory_allocated_after or memory_reserved_before != memory_reserved_after:
        print("Cache was cleared.")
    else:
        print("Cache was already empty.")

def compute_metrics(model, data_loader, params):
    model.eval()
    all_labels = []
    all_predictions = []
    auc, acc = 0, 0
    y_score = torch.tensor([]).to(params['device'])
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(params['device']), labels.to(params['device'])
            y_logits = model(inputs)
            _, predicted = y_logits.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            if params['task'] == 'multi-class':
                output = y_logits.softmax(dim=-1)
                y_score = torch.cat((y_score, output), 0)

        if params['task'] == 'multi-class':
            y_score = y_score.cpu().detach().numpy()
            evaluator = Evaluator(params['dataset'], split='test', root=params['data_path'])
            metrics = evaluator.evaluate(y_score)
            auc, acc = metrics

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    return conf_matrix, auc, acc

def reset_and_load_best_model(params, best_model_path):
    # Reinitialize the original model
    
    best_model = model.NetworkGraph(num_classes=params["num_classes"], mu=0.99)
    filtered_dict = {key: item for key, item in params['fn_dict'].items() if key in params['net_list']}
    best_model.create_functions(fn_dict=filtered_dict, net_list=params['net_list'])

    input_random = torch.randn(params['input_shape'])
    with torch.no_grad():
        _ = best_model(input_random)
    # Load the state dictionary of the best model into the new model
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(params['device'])

    return best_model

def train_epoch(model, criterion, optimizer, data_loader, params):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(params['device']), labels.to(params['device'])
        optimizer.zero_grad()
        y_logits = model(inputs)
        
        if params['task'] == 'multi-class':
            labels = labels.squeeze().long() # medmnist
            
        loss = criterion(y_logits, labels)
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item()
        _, predicted = y_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    accuracy = 100 * correct / total
    train_loss /= len(data_loader)
    return train_loss, accuracy

def evaluate(model, criterion, data_loader, params, test=False):
    model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(params['device']), labels.to(params['device'])
            y_logits = model(inputs)
            
            if params['task'] == 'multi-class':
                labels = labels.squeeze().long() # medmnist
                
            loss = criterion(y_logits, labels)
            eval_loss += loss.item()
            _, predicted = y_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    eval_loss /= len(data_loader)
    
    if test:
        confusion_matrix, auc, acc = compute_metrics(model, data_loader, params)
        return eval_loss, accuracy, auc, acc , confusion_matrix

    return eval_loss, accuracy

def train(model: torch.nn.Module,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          params: Dict[str, Union[int, float, str]]) -> Dict[str, Union[List[float], float]]:
    """
    Retrain a convolutional neural network model.

    Args:
        model (Module): Model to be trained.
        criterion (Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        test_loader (DataLoader): DataLoader for the test set.
        params (Dict[str, Union[int, float, str]]): Dictionary with parameters necessary for training.
            - 'max_epochs' (int): Number of epochs to train.
            - 'model_path' (str): Path to save the trained model.
            - 'lr_scheduler' (str): Learning rate scheduler to use.
            - 'experiment_path' (str): Path to save the experiment.
            - 'device' (str): Device to use for training.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'auc_score' (float): AUC score on the test set.
        - 'acc_medmnist' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
        - 'total_trainable_params' (int): Total number of trainable parameters in the model.
    """
    model.train()
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    best_accuracy = 0.0
    auc_value = 0.0
    acc_med = 0.0
    training_results = {}
    max_epochs = params['max_epochs']
    milestones = [0.5 * max_epochs, 0.75 * max_epochs]

    best_model_path = os.path.join(params['model_path'], 'best_model.pth')
    
    if params['lr_scheduler'] == 'exponential':
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif params['lr_scheduler'] == 'reduce_on_plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.1)
    elif params['lr_scheduler'] == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0, last_epoch=-1)
    elif params['lr_scheduler'] == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    else:
        lr_scheduler = None
    #for epoch in tqdm(range(1, max_epochs + 1), desc="Retrain Scheme"):
    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = train_epoch(model, criterion, optimizer, train_loader, params)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy)
        
        validation_loss, accuracy = evaluate(model, criterion, val_loader, params)
        validation_losses.append(validation_loss)
        validation_accuracies.append(accuracy)
        
        if accuracy > best_accuracy and  epoch % params['save_checkpoints_epochs'] == 0: 
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            create_info_file(params['model_path'], {'best_accuracy': best_accuracy}, 'best_accuracy.txt')

        if epoch % 50 == 0:
            LOGGER.info(f"Experiment: {params['experiment_path']} - Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss:.2f} - Validation loss: {validation_loss:.2f} - Validation accuracy: {accuracy:.2f}%")
            #print(f"Epoch [{epoch}/{max_epochs}] - Training loss: {train_loss} - Validation loss: {validation_loss} - Validation accuracy: {accuracy}%")

        if lr_scheduler is not None:
            if params['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(accuracy)                
            else:
                lr_scheduler.step()
        
    best_model_loaded = reset_and_load_best_model(params, best_model_path)
    test_loss, test_accuracy, auc_value, acc_med, confusion_matrix = evaluate(best_model_loaded, criterion, test_loader, params, test=True)
    
    LOGGER.info(f"Experiment: {params['experiment_path']} - Test loss: {test_loss:.2f} - Test accuracy: {test_accuracy:.2f}%")
    #print(f"Test loss: {test_loss} - Test accuracy: {test_accuracy}%")
            
    params['t1'] = time.time()
    
    create_info_file(params['model_path'], params, 'retraining_params.txt')
    
        # Measure inference time
    inference_images = next(iter(test_loader))[0][:10].to(params['device'])

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],profile_memory=True, record_shapes=True) as prof:
        with record_function("model_inference"):
            best_model_loaded(inference_images)

    model_memory_usage = sum(event.cuda_memory_usage for event in prof.key_averages()) / (1024 ** 2)
    cpu_inference_time = prof.key_averages()[0].cpu_time
    cuda_inference_time = prof.key_averages()[0].cuda_time
    
    total_trainable_params = sum(p.numel() for p in best_model_loaded.parameters() if p.requires_grad)
    
    
    
    training_results['total_trainable_params'] = total_trainable_params
    training_results['cuda_inference_time'] = cuda_inference_time
    training_results['cpu_inference_time'] = cpu_inference_time
    training_results['model_memory_usage'] = model_memory_usage
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = training_accuracies
    training_results['validation_losses'] = validation_losses
    training_results['validation_accuracies'] = validation_accuracies
    training_results['best_accuracy'] = best_accuracy
    training_results['test_loss'] = test_loss
    training_results['test_accuracy'] = test_accuracy
    training_results['auc_score'] = auc_value
    training_results['acc_medmnist'] = acc_med
    training_results['confusion_matrix'] = confusion_matrix.tolist()
            
    return training_results


def train_and_eval(params: Dict[str, Any], 
                    fn_dict: Dict[str, Any],net_list:List[str],
                    train_loader:torch.utils.data.DataLoader, 
                    val_loader:torch.utils.data.DataLoader,
                    test_loader:torch.utils.data.DataLoader) -> Dict[str, Union[List[float], float]]:
    """
    This function retrains and evaluates a convolutional neural network model using the specified
    configuration.

    Args:
        params (Dict[str, Any]): A dictionary with parameters necessary for training, including.
        fn_dict (Dict[str, Any]): A dictionary with definitions of the possible layers, including
            their names and parameters.
        net_list (List[str]): A list with names of layers defining the network, in the order they appear.

    Returns:
        Dict[str, Union[List[float], float]]: Dictionary with the training results.
        
        - 'training_losses' (List[float]): List of training losses for each epoch.
        - 'training_accuracies' (List[float]): List of training accuracies for each epoch.
        - 'validation_losses' (List[float]): List of validation losses for each epoch.
        - 'validation_accuracies' (List[float]): List of validation accuracies for each epoch.
        - 'best_accuracy' (float): Best validation accuracy achieved.
        - 'test_loss' (float): Loss on the test set.
        - 'test_accuracy' (float): Accuracy on the test set.
        - 'auc_score' (float): AUC score on the test set.
        - 'acc_medmnist' (float): Accuracy on the test set.
        - 'confusion_matrix' (numpy.ndarray): Confusion matrix on the test set.
        - 'total_trainable_params' (int): Total number of trainable parameters in the model.
    """
    
    device = params['device']
    model_path = os.path.join(params['experiment_path'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    params['model_path'] = model_path
    
    LOGGER.info(f"Start retraining of the experiment: {params['experiment_path']}")
    # Load data information
    if params['dataset'].lower() in input.available_datasets:
        dataset_info = input.available_datasets[params['dataset'].lower()]
    else:
        dataset_info = load_yaml(os.path.join(params['data_path'], 'data_info.txt'))
    

    model_net = model.NetworkGraph(num_classes=dataset_info["num_classes"], mu=0.99)
    filtered_dict = {key: item for key, item in fn_dict.items() if key in net_list}
    model_net.create_functions(fn_dict=filtered_dict, net_list=net_list)

    #params['model_net'] = model_net
    params['net_list'] = net_list
    params['fn_dict'] = fn_dict
    params['num_classes'] = dataset_info["num_classes"]
    params['task'] = dataset_info["task"]
    
    # Add the fully connected layer to the model
    input_shape =  [params['batch_size']] + dataset_info['shape']
    inputs = torch.randn(input_shape)
    _ = model_net(inputs)
    model_net.to(device)
    
    params['input_shape'] = input_shape
    
    criterion = nn.CrossEntropyLoss()
    
    if params['optimizer'] == 'RMSProp':
        optimizer = torch.optim.RMSprop(model_net.parameters())
    elif params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model_net.parameters())
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model_net.parameters())
    elif params['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model_net.parameters(), lr=params['learning_rate'])
    else:
        raise ValueError(f"Invalid optimizer: {params['optimizer']}")
        

    # Training time start counting here.
    params['t0'] = time.time()
    
    try:
        results_dict = train(model_net, criterion, optimizer, train_loader, val_loader, test_loader, params)
    except Exception as e:
        if "out of memory" in str(e):
            LOGGER.error(f"Out of memory error: {e}")
            results_dict = None
        else:
            LOGGER.error(f"An error occurred during training: {e}")
            results_dict = None
    
    release_gpu_memory(gpu_name=params['device'])
    
    return results_dict
