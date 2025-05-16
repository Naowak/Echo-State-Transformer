from libs.TransformerDecoderOnly import TransformerDecoderOnly
from libs.EST import EST
from libs.LSTM import LSTM
from libs.GRU import GRU
from datetime import datetime
from libs.stream_tasks import compute_score
from libs.stream_evals import small_eval
import torch
import os
import json
import argparse
import time
import numpy as np

# Define models and parameters
models = {
    "Transformer": {
        "modelClass": TransformerDecoderOnly,
        "model_params": {
            # 1k
            'transformer-1-1k': {"d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 29}, # FF, low Head
            'transformer-2-1k': {"d_model": 8, "nhead": 4, "num_layers": 1, "dim_feedforward": 29}, # FF, big Head
            'transformer-3-1k': {"d_model": 10, "nhead": 2, "num_layers": 1, "dim_feedforward": 10}, # Same, low Head
            'transformer-4-1k': {"d_model": 10, "nhead": 5, "num_layers": 1, "dim_feedforward": 10}, # Same, big Head
            # 10k
            'transformer-1-10k': {"d_model": 28, "nhead": 4, "num_layers": 1, "dim_feedforward": 112}, # FF, low Head
            'transformer-2-10k': {"d_model": 28, "nhead": 7, "num_layers": 1, "dim_feedforward": 112}, # FF, big Head
            'transformer-3-10k': {"d_model": 38, "nhead": 2, "num_layers": 1, "dim_feedforward": 38}, # Same, low Head
            'transformer-4-10k': {"d_model": 38, "nhead": 19, "num_layers": 1, "dim_feedforward": 38}, # Same, big Head
            # 100k
            'transformer-1-100k': {"d_model": 90, "nhead": 5, "num_layers": 1, "dim_feedforward": 360}, # FF, low Head
            'transformer-2-100k': {"d_model": 90, "nhead": 18, "num_layers": 1, "dim_feedforward": 360}, # FF, big Head
            'transformer-3-100k': {"d_model": 128, "nhead": 8, "num_layers": 1, "dim_feedforward": 128}, # Same, low Head
            'transformer-4-100k': {"d_model": 128, "nhead": 16, "num_layers": 1, "dim_feedforward": 128}, # Same, big Head
            # 1M
            'transformer-1-1M': {"d_model": 290, "nhead": 29, "num_layers": 1, "dim_feedforward": 1130}, # FF, low Head
            'transformer-2-1M': {"d_model": 290, "nhead": 58, "num_layers": 1, "dim_feedforward": 1130}, # FF, big Head
            'transformer-3-1M': {"d_model": 405, "nhead": 9, "num_layers": 1, "dim_feedforward": 405}, # FF, low Head
            'transformer-4-1M': {"d_model": 405, "nhead": 45, "num_layers": 1, "dim_feedforward": 405}, # FF, big Head
        },
        "backprop_params": True,
    },
    "LSTM": {
        "modelClass": LSTM,
        "model_params": {
            # 1k
            'lstm-1k': {"hidden_size": 10, "num_layers": 1},
            # 10k
            'lstm-10k': {"hidden_size": 43, "num_layers": 1},
            # 100k
            'lstm-100k': {"hidden_size": 151, "num_layers": 1},
            # 1M
            'lstm-1M': {"hidden_size": 493, "num_layers": 1},
        },
        "backprop_params": True,
    },
    "GRU": {
        "modelClass": GRU,
        "model_params": {
            # 1k
            'lstm-1k': {"hidden_size": 12, "num_layers": 1},
            # 10k
            'lstm-10k': {"hidden_size": 51, "num_layers": 1},
            # 100k
            'lstm-100k': {"hidden_size": 175, "num_layers": 1},
            # 1M
            'lstm-1M': {"hidden_size": 570, "num_layers": 1},
        },
        "backprop_params": True,
    },
    "EST": {
        "modelClass": EST,
        "model_params": {
            # 1k
            'est-1-1k': {"memory_units": 2, "memory_dim": 13, "attention_dim": 6}, # More dim
            'est-2-1k': {"memory_units": 10, "memory_dim": 15, "attention_dim": 3}, # More units
            'est-3-1k': {"memory_units": 5, "memory_dim": 5, "attention_dim": 5}, # Same, but lower reservoir dim
            'est-4-1k': {"memory_units": 4, "memory_dim": 29, "attention_dim": 4}, # Same, but bigger reservoir dim
            # 10k
            'est-1-10k': {"memory_units": 6, "memory_dim": 48, "attention_dim": 13}, # More dim
            'est-2-10k': {"memory_units":14, "memory_dim": 49, "attention_dim": 8}, # More units
            'est-3-10k': {"memory_units":10, "memory_dim": 46, "attention_dim": 10}, # Same, but lower reservoir dim
            'est-4-10k': {"memory_units":8, "memory_dim": 110, "attention_dim": 8}, # Same, but bigger reservoir dim
            # 100k
            'est-1-100k': {"memory_units":12, "memory_dim": 101, "attention_dim": 32}, # More dim
            'est-2-100k': {"memory_units":34, "memory_dim": 100, "attention_dim": 17}, # More units
            'est-3-100k': {"memory_units":23, "memory_dim": 85, "attention_dim": 23}, # Same, but lower reservoir dim
            'est-4-100k': {"memory_units":16, "memory_dim": 314, "attention_dim": 16}, # Same, but bigger reservoir dim
            # 1M
            'est-1-1M': {"memory_units":30, "memory_dim": 241, "attention_dim": 64}, # More dim
            'est-2-1M': {"memory_units":64, "memory_dim": 252, "attention_dim": 38}, # More units
            'est-3-1M': {"memory_units":47, "memory_dim": 253, "attention_dim": 47}, # Same, but lower reservoir dim
            'est-4-1M': {"memory_units":38, "memory_dim": 529, "attention_dim": 38}, # Same, but bigger reservoir dim
        },  
        "backprop_params": True,
    },
}



def test_model(modelClass, model_params, backprop_params, data, classification, weight_decay=1e-5, learning_rate=1e-3, path=None, device='cpu'):
    """
    Test the model on the test set.

    Parameters:
    - modelClass (class): Model class to test.
    - model_params (dict): Parameters of the model.
    - data (dict): Data dictionary containing the train, valid and test data.
    - classification (bool): If the task is a classification task.
    - device (str): Device to use.

    Returns:
    - Score (float): The accuracy or MSE of the model.
    """
    # Unpack the data
    X_train, Y_train, T_train = data['X_train'], data['Y_train'], data['T_train']
    X_valid, Y_valid, T_valid = data['X_valid'], data['Y_valid'], data['T_valid']
    X_test, Y_test, T_test = data['X_test'], data['Y_test'], data['T_test']


    # Train the model
    model = modelClass(**model_params, device=device)
    model.run_training(X_train, Y_train, T_train, X_valid=X_valid, Y_valid=Y_valid, T_valid=T_valid,
                       learning_rate=learning_rate, weight_decay=weight_decay,
                       classification=classification, path=path,
                       **backprop_params)
    
    # Load the best model (the one with the best validation score accross epochs)
    if path is not None:
        model = modelClass.load(path)

    # Compute the loss 
    run_params = {"batch_size": backprop_params['batch_size']} if 'batch_size' in backprop_params else {}
    y_hat = model.run_inference(X_test, **run_params)
    score = compute_score(Y_test, y_hat, T_test, classification)

    return model, score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=0, nargs='+', help='Seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--exp_id', type=str, default=datetime.today().strftime('%Y-%m-%d-%HH-%MM-%SS'), help='Experiment ID')
    parser.add_argument('--size', type=str, default='all', help='Size of the models')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--tasks', type=str, default='all', nargs='+', help='Tasks to run')
    parser.add_argument('--models', type=str, default='all', nargs='+', help='Models to run')
    parser.add_argument('--learning_rates', type=float, default=1e-3, nargs='+', help='Learning rate')
    parser.add_argument('--weight_decays', type=float, default=1e-5, nargs='+', help='Weight decay')
    return parser.parse_args()


if __name__ == "__main__":

    # Parse user args
    args = parse_args()
    seeds = [args.seeds] if type(args.seeds) is int else args.seeds
    learning_rates = [args.learning_rates] if type(args.learning_rates) is float else args.learning_rates
    weight_decays = [args.weight_decays] if type(args.weight_decays) is float else args.weight_decays
    tasks = small_eval

    # Create result & save folder
    folder = f'./results/{args.exp_id}'
    os.makedirs(folder)
    os.makedirs('./models/', exist_ok=True)

    # Filter models by name
    if args.models != 'all':
        models = {model_name: model for model_name, model in models.items() if model_name in args.models}

    # Filter models by size
    if args.size != 'all':
        for model_name, model_config in models.items():
            model_params = {k:v for k,v in model_config['model_params'].items() if args.size in k}
            models[model_name]['model_params'] = model_params

    # Filter tasks by name
    if args.tasks != 'all':
        tasks = {task_name: task for task_name, task in tasks.items() if task_name in args.tasks}
        print('Tasks:', tasks)
    
    # Add layers to the model params
    for model_name, model_config in models.items():
        for model_params_name, model_params in model_config['model_params'].items():
            model_config['model_params'][model_params_name]['num_layers'] = args.layers

    # Print before starting
    print('Experiment ID:', args.exp_id)
    print('Models:', models)
    print('Number of tasks:', len(tasks))
    print('Number of models:', len(models))
    print('Seeds:', seeds)
    print('Learning rates:', learning_rates)
    print('Weight decays:', weight_decays)
    print('Device:', args.device)
    print('Results folder:', folder)
    print('Starting evaluation...')

    # Test the models
    for task_name, task in tasks.items():

        for learning_rate in learning_rates:

            for weight_decay in weight_decays:

                for seed in seeds:

                    # Set the seed
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    # Generate the task
                    # X_train, Y_train, T_train, X_test, Y_test, T_test = task['fct'](**task['params'])
                    data = task['fct'](**task['params'])
                    print('\n ----- TASK ', task_name, '----- ')
                    print("NB samples train:", len(data['X_train']), 
                            "/ NB samples valid:", len(data['X_valid']),
                            "/ NB samples test:", len(data['X_test']))

                    # Test each model
                    for model_name, model in models.items():
                        # Retrieve the backprop params if needed
                        backprop_params = task['backprop_params'] if model['backprop_params'] else {}

                        # Test each model parameter
                        for model_params_name, model_params in model['model_params'].items():
                            # Test the model
                            begin = time.time()
                            m, score = test_model(modelClass=model['modelClass'], model_params=model_params, backprop_params=backprop_params,
                                            data=data, classification=task['classification'],
                                            learning_rate=learning_rate, weight_decay=weight_decay, 
                                            path=f'./models/{task_name}-{model_name}-{args.size}', device=args.device)
                            time_taken = time.time() - begin
                            
                            # Print the results
                            print("----> Model:", model_name, "/ Seed:", seed, "/ Learning Rate", learning_rate, 
                                  "/ Weight Decay", weight_decay, "/ Model Params:", model_params_name, "/ Score:", score)
                            
                            # Save the results
                            result_obj = {
                                "task": task_name,
                                "model": model_name,
                                "model_params": model_params_name,
                                "seed": seed,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "score": score,
                                "time": time_taken,
                            }
                            with open(folder + '/results.txt', 'a') as f:
                                f.write(json.dumps(result_obj) + '\n')


















