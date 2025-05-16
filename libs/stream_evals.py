from libs.stream_tasks import *

small_eval = {
    'sin_forecasting': {
        'fct': generate_sin_forecasting,
        'params': {"sequence_length": 200, "forecast_length": 5, "training_ratio": 0.45, "validation_ratio": 0.1, "testing_ratio": 0.45},
        'classification': False,
        'backprop_params': {"batch_size": 1, "epochs": 250, "patience": 30}
    },
    'chaotic_forecasting': {
        'fct': generate_chaotic_forecasting,
        'params': {"sequence_length": 200, "forecast_length": 5, "training_ratio": 0.45, "validation_ratio": 0.1, "testing_ratio": 0.45},
        'classification': False,
        'backprop_params': {"batch_size": 1, "epochs": 250, "patience": 30}
    },
    'discrete_postcasting': {
        'fct': generate_discrete_postcasting,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 50, "delay": 5, "n_symbols": 3},
        'classification': True,
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30}
    },
    'continue_postcasting': {
        'fct': generate_continue_postcasting,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 50, "delay": 5},
        'classification': False,
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30}
    },
    'discrete_pattern_completion': {
        'fct': generate_discrete_pattern_completion,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 60, "n_symbols": 3, "base_length": 4, "mask_ratio": 0.2}, 
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30},
    },
    'continue_pattern_completion': {
        'fct': generate_continue_pattern_completion,
        'classification': False,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 60, "base_length": 4, "mask_ratio": 0.2}, 
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30},
    },
    'bracket_matching': {
        'fct': generate_bracket_matching,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 50, "max_depth": 5},
        'classification': True,
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30}
    },
    'copy_task': {
        'fct': generate_copy_task,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 22, "delay": 5, "n_symbols": 3}, 
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30},
    },
    'selective_copy_task': {
        'fct': generate_selective_copy_task,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 40, "delay": 5, "n_markers": 5, "n_symbols": 3},
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30},
    },
    'adding_problem': {
        'fct': generate_adding_problem,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 10, "max_number": 3},
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30},
    },
    'sorting_problem': {
        'fct': generate_sorting_problem,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "sequence_length": 10, "n_symbols": 3}, 
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30},
    },
    'mnist_classification': {
        'fct': generate_mnist_classification,
        'classification': True,
        'params': {"n_train": 100, "n_valid": 20, "n_test": 100, "path": "./data/mnist/", "cache_dir": "./data/"},
        'backprop_params': {"batch_size": 10, "epochs": 250, "patience": 30}
    },
}
