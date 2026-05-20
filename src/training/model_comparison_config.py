FIXED_CAUCHY_PARAMS = {"lambda1": 0.7, "lambda2": 0.1, "d": 0.5}

MODEL_COMPARISON_CONFIG = {
    "baseline_models": ["FC", "LSTM", "GRU", "CNN", "ResNet"],
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "baseline_params": {
        "lr": 0.01,
        "hidden_dim": 64,
        "num_layers": 2,
        "weight_decay": 0.001,
        "cauchy_params": FIXED_CAUCHY_PARAMS,
    },
    "xnet_param_grid": {
        "lr": [0.01],
        "hidden_dim": [128],
        "num_layers": [2],
        "cauchy_params": [FIXED_CAUCHY_PARAMS],
    },
    "common_params": {
        "feature_size": 16,
        "train_split_ratio": 0.4,
        "weight_decay": 0.0015,
        "epochs": 800,
        "eval_interval": 40,
        "seed": 42,
        "rated_capacity": 2.0,
        "normalize_features": True,
        "scheduler_enabled": True,
        "grad_clip": 5.0,
    },
}
