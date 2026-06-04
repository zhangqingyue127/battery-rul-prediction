"""Ablation study for XNet Cauchy Activation Function components."""

import numpy as np

from src.training.model_comparison import train_model_across_batteries
from src.training.model_comparison_config import FIXED_CAUCHY_PARAMS

ABLATION_MODELS = ["XNet-Full", "XNet-NoEven", "XNet-NoOdd", "XNet-Fixed", "XNet-LN"]

ABLATION_LABELS = {
    "XNet-Full":   "XNet-Full",
    "XNet-NoEven": "w/o Even (λ₂)",
    "XNet-NoOdd":  "w/o Odd (λ₁)",
    "XNet-Fixed":  "Fixed Params",
    "XNet-LN":     "w/ LayerNorm",
}

ABLATION_CONFIG = {
    "models": ABLATION_MODELS,
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "common_params": {
        "feature_size": 16,
        "hidden_dim": 128,
        "num_layers": 2,
        "lr": 0.01,
        "weight_decay": 0.0015,
        "epochs": 800,
        "eval_interval": 40,
        "seed": 42,
        "rated_capacity": 2.0,
        "normalize_features": True,
        "scheduler_enabled": True,
        "grad_clip": 5.0,
        "cauchy_params": FIXED_CAUCHY_PARAMS,
    },
}


def run_ablation_study(battery_data, config=None):
    """Run ablation study: train each XNet variant across all train ratios.

    Returns a nested dict: results[ratio_label][model_name] = {scores, avg_rmse}
    """
    if config is None:
        config = ABLATION_CONFIG

    common = config["common_params"]
    train_ratios = config["train_ratios"]
    model_names = config["models"]

    results = {}
    for ratio in train_ratios:
        ratio_key = f"{int(ratio * 100)}%"
        print(f"\n=== Ablation — {ratio_key} training data ===")
        results[ratio_key] = {}
        for model_name in model_names:
            label = ABLATION_LABELS.get(model_name, model_name)
            print(f"  [{model_name}] {label}")
            _, scores, _, avg_rmse = train_model_across_batteries(
                battery_data=battery_data,
                model_name=model_name,
                lr=common["lr"],
                feature_size=common["feature_size"],
                hidden_dim=common["hidden_dim"],
                num_layers=common["num_layers"],
                weight_decay=common["weight_decay"],
                cauchy_params=common["cauchy_params"],
                train_split_ratio=ratio,
                epochs=common["epochs"],
                seed=common["seed"],
                eval_interval=common["eval_interval"],
                rated_capacity=common["rated_capacity"],
                normalize_features=common.get("normalize_features", False),
                scheduler_enabled=common.get("scheduler_enabled", False),
                grad_clip=common.get("grad_clip"),
            )
            results[ratio_key][model_name] = {
                "scores": [float(s) for s in scores],
                "avg_rmse": float(avg_rmse),
            }
            print(f"    Avg RMSE: {avg_rmse:.4f}")

    return results
