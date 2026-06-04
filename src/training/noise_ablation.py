"""Noise robustness ablation for activation functions in XNet."""

import copy

import numpy as np
import torch
import torch.nn as nn

from src.data.loader import setup_seed
from src.data.preprocess import get_train_test
from src.model.network import XNet
from src.training.metrics import (
    evaluation_mae,
    evaluation_mape,
    evaluation_rmse,
)
from src.training.model_comparison import _get_train_test_for_model_comparison
from src.training.model_comparison_config import FIXED_CAUCHY_PARAMS


NOISE_ABLATION_CONFIG = {
    "activation_functions": ["cauchy", "tanh", "relu", "gelu", "leaky_relu"],
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "noise_levels": [0.0, 0.01, 0.03, 0.05, 0.1],
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
        "use_layer_norm": False,
        "cauchy_params": FIXED_CAUCHY_PARAMS,
    },
}


NOISE_TRAINING_ABLATION_CONFIG = {
    "training_modes": ["clean_training", "noise_injected_training"],
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "common_params": {
        "activation": "cauchy",
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
        "use_layer_norm": False,
        "cauchy_params": FIXED_CAUCHY_PARAMS,
        "train_noise_level": 0.03,
        "train_noise_period": 2,
        "eval_noise_level": 0.05,
    },
}


def add_gaussian_noise(features, noise_level, rng, reference_std=None):
    """Add zero-mean Gaussian noise scaled by feature standard deviation."""
    features = np.asarray(features, dtype=np.float32)
    if noise_level <= 0:
        return features.copy()

    scale = float(reference_std if reference_std is not None else np.std(features))
    scale = max(scale, 1e-8)
    noise = rng.normal(loc=0.0, scale=noise_level * scale, size=features.shape)
    return (features + noise).astype(np.float32)


def _as_model_output(output):
    if output.dim() == 3:
        output = output.squeeze(-1)
    return output


def _make_xy(features, targets, feature_size, rated_capacity, device):
    x = np.reshape(features / rated_capacity, (-1, feature_size, 1))
    y = np.reshape(targets / rated_capacity, (-1, 1))
    return (
        torch.from_numpy(x).float().to(device),
        torch.from_numpy(y).float().to(device),
    )


def _split_battery_data(
    battery_data,
    name,
    feature_size,
    train_split_ratio,
    normalize_features,
):
    if normalize_features:
        return _get_train_test_for_model_comparison(
            battery_data,
            name,
            window_size=feature_size,
            train_split_ratio=train_split_ratio,
            normalize_features=True,
        )

    train_x, train_y, train_data, _, _, data_seq, test_x, test_y = get_train_test(
        battery_data,
        name,
        window_size=feature_size,
        train_split_ratio=train_split_ratio,
    )
    return train_x, train_y, train_data, None, data_seq, test_x, test_y


def train_activation_under_noise(
    battery_data,
    activation,
    noise_levels,
    train_split_ratio,
    common_params,
    device=None,
):
    """Train one activation on clean data and evaluate it under noisy inputs."""
    seed = common_params["seed"]
    setup_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    level_keys = [f"{level:.3g}" for level in noise_levels]
    metric_store = {
        level_key: {"rmse": [], "mae": [], "mape": []}
        for level_key in level_keys
    }
    per_battery = {}

    for battery_idx, name in enumerate(battery_data):
        (
            train_x,
            train_y,
            train_data,
            _,
            data_seq,
            test_x,
            test_y,
        ) = _split_battery_data(
            battery_data,
            name,
            common_params["feature_size"],
            train_split_ratio,
            common_params.get("normalize_features", False),
        )

        model = XNet(
            common_params["feature_size"],
            common_params["hidden_dim"],
            common_params["num_layers"],
            activation,
            common_params.get("cauchy_params"),
            common_params.get("use_layer_norm", False),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=common_params["lr"],
            weight_decay=common_params["weight_decay"],
        )
        scheduler = None
        if common_params.get("scheduler_enabled", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=common_params["epochs"],
                eta_min=common_params["lr"] * 0.01,
            )
        criterion = nn.MSELoss()
        best_rmse = float("inf")
        best_state = None

        x_train, y_train = _make_xy(
            train_x,
            train_y,
            common_params["feature_size"],
            common_params["rated_capacity"],
            device,
        )

        for epoch in range(common_params["epochs"]):
            model.train()
            optimizer.zero_grad()
            output = _as_model_output(model(x_train))
            loss = criterion(output, y_train)
            loss.backward()
            if common_params.get("grad_clip") is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    common_params["grad_clip"],
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % common_params["eval_interval"] == 0 or (
                epoch + 1
            ) == common_params["epochs"]:
                model.eval()
                with torch.no_grad():
                    x_val, _ = _make_xy(
                        test_x,
                        test_y,
                        common_params["feature_size"],
                        common_params["rated_capacity"],
                        device,
                    )
                    pred = _as_model_output(model(x_val))
                    pred = (
                        pred.detach().cpu().numpy().reshape(-1)
                        * common_params["rated_capacity"]
                    )
                current_rmse = evaluation_rmse(test_y, pred)
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError(f"No checkpoint was produced for {activation}/{name}")

        model.load_state_dict(best_state)
        model.eval()

        per_battery[name] = {}
        reference_std = float(np.std(train_x) + 1e-8)
        for level, level_key in zip(noise_levels, level_keys):
            rng = np.random.default_rng(seed + battery_idx * 1009 + int(level * 10000))
            noisy_test_x = add_gaussian_noise(
                test_x,
                level,
                rng,
                reference_std=reference_std,
            )
            with torch.no_grad():
                x_test, _ = _make_xy(
                    noisy_test_x,
                    test_y,
                    common_params["feature_size"],
                    common_params["rated_capacity"],
                    device,
                )
                pred = _as_model_output(model(x_test))
                pred = (
                    pred.detach().cpu().numpy().reshape(-1)
                    * common_params["rated_capacity"]
                )

            metrics = {
                "rmse": float(evaluation_rmse(test_y, pred)),
                "mae": float(evaluation_mae(test_y, pred)),
                "mape": float(evaluation_mape(test_y, pred)),
            }
            per_battery[name][level_key] = metrics
            for metric_name, value in metrics.items():
                metric_store[level_key][metric_name].append(value)

    aggregate = {}
    for level_key, metrics in metric_store.items():
        aggregate[level_key] = {
            metric_name: float(np.mean(values))
            for metric_name, values in metrics.items()
        }

    return {"aggregate": aggregate, "per_battery": per_battery}


def run_noise_ablation_study(battery_data, config=None):
    """Run noise robustness ablation over activations, ratios, and noise levels."""
    if config is None:
        config = NOISE_ABLATION_CONFIG

    common = config["common_params"]
    activations = config["activation_functions"]
    train_ratios = config["train_ratios"]
    noise_levels = config["noise_levels"]

    results = {}
    for ratio in train_ratios:
        ratio_key = f"{int(ratio * 100)}%"
        print(f"\n=== Noise ablation - {ratio_key} training data ===")
        results[ratio_key] = {}
        for activation in activations:
            print(f"  [{activation}] evaluating noise levels: {noise_levels}")
            results[ratio_key][activation] = train_activation_under_noise(
                battery_data=battery_data,
                activation=activation,
                noise_levels=noise_levels,
                train_split_ratio=ratio,
                common_params=common,
            )
            clean_rmse = results[ratio_key][activation]["aggregate"]["0"]["rmse"]
            max_key = f"{max(noise_levels):.3g}"
            max_rmse = results[ratio_key][activation]["aggregate"][max_key]["rmse"]
            print(f"    RMSE clean={clean_rmse:.4f}, max-noise={max_rmse:.4f}")

    return results


def _average_epoch_histories(epoch_values):
    epochs = sorted(epoch_values.keys())
    return epochs, [float(np.mean(epoch_values[epoch])) for epoch in epochs]


def train_noise_training_mode(
    battery_data,
    training_mode,
    train_split_ratio,
    common_params,
    device=None,
):
    """Train clean vs. periodically noise-injected XNet and record convergence."""
    if training_mode not in {"clean_training", "noise_injected_training"}:
        raise ValueError(f"Unsupported training mode: {training_mode}")

    seed = common_params["seed"]
    setup_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hist_train_mse = {}
    hist_train_mae = {}
    hist_val_rmse = {}
    hist_val_mae = {}
    hist_noisy_val_rmse = {}
    hist_noisy_val_mae = {}
    clean_metrics = {"rmse": [], "mae": [], "mape": []}
    noisy_metrics = {"rmse": [], "mae": [], "mape": []}
    per_battery = {}

    for battery_idx, name in enumerate(battery_data):
        (
            train_x,
            train_y,
            train_data,
            _,
            data_seq,
            test_x,
            test_y,
        ) = _split_battery_data(
            battery_data,
            name,
            common_params["feature_size"],
            train_split_ratio,
            common_params.get("normalize_features", False),
        )

        model = XNet(
            common_params["feature_size"],
            common_params["hidden_dim"],
            common_params["num_layers"],
            common_params["activation"],
            common_params.get("cauchy_params"),
            common_params.get("use_layer_norm", False),
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=common_params["lr"],
            weight_decay=common_params["weight_decay"],
        )
        scheduler = None
        if common_params.get("scheduler_enabled", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=common_params["epochs"],
                eta_min=common_params["lr"] * 0.01,
            )
        criterion = nn.MSELoss()
        best_clean_rmse = float("inf")
        best_state = None
        reference_std = float(np.std(train_x) + 1e-8)

        base_x_train, y_train = _make_xy(
            train_x,
            train_y,
            common_params["feature_size"],
            common_params["rated_capacity"],
            device,
        )

        for epoch in range(common_params["epochs"]):
            model.train()
            train_features = train_x
            should_inject = (
                training_mode == "noise_injected_training"
                and common_params.get("train_noise_level", 0.0) > 0
                and (epoch + 1) % max(int(common_params.get("train_noise_period", 1)), 1)
                == 0
            )
            if should_inject:
                rng = np.random.default_rng(seed + battery_idx * 1009 + epoch)
                train_features = add_gaussian_noise(
                    train_x,
                    common_params["train_noise_level"],
                    rng,
                    reference_std=reference_std,
                )
                x_train, _ = _make_xy(
                    train_features,
                    train_y,
                    common_params["feature_size"],
                    common_params["rated_capacity"],
                    device,
                )
            else:
                x_train = base_x_train

            optimizer.zero_grad()
            output = _as_model_output(model(x_train))
            loss = criterion(output, y_train)
            loss.backward()
            if common_params.get("grad_clip") is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    common_params["grad_clip"],
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % common_params["eval_interval"] == 0 or (
                epoch + 1
            ) == common_params["epochs"]:
                current_epoch = epoch + 1
                model.eval()
                with torch.no_grad():
                    train_pred = (
                        _as_model_output(model(base_x_train))
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        * common_params["rated_capacity"]
                    )
                    y_train_np = train_y.reshape(-1)

                    x_val, _ = _make_xy(
                        test_x,
                        test_y,
                        common_params["feature_size"],
                        common_params["rated_capacity"],
                        device,
                    )
                    clean_pred = _as_model_output(model(x_val))
                    clean_pred = (
                        clean_pred.detach().cpu().numpy().reshape(-1)
                        * common_params["rated_capacity"]
                    )

                    noisy_rng = np.random.default_rng(
                        seed + battery_idx * 1009 + current_epoch * 17
                    )
                    noisy_test_x = add_gaussian_noise(
                        test_x,
                        common_params.get("eval_noise_level", 0.0),
                        noisy_rng,
                        reference_std=reference_std,
                    )
                    x_noisy_val, _ = _make_xy(
                        noisy_test_x,
                        test_y,
                        common_params["feature_size"],
                        common_params["rated_capacity"],
                        device,
                    )
                    noisy_pred = _as_model_output(model(x_noisy_val))
                    noisy_pred = (
                        noisy_pred.detach().cpu().numpy().reshape(-1)
                        * common_params["rated_capacity"]
                    )

                train_mse = float(np.mean((y_train_np - train_pred) ** 2))
                train_mae = float(np.mean(np.abs(y_train_np - train_pred)))
                clean_rmse = float(evaluation_rmse(test_y, clean_pred))
                clean_mae = float(evaluation_mae(test_y, clean_pred))
                noisy_rmse = float(evaluation_rmse(test_y, noisy_pred))
                noisy_mae = float(evaluation_mae(test_y, noisy_pred))

                hist_train_mse.setdefault(current_epoch, []).append(train_mse)
                hist_train_mae.setdefault(current_epoch, []).append(train_mae)
                hist_val_rmse.setdefault(current_epoch, []).append(clean_rmse)
                hist_val_mae.setdefault(current_epoch, []).append(clean_mae)
                hist_noisy_val_rmse.setdefault(current_epoch, []).append(noisy_rmse)
                hist_noisy_val_mae.setdefault(current_epoch, []).append(noisy_mae)

                if clean_rmse < best_clean_rmse:
                    best_clean_rmse = clean_rmse
                    best_state = copy.deepcopy(model.state_dict())

        if best_state is None:
            raise RuntimeError(f"No checkpoint was produced for {training_mode}/{name}")

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            x_clean, _ = _make_xy(
                test_x,
                test_y,
                common_params["feature_size"],
                common_params["rated_capacity"],
                device,
            )
            clean_pred = _as_model_output(model(x_clean))
            clean_pred = (
                clean_pred.detach().cpu().numpy().reshape(-1)
                * common_params["rated_capacity"]
            )

            noisy_rng = np.random.default_rng(seed + battery_idx * 1009 + 99991)
            noisy_test_x = add_gaussian_noise(
                test_x,
                common_params.get("eval_noise_level", 0.0),
                noisy_rng,
                reference_std=reference_std,
            )
            x_noisy, _ = _make_xy(
                noisy_test_x,
                test_y,
                common_params["feature_size"],
                common_params["rated_capacity"],
                device,
            )
            noisy_pred = _as_model_output(model(x_noisy))
            noisy_pred = (
                noisy_pred.detach().cpu().numpy().reshape(-1)
                * common_params["rated_capacity"]
            )

        battery_clean = {
            "rmse": float(evaluation_rmse(test_y, clean_pred)),
            "mae": float(evaluation_mae(test_y, clean_pred)),
            "mape": float(evaluation_mape(test_y, clean_pred)),
        }
        battery_noisy = {
            "rmse": float(evaluation_rmse(test_y, noisy_pred)),
            "mae": float(evaluation_mae(test_y, noisy_pred)),
            "mape": float(evaluation_mape(test_y, noisy_pred)),
        }
        per_battery[name] = {
            "clean_test": battery_clean,
            "noisy_test": battery_noisy,
            "train_length": int(len(train_data)),
            "sequence_length": int(len(data_seq)),
        }
        for metric_name in clean_metrics:
            clean_metrics[metric_name].append(battery_clean[metric_name])
            noisy_metrics[metric_name].append(battery_noisy[metric_name])

    epochs, train_mse = _average_epoch_histories(hist_train_mse)
    _, train_mae = _average_epoch_histories(hist_train_mae)
    _, val_rmse = _average_epoch_histories(hist_val_rmse)
    _, val_mae = _average_epoch_histories(hist_val_mae)
    _, noisy_val_rmse = _average_epoch_histories(hist_noisy_val_rmse)
    _, noisy_val_mae = _average_epoch_histories(hist_noisy_val_mae)

    return {
        "history": {
            "epochs": epochs,
            "train_mse": train_mse,
            "train_mae": train_mae,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "noisy_val_rmse": noisy_val_rmse,
            "noisy_val_mae": noisy_val_mae,
        },
        "aggregate": {
            "clean_test": {
                metric_name: float(np.mean(values))
                for metric_name, values in clean_metrics.items()
            },
            "noisy_test": {
                metric_name: float(np.mean(values))
                for metric_name, values in noisy_metrics.items()
            },
        },
        "per_battery": per_battery,
    }


def run_noise_training_ablation_study(battery_data, config=None):
    """Compare clean training against periodic noise-injected training."""
    if config is None:
        config = NOISE_TRAINING_ABLATION_CONFIG

    common = config["common_params"]
    results = {}
    for ratio in config["train_ratios"]:
        ratio_key = f"{int(ratio * 100)}%"
        print(f"\n=== Noise training ablation - {ratio_key} training data ===")
        results[ratio_key] = {}
        for mode in config["training_modes"]:
            print(f"  [{mode}]")
            results[ratio_key][mode] = train_noise_training_mode(
                battery_data=battery_data,
                training_mode=mode,
                train_split_ratio=ratio,
                common_params=common,
            )
            clean_rmse = results[ratio_key][mode]["aggregate"]["clean_test"]["rmse"]
            noisy_rmse = results[ratio_key][mode]["aggregate"]["noisy_test"]["rmse"]
            print(f"    clean RMSE={clean_rmse:.4f}, noisy RMSE={noisy_rmse:.4f}")

    return results
