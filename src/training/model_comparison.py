import itertools

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.loader import setup_seed
from src.data.preprocess import build_instances, get_train_test
from src.model.baselines import DEFAULT_CAUCHY_PARAMS, build_model
from src.training.metrics import evaluation_rmse


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


def _get_train_test_for_model_comparison(
    battery_data, name, window_size=16, train_split_ratio=0.9, normalize_features=True
):
    """Build the multi-model benchmark split used by the original tuning script."""
    data_seq = battery_data[name][1]
    split_idx = int(len(data_seq) * train_split_ratio) - window_size
    split_idx = max(1, split_idx)
    train_data = data_seq[: split_idx + window_size]
    test_data = data_seq[split_idx + window_size :]

    train_x, train_y = build_instances(train_data, window_size)
    train_mean = float(np.mean(train_x))
    train_std = float(np.std(train_x) + 1e-8)
    if normalize_features:
        train_x = (train_x - train_mean) / train_std

    for other_name, other_data in battery_data.items():
        if other_name == name:
            continue
        other_seq = other_data[1]
        other_split_idx = int(len(other_seq) * train_split_ratio) - window_size
        other_split_idx = max(1, other_split_idx)
        other_train = other_seq[: other_split_idx + window_size]
        other_x, other_y = build_instances(other_train, window_size)
        if normalize_features:
            other_x = (other_x - train_mean) / train_std
        train_x = np.r_[train_x, other_x]
        train_y = np.r_[train_y, other_y]

    test_seq = train_data + test_data
    test_x, test_y = build_instances(test_seq, window_size)
    if normalize_features:
        test_x = (test_x - train_mean) / train_std

    return train_x, train_y, train_data, test_data, data_seq, test_x, test_y


def train_model_across_batteries(
    battery_data,
    model_name,
    lr,
    feature_size,
    hidden_dim=64,
    num_layers=2,
    weight_decay=0.001,
    cauchy_params=None,
    train_split_ratio=0.4,
    epochs=500,
    seed=42,
    device=None,
    eval_interval=50,
    rated_capacity=2.0,
    normalize_features=False,
    scheduler_enabled=False,
    grad_clip=None,
):
    """Train one model on each target battery and return histories/predictions."""
    setup_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cauchy_params = dict(cauchy_params or DEFAULT_CAUCHY_PARAMS)

    hist_epochs = []
    per_epoch_train_mse = {}
    per_epoch_train_mae = {}
    per_epoch_val_rmse = {}
    per_epoch_val_mae = {}
    score_list = []
    result_list = []

    for name in battery_data:
        if normalize_features:
            (
                train_x,
                train_y,
                train_data,
                _,
                data_seq,
                test_x,
                test_y,
            ) = _get_train_test_for_model_comparison(
                battery_data,
                name,
                window_size=feature_size,
                train_split_ratio=train_split_ratio,
                normalize_features=True,
            )
        else:
            train_x, train_y, train_data, _, _, data_seq, test_x, test_y = get_train_test(
                battery_data, name, feature_size, train_split_ratio
            )

        model = build_model(
            model_name,
            feature_size=feature_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cauchy_params=cauchy_params,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = None
        if scheduler_enabled:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=lr * 0.01
            )
        criterion = nn.MSELoss()
        best_score = float("inf")
        best_pred = None

        for epoch in range(epochs):
            model.train()
            x_train, y_train = _make_xy(
                train_x, train_y, feature_size, rated_capacity, device
            )
            optimizer.zero_grad()
            output = _as_model_output(model(x_train))
            loss = criterion(output, y_train)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if (epoch + 1) % eval_interval == 0 or (epoch + 1) == epochs:
                model.eval()
                with torch.no_grad():
                    pred_train = (
                        output.detach().cpu().numpy().reshape(-1) * rated_capacity
                    )
                    y_train_np = (
                        y_train.detach().cpu().numpy().reshape(-1) * rated_capacity
                    )
                    x_val, _ = _make_xy(
                        test_x, test_y, feature_size, rated_capacity, device
                    )
                    pred_val = _as_model_output(model(x_val))
                    pred_val = (
                        pred_val.detach().cpu().numpy().reshape(-1) * rated_capacity
                    )

                train_mse = mean_squared_error(y_train_np, pred_train)
                train_mae = mean_absolute_error(y_train_np, pred_train)
                val_rmse = evaluation_rmse(test_y, pred_val)
                val_mae = mean_absolute_error(test_y, pred_val)
                current_epoch = epoch + 1

                hist_epochs.append(current_epoch)
                per_epoch_train_mse.setdefault(current_epoch, []).append(train_mse)
                per_epoch_train_mae.setdefault(current_epoch, []).append(train_mae)
                per_epoch_val_rmse.setdefault(current_epoch, []).append(val_rmse)
                per_epoch_val_mae.setdefault(current_epoch, []).append(val_mae)

                if val_rmse < best_score:
                    best_score = val_rmse
                    best_pred = pred_val.copy()

        if best_pred is None:
            raise RuntimeError(f"No prediction was produced for {model_name}/{name}")

        full_pred = np.full(len(data_seq), np.nan, dtype=float)
        full_pred[: len(train_data)] = train_data
        pred_start = feature_size
        pred_end = min(pred_start + len(best_pred), len(full_pred))
        full_pred[pred_start:pred_end] = best_pred[: pred_end - pred_start]
        result_list.append(np.nan_to_num(full_pred, nan=0.0).tolist())
        score_list.append(float(best_score))

    unique_epochs = sorted(set(hist_epochs))
    history = {
        "epochs": unique_epochs,
        "train_mse": [
            float(np.mean(per_epoch_train_mse[epoch])) for epoch in unique_epochs
        ],
        "train_mae": [
            float(np.mean(per_epoch_train_mae[epoch])) for epoch in unique_epochs
        ],
        "val_rmse": [
            float(np.mean(per_epoch_val_rmse[epoch])) for epoch in unique_epochs
        ],
        "val_mae": [
            float(np.mean(per_epoch_val_mae[epoch])) for epoch in unique_epochs
        ],
    }

    return history, score_list, result_list, float(np.mean(score_list))


def iter_xnet_grid(param_grid):
    keys = ["lr", "hidden_dim", "num_layers", "cauchy_params"]
    for values in itertools.product(*(param_grid[key] for key in keys)):
        yield dict(zip(keys, values))


def search_xnet_params(battery_data, param_grid, common_params):
    candidates = []
    for combo_idx, combo in enumerate(iter_xnet_grid(param_grid), start=1):
        print(
            f"\n--- XNet combo {combo_idx}: lr={combo['lr']}, "
            f"hidden_dim={combo['hidden_dim']}, layers={combo['num_layers']}, "
            f"cauchy={combo['cauchy_params']} ---"
        )
        history, scores, results, avg_rmse = train_model_across_batteries(
            battery_data=battery_data,
            model_name="XNet",
            lr=combo["lr"],
            feature_size=common_params["feature_size"],
            hidden_dim=combo["hidden_dim"],
            num_layers=combo["num_layers"],
            weight_decay=common_params["weight_decay"],
            cauchy_params=combo["cauchy_params"],
            train_split_ratio=common_params["train_split_ratio"],
            epochs=common_params["epochs"],
            seed=common_params["seed"],
            eval_interval=common_params["eval_interval"],
            rated_capacity=common_params["rated_capacity"],
            normalize_features=common_params.get("normalize_features", False),
            scheduler_enabled=common_params.get("scheduler_enabled", False),
            grad_clip=common_params.get("grad_clip"),
        )
        candidates.append(
            {
                **combo,
                "combo_idx": combo_idx,
                "avg_rmse": avg_rmse,
                "hist": history,
                "scores": scores,
                "results": results,
            }
        )
        print(f"    Avg RMSE: {avg_rmse:.4f}")

    return min(candidates, key=lambda item: item["avg_rmse"]), candidates


def _run_model_comparison_single_ratio(battery_data, config, train_ratio):
    common_params = dict(config["common_params"])
    common_params["train_split_ratio"] = train_ratio
    xnet_best, xnet_candidates = search_xnet_params(
        battery_data, config["xnet_param_grid"], common_params
    )

    histories = {"XNet": xnet_best["hist"]}
    scores = {"XNet": xnet_best["scores"]}
    predictions = {"XNet": xnet_best["results"]}

    baseline_params = config["baseline_params"]
    for model_name in config["baseline_models"]:
        print(f"\n--- Training {model_name} baseline ---")
        history, model_scores, results, avg_rmse = train_model_across_batteries(
            battery_data=battery_data,
            model_name=model_name,
            lr=baseline_params["lr"],
            feature_size=common_params["feature_size"],
            hidden_dim=baseline_params["hidden_dim"],
            num_layers=baseline_params["num_layers"],
            weight_decay=baseline_params["weight_decay"],
            cauchy_params=baseline_params["cauchy_params"],
            train_split_ratio=common_params["train_split_ratio"],
            epochs=common_params["epochs"],
            seed=common_params["seed"],
            eval_interval=common_params["eval_interval"],
            rated_capacity=common_params["rated_capacity"],
            normalize_features=common_params.get("normalize_features", False),
            scheduler_enabled=common_params.get("scheduler_enabled", False),
            grad_clip=common_params.get("grad_clip"),
        )
        histories[model_name] = history
        scores[model_name] = model_scores
        predictions[model_name] = results
        print(f"    Avg RMSE: {avg_rmse:.4f}")

    return {
        "xnet_best": xnet_best,
        "xnet_candidates": xnet_candidates,
        "histories": histories,
        "scores": scores,
        "predictions": predictions,
    }


def run_model_comparison(battery_data, config):
    """Compare XNet and baseline models across the configured train ratios."""
    train_ratios = config.get(
        "train_ratios",
        [config["common_params"].get("train_split_ratio", 0.4)],
    )
    ratio_results = {}

    for ratio in train_ratios:
        ratio_key = f"{int(ratio * 100)}%"
        print(f"\n=== Multi-model comparison with {ratio_key} training data ===")
        ratio_results[ratio_key] = _run_model_comparison_single_ratio(
            battery_data,
            config,
            ratio,
        )

    first_key = f"{int(train_ratios[0] * 100)}%"
    first_result = ratio_results[first_key]
    return {
        **first_result,
        "train_ratios": train_ratios,
        "ratio_results": ratio_results,
    }
