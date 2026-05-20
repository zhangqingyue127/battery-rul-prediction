import json
import os

import numpy as np
import pandas as pd

from main import CONFIG, convert_numpy_types
from src.data.loader import load_battery_data, setup_seed
from src.training.model_comparison_config import MODEL_COMPARISON_CONFIG
from src.training.model_comparison import run_model_comparison
from src.visualization.plot_model_comparison import (
    plot_model_metric_histories,
    plot_model_predictions,
)


def save_model_comparison_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "model_comparison_results.json")
    csv_path = os.path.join(out_dir, "model_comparison_rmse.csv")
    xnet_search_path = os.path.join(out_dir, "xnet_param_search.csv")

    with open(json_path, "w") as file:
        json.dump(convert_numpy_types(results), file, indent=4)

    rows = []
    battery_names = CONFIG["battery_list"]
    baseline_models = MODEL_COMPARISON_CONFIG["baseline_models"]
    for model_name, model_scores in results["scores"].items():
        for battery_name, rmse in zip(battery_names, model_scores):
            rows.append(
                {
                    "model": model_name,
                    "battery": battery_name,
                    "rmse": float(rmse),
                }
            )
        rows.append(
            {
                "model": model_name,
                "battery": "average",
                "rmse": float(np.mean(model_scores)),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    candidate_rows = []
    ratio_results = results.get("ratio_results")
    if ratio_results:
        for ratio_label, ratio_result in ratio_results.items():
            for candidate in ratio_result.get("xnet_candidates", []):
                cauchy_params = candidate["cauchy_params"]
                candidate_rows.append(
                    {
                        "train_ratio": ratio_label,
                        "combo_idx": candidate["combo_idx"],
                        "lr": candidate["lr"],
                        "hidden_dim": candidate["hidden_dim"],
                        "num_layers": candidate["num_layers"],
                        "lambda1": cauchy_params["lambda1"],
                        "lambda2": cauchy_params["lambda2"],
                        "d": cauchy_params["d"],
                        "avg_rmse": candidate["avg_rmse"],
                    }
                )
    else:
        for candidate in results.get("xnet_candidates", []):
            cauchy_params = candidate["cauchy_params"]
            candidate_rows.append(
                {
                    "train_ratio": "single",
                    "combo_idx": candidate["combo_idx"],
                    "lr": candidate["lr"],
                    "hidden_dim": candidate["hidden_dim"],
                    "num_layers": candidate["num_layers"],
                    "lambda1": cauchy_params["lambda1"],
                    "lambda2": cauchy_params["lambda2"],
                    "d": cauchy_params["d"],
                    "avg_rmse": candidate["avg_rmse"],
                }
            )
    if candidate_rows:
        pd.DataFrame(candidate_rows).to_csv(xnet_search_path, index=False)

    table_path = os.path.join(out_dir, "model_comparison_summary.csv")
    summary_rows = []
    scores = results["scores"]
    for index, battery_name in enumerate(battery_names):
        row = {"battery": battery_name}
        for model_name in ["XNet"] + baseline_models:
            row[f"{model_name}_rmse"] = float(scores[model_name][index])
        baseline_avg = float(np.mean([scores[m][index] for m in baseline_models]))
        row["baseline_average_rmse"] = baseline_avg
        row["xnet_improve_percent"] = (
            (baseline_avg - float(scores["XNet"][index])) / max(1e-12, baseline_avg) * 100
        )
        summary_rows.append(row)

    avg_row = {"battery": "average"}
    for model_name in ["XNet"] + baseline_models:
        avg_row[f"{model_name}_rmse"] = float(np.mean(scores[model_name]))
    baseline_avg = float(np.mean([avg_row[f"{m}_rmse"] for m in baseline_models]))
    avg_row["baseline_average_rmse"] = baseline_avg
    avg_row["xnet_improve_percent"] = (
        (baseline_avg - avg_row["XNet_rmse"]) / max(1e-12, baseline_avg) * 100
    )
    summary_rows.append(avg_row)
    pd.DataFrame(summary_rows).to_csv(table_path, index=False)

    print(f"Model comparison JSON saved: {json_path}")
    print(f"Model comparison CSV saved: {csv_path}")
    if candidate_rows:
        print(f"XNet parameter search CSV saved: {xnet_search_path}")
    print(f"Model comparison summary saved: {table_path}")


def print_summary(results):
    baseline_models = MODEL_COMPARISON_CONFIG["baseline_models"]
    xnet_best = results["xnet_best"]
    scores = results["scores"]
    print("\nModel Comparison Summary")
    print("=" * 100)
    print(
        "Best XNet params: "
        f"lr={xnet_best['lr']}, hidden_dim={xnet_best['hidden_dim']}, "
        f"num_layers={xnet_best['num_layers']}, cauchy={xnet_best['cauchy_params']}"
    )
    header = f"{'Battery':<12}" + "".join(
        f"{model:<12}" for model in ["XNet"] + baseline_models
    )
    print(header)
    print("-" * len(header))

    for index, battery_name in enumerate(CONFIG["battery_list"]):
        row = f"{battery_name:<12}"
        for model_name in ["XNet"] + baseline_models:
            row += f"{scores[model_name][index]:<12.4f}"
        print(row)

    print("-" * len(header))
    row = f"{'Average':<12}"
    for model_name in ["XNet"] + baseline_models:
        row += f"{np.mean(scores[model_name]):<12.4f}"
    print(row)


def main():
    setup_seed(MODEL_COMPARISON_CONFIG["common_params"]["seed"])
    os.makedirs(CONFIG["fig_out_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_out_dir"], exist_ok=True)

    battery_data = load_battery_data(CONFIG["data_dir"], CONFIG["battery_list"])
    results = run_model_comparison(battery_data, MODEL_COMPARISON_CONFIG)
    save_model_comparison_results(results, CONFIG["data_out_dir"])
    plot_model_metric_histories(results["histories"], CONFIG["fig_out_dir"])
    plot_model_predictions(
        battery_data,
        results["predictions"],
        CONFIG["fig_out_dir"],
        rated_capacity=MODEL_COMPARISON_CONFIG["common_params"]["rated_capacity"],
    )
    print_summary(results)


if __name__ == "__main__":
    main()
