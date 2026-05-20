import os
import sys
import json
import pandas as pd
import numpy as np

# Get project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.data.loader import load_battery_data, setup_seed
from src.training.trainer import run_experiments
from src.visualization.plot_activation import plot_activation_characteristics, plot_cauchy_parameter_history
from src.visualization.plot_metrics import plot_metrics_vs_ratio, plot_boxplot_metrics
from src.visualization.plot_prediction import visualize_all_batteries
from src.training.model_comparison import run_model_comparison
from src.training.model_comparison_config import FIXED_CAUCHY_PARAMS, MODEL_COMPARISON_CONFIG
from src.visualization.plot_model_comparison import (
    plot_model_metric_histories,
    plot_model_predictions_by_battery_ratios,
)

# ===================== Experiment Configuration =====================
CONFIG = {
    "battery_list": ["B0005", "B0006", "B0007", "B0018"],
    "data_dir": os.path.join(ROOT_DIR, "data", "raw"),
    "fig_out_dir": os.path.join(ROOT_DIR, "result", "figure"),
    "data_out_dir": os.path.join(ROOT_DIR, "result", "data_results"),
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "activation_functions": ["cauchy", "tanh", "relu", "gelu", "leaky_relu"],
    "run_model_comparison": True,  # Set False during daily debugging to reduce training time.
    "model_params": {
        "lr": 0.01,
        "feature_size": 16,
        "hidden_dim": 64,
        "num_layers": 3,
        "weight_decay": 0.001,
        "epochs": 300,
        "seed": 42,
        "rated_capacity": 2.0,
        # Optional identifiability ablation: set True to apply LayerNorm before CAF.
        # It is disabled in the main experiment because the original benchmark
        # performs better while Cauchy parameters are already regularized by weight_decay.
        "use_layer_norm": False,
        "cauchy_params": FIXED_CAUCHY_PARAMS
    }
}

def convert_numpy_types(obj):
    """Recursively convert NumPy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_experiment_results(final_results, save_dir):
    """Save experiment metrics to CSV and JSON files (for reproducibility)"""
    # Convert NumPy types to native Python types (fix JSON serialization error)
    final_results_converted = convert_numpy_types(final_results)
    
    # 1. Save as JSON (complete raw data)
    json_path = os.path.join(save_dir, "experiment_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(final_results_converted, f, indent=4)
    
    # 2. Save as CSV (easy for table analysis)
    csv_data = []
    for act in CONFIG["activation_functions"]:
        for i, ratio in enumerate(CONFIG["train_ratios"]):
            # Explicitly convert to float to avoid NumPy type issues
            row = {
                "activation_function": act,
                "train_ratio": f"{int(ratio*100)}%",
                "rmse": float(final_results[act]["rmse"][i]),
                "mae": float(final_results[act]["mae"][i]),
                "mape": float(final_results[act]["mape"][i]),
                "r2": float(final_results[act]["r2"][i])
            }
            csv_data.append(row)
    
    csv_path = os.path.join(save_dir, "experiment_metrics.csv")
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    
    print(f"Experiment data saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")

def save_cauchy_parameter_history(param_histories, save_dir):
    """Save trainable Cauchy parameter trajectories for identifiability diagnostics."""
    rows = []
    for ratio, records in param_histories.items():
        for record in records:
            row = {"train_ratio": f"{int(ratio * 100)}%"}
            row.update(record)
            rows.append(row)

    if not rows:
        print("No Cauchy parameter history was recorded.")
        return

    csv_path = os.path.join(save_dir, "cauchy_parameter_history.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = os.path.join(save_dir, "cauchy_parameter_history.json")
    with open(json_path, 'w') as f:
        json.dump(convert_numpy_types(param_histories), f, indent=4)

    print(f"Cauchy parameter history saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")

def save_model_comparison_results(results, out_dir):
    """Save multi-model comparison results to JSON and CSV files."""
    json_path = os.path.join(out_dir, "model_comparison_results.json")
    csv_path = os.path.join(out_dir, "model_comparison_rmse.csv")
    xnet_search_path = os.path.join(out_dir, "xnet_param_search.csv")

    with open(json_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=4)

    battery_names = CONFIG["battery_list"]
    baseline_models = MODEL_COMPARISON_CONFIG["baseline_models"]

    rows = []
    ratio_results = results.get("ratio_results")
    if ratio_results:
        for ratio_label, ratio_result in ratio_results.items():
            for model_name, model_scores in ratio_result["scores"].items():
                for battery_name, rmse in zip(battery_names, model_scores):
                    rows.append({
                        "train_ratio": ratio_label,
                        "model": model_name,
                        "battery": battery_name,
                        "rmse": float(rmse),
                    })
                rows.append({
                    "train_ratio": ratio_label,
                    "model": model_name,
                    "battery": "average",
                    "rmse": float(np.mean(model_scores)),
                })
    else:
        for model_name, model_scores in results["scores"].items():
            for battery_name, rmse in zip(battery_names, model_scores):
                rows.append({
                    "model": model_name,
                    "battery": battery_name,
                    "rmse": float(rmse),
                })
            rows.append({
                "model": model_name,
                "battery": "average",
                "rmse": float(np.mean(model_scores)),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    candidate_rows = []
    ratio_results = results.get("ratio_results")
    if ratio_results:
        for ratio_label, ratio_result in ratio_results.items():
            for candidate in ratio_result.get("xnet_candidates", []):
                cauchy_params = candidate["cauchy_params"]
                candidate_rows.append({
                    "train_ratio": ratio_label,
                    "combo_idx": candidate["combo_idx"],
                    "lr": candidate["lr"],
                    "hidden_dim": candidate["hidden_dim"],
                    "num_layers": candidate["num_layers"],
                    "lambda1": cauchy_params["lambda1"],
                    "lambda2": cauchy_params["lambda2"],
                    "d": cauchy_params["d"],
                    "avg_rmse": candidate["avg_rmse"],
                })
    else:
        for candidate in results.get("xnet_candidates", []):
            cauchy_params = candidate["cauchy_params"]
            candidate_rows.append({
                "train_ratio": "single",
                "combo_idx": candidate["combo_idx"],
                "lr": candidate["lr"],
                "hidden_dim": candidate["hidden_dim"],
                "num_layers": candidate["num_layers"],
                "lambda1": cauchy_params["lambda1"],
                "lambda2": cauchy_params["lambda2"],
                "d": cauchy_params["d"],
                "avg_rmse": candidate["avg_rmse"],
            })
    if candidate_rows:
        pd.DataFrame(candidate_rows).to_csv(xnet_search_path, index=False)

    summary_rows = []
    summary_source = ratio_results or {"single": results}
    for ratio_label, ratio_result in summary_source.items():
        scores = ratio_result["scores"]
        for index, battery_name in enumerate(battery_names):
            row = {"train_ratio": ratio_label, "battery": battery_name}
            for model_name in ["XNet"] + baseline_models:
                row[f"{model_name}_rmse"] = float(scores[model_name][index])
            baseline_avg = float(np.mean([scores[m][index] for m in baseline_models]))
            row["baseline_average_rmse"] = baseline_avg
            row["xnet_improve_percent"] = (
                (baseline_avg - float(scores["XNet"][index])) / max(1e-12, baseline_avg) * 100
            )
            summary_rows.append(row)

        avg_row = {"train_ratio": ratio_label, "battery": "average"}
        for model_name in ["XNet"] + baseline_models:
            avg_row[f"{model_name}_rmse"] = float(np.mean(scores[model_name]))
        baseline_avg = float(np.mean([avg_row[f"{m}_rmse"] for m in baseline_models]))
        avg_row["baseline_average_rmse"] = baseline_avg
        avg_row["xnet_improve_percent"] = (
            (baseline_avg - avg_row["XNet_rmse"]) / max(1e-12, baseline_avg) * 100
        )
        summary_rows.append(avg_row)

    summary_path = os.path.join(out_dir, "model_comparison_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Model comparison data saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    if candidate_rows:
        print(f"   XNet search CSV: {xnet_search_path}")
    print(f"   Summary CSV: {summary_path}")

def print_model_comparison_summary(results):
    """Print compact multi-model RMSE summary."""
    baseline_models = MODEL_COMPARISON_CONFIG["baseline_models"]
    ratio_results = results.get("ratio_results") or {"single": results}
    print("\nModel Comparison Summary")
    print("=" * 100)
    for ratio_label, ratio_result in ratio_results.items():
        xnet_best = ratio_result["xnet_best"]
        scores = ratio_result["scores"]
        print(f"\n--- Train Ratio: {ratio_label} ---")
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
    # 1. Create result directories (auto-recursive creation, no manual setup needed)
    os.makedirs(CONFIG["fig_out_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_out_dir"], exist_ok=True)
    
    # 2. Check data file integrity (fixed logic: collect all missing files first)
    missing_files = []
    for bat in CONFIG["battery_list"]:
        mat_path = os.path.join(CONFIG["data_dir"], f"{bat}.mat")
        if not os.path.exists(mat_path):
            missing_files.append(mat_path)
    
    # Print missing files and exit
    if missing_files:
        print("Error: Missing the following data files:")
        for f in missing_files:
            print(f"   {f}")
        print("\nPlease place the NASA battery dataset MAT files in the directory:")
        print(f"   {CONFIG['data_dir']}")
        print("Dataset source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        sys.exit(1)

    # 3. Load battery data
    print("Loading battery data...")
    battery_data = load_battery_data(CONFIG["data_dir"], CONFIG["battery_list"])
    print(f"Successfully loaded data for {len(battery_data)} batteries")

    # 4. Run experiments
    print("\nRunning experiments...")
    final_results, pred_results, cycle_results, cauchy_param_histories = run_experiments(
        battery_data, 
        CONFIG["train_ratios"], 
        CONFIG["activation_functions"], 
        CONFIG["model_params"]
    )

    # 5. Save experiment data results (CSV/JSON)
    print("\nSaving experiment data results...")
    save_experiment_results(final_results, CONFIG["data_out_dir"])
    save_cauchy_parameter_history(cauchy_param_histories, CONFIG["data_out_dir"])

    # 6. Generate visualization results (output to result/figure)
    print("\nGenerating visualization plots...")
    visualize_all_batteries(pred_results, cycle_results, battery_data, CONFIG["fig_out_dir"])
    plot_metrics_vs_ratio(CONFIG["train_ratios"], final_results, CONFIG["fig_out_dir"])
    plot_boxplot_metrics(final_results, CONFIG["fig_out_dir"])
    plot_activation_characteristics(CONFIG["fig_out_dir"])
    plot_cauchy_parameter_history(cauchy_param_histories, CONFIG["fig_out_dir"])

    # 7. Print experiment results summary
    print("\nExperiment Results Summary")
    print("=" * 80)
    for metric in ["rmse", "mae", "mape", "r2"]:
        print(f"\n--- {metric.upper()} Metrics ---")
        header = f"{'Activation':<12}" + "".join([f"{int(r*100)}%".center(15) for r in CONFIG["train_ratios"]])
        print(header)
        print("-" * len(header))
        for act in CONFIG["activation_functions"]:
            row = f"{act:<12}"
            for i in range(len(CONFIG["train_ratios"])):
                val = final_results[act][metric][i]
                if metric == "mape":
                    row += f"{val:.2f}%".center(15)
                else:
                    row += f"{val:.4f}".center(15)
            print(row)

    print(f"\nAll visualization plots saved to: {CONFIG['fig_out_dir']}")
    print(f"All data results saved to: {CONFIG['data_out_dir']}")

    if CONFIG["run_model_comparison"]:
        print("\nRunning multi-model comparison...")
        model_comparison_results = run_model_comparison(
            battery_data,
            MODEL_COMPARISON_CONFIG
        )
        save_model_comparison_results(
            model_comparison_results,
            CONFIG["data_out_dir"]
        )
        plot_model_metric_histories(
            model_comparison_results["histories"],
            CONFIG["fig_out_dir"]
        )
        plot_model_predictions_by_battery_ratios(
            battery_data,
            model_comparison_results["ratio_results"],
            CONFIG["fig_out_dir"],
        )
        print_model_comparison_summary(model_comparison_results)

if __name__ == "__main__":
    setup_seed(CONFIG["model_params"]["seed"])
    main()
