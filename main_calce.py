import json
import os
import sys

import pandas as pd

from main import convert_numpy_types, save_cauchy_parameter_history
from src.data.calce_loader import DEFAULT_CALCE_BATTERIES, load_calce_battery_data
from src.data.loader import setup_seed
from src.training.model_comparison import run_model_comparison
from src.training.model_comparison_config import FIXED_CAUCHY_PARAMS, MODEL_COMPARISON_CONFIG
from src.training.trainer import run_experiments
from src.visualization.plot_activation import (
    plot_activation_characteristics,
    plot_cauchy_parameter_history,
)
from src.visualization.plot_metrics import plot_boxplot_metrics, plot_metrics_vs_ratio
from src.visualization.plot_model_comparison import (
    plot_model_metric_histories,
    plot_model_predictions,
)
from src.visualization.plot_prediction import visualize_all_batteries


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)


CONFIG = {
    "battery_list": DEFAULT_CALCE_BATTERIES,
    "data_dir": os.path.join(ROOT_DIR, "data", "raw", "CALCE"),
    "fig_out_dir": os.path.join(ROOT_DIR, "result", "calce_figure"),
    "data_out_dir": os.path.join(ROOT_DIR, "result", "calce_data_results"),
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "activation_functions": ["cauchy", "tanh", "relu", "gelu", "leaky_relu"],
    "auto_download": True,
    "clean_data": True,
    "cleaning_config": {
        "hampel_window": 15,
        "hampel_sigma": 3.0,
        "smooth_window": 31,
        "smooth_polyorder": 2,
        "min_capacity": 0.05,
        "max_capacity": 1.25,
    },
    "run_model_comparison": False,
    "model_params": {
        "lr": 0.01,
        "feature_size": 16,
        "hidden_dim": 64,
        "num_layers": 3,
        "weight_decay": 0.001,
        "epochs": 300,
        "seed": 42,
        "rated_capacity": 1.1,
        "use_layer_norm": False,
        "cauchy_params": FIXED_CAUCHY_PARAMS,
    },
}


def save_calce_experiment_results(final_results, save_dir):
    json_path = os.path.join(save_dir, "calce_experiment_metrics.json")
    with open(json_path, "w") as file:
        json.dump(convert_numpy_types(final_results), file, indent=4)

    rows = []
    for activation in CONFIG["activation_functions"]:
        for index, ratio in enumerate(CONFIG["train_ratios"]):
            rows.append({
                "activation_function": activation,
                "train_ratio": f"{int(ratio * 100)}%",
                "rmse": float(final_results[activation]["rmse"][index]),
                "mae": float(final_results[activation]["mae"][index]),
                "mape": float(final_results[activation]["mape"][index]),
                "r2": float(final_results[activation]["r2"][index]),
            })

    csv_path = os.path.join(save_dir, "calce_experiment_metrics.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("CALCE experiment data saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")


def save_calce_model_comparison_results(results, out_dir):
    json_path = os.path.join(out_dir, "calce_model_comparison_results.json")
    csv_path = os.path.join(out_dir, "calce_model_comparison_rmse.csv")

    with open(json_path, "w") as file:
        json.dump(convert_numpy_types(results), file, indent=4)

    baseline_models = MODEL_COMPARISON_CONFIG["baseline_models"]
    rows = []
    for model_name, model_scores in results["scores"].items():
        for battery_name, rmse in zip(CONFIG["battery_list"], model_scores):
            rows.append({
                "model": model_name,
                "battery": battery_name,
                "rmse": float(rmse),
            })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("CALCE model comparison data saved to:")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")


def main():
    os.makedirs(CONFIG["fig_out_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_out_dir"], exist_ok=True)

    print("Loading CALCE battery data...")
    battery_data = load_calce_battery_data(
        CONFIG["data_dir"],
        CONFIG["battery_list"],
        auto_download=CONFIG["auto_download"],
        clean=CONFIG["clean_data"],
        cleaning_config=CONFIG["cleaning_config"],
    )
    print(f"Successfully loaded CALCE data for {len(battery_data)} batteries")
    for name, (_, capacities) in battery_data.items():
        print(f"   {name}: {len(capacities)} cycles")

    print("\nRunning CALCE experiments...")
    final_results, pred_results, cycle_results, cauchy_param_histories = run_experiments(
        battery_data,
        CONFIG["train_ratios"],
        CONFIG["activation_functions"],
        CONFIG["model_params"],
    )

    print("\nSaving CALCE experiment data results...")
    save_calce_experiment_results(final_results, CONFIG["data_out_dir"])
    save_cauchy_parameter_history(cauchy_param_histories, CONFIG["data_out_dir"])

    print("\nGenerating CALCE visualization plots...")
    visualize_all_batteries(pred_results, cycle_results, battery_data, CONFIG["fig_out_dir"])
    plot_metrics_vs_ratio(CONFIG["train_ratios"], final_results, CONFIG["fig_out_dir"])
    plot_boxplot_metrics(final_results, CONFIG["fig_out_dir"])
    plot_activation_characteristics(CONFIG["fig_out_dir"])
    plot_cauchy_parameter_history(cauchy_param_histories, CONFIG["fig_out_dir"])

    print("\nCALCE Experiment Results Summary")
    print("=" * 80)
    for metric in ["rmse", "mae", "mape", "r2"]:
        print(f"\n--- {metric.upper()} Metrics ---")
        header = f"{'Activation':<12}" + "".join(
            [f"{int(r * 100)}%".center(15) for r in CONFIG["train_ratios"]]
        )
        print(header)
        print("-" * len(header))
        for act in CONFIG["activation_functions"]:
            row = f"{act:<12}"
            for index in range(len(CONFIG["train_ratios"])):
                value = final_results[act][metric][index]
                row += (f"{value:.2f}%" if metric == "mape" else f"{value:.4f}").center(15)
            print(row)

    print(f"\nAll CALCE visualization plots saved to: {CONFIG['fig_out_dir']}")
    print(f"All CALCE data results saved to: {CONFIG['data_out_dir']}")

    if CONFIG["run_model_comparison"]:
        print("\nRunning CALCE multi-model comparison...")
        calce_model_config = convert_numpy_types(MODEL_COMPARISON_CONFIG.copy())
        calce_model_config["common_params"]["rated_capacity"] = CONFIG["model_params"]["rated_capacity"]
        model_comparison_results = run_model_comparison(battery_data, calce_model_config)
        save_calce_model_comparison_results(model_comparison_results, CONFIG["data_out_dir"])
        plot_model_metric_histories(model_comparison_results["histories"], CONFIG["fig_out_dir"])
        plot_model_predictions(
            battery_data,
            model_comparison_results["predictions"],
            CONFIG["fig_out_dir"],
            rated_capacity=calce_model_config["common_params"]["rated_capacity"],
        )


if __name__ == "__main__":
    setup_seed(CONFIG["model_params"]["seed"])
    main()
