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
from src.visualization.plot_activation import plot_activation_characteristics
from src.visualization.plot_metrics import plot_metrics_vs_ratio, plot_boxplot_metrics
from src.visualization.plot_prediction import visualize_all_batteries

# ===================== Experiment Configuration =====================
CONFIG = {
    "battery_list": ["B0005", "B0006", "B0007", "B0018"],
    "data_dir": os.path.join(ROOT_DIR, "data", "raw"),
    "fig_out_dir": os.path.join(ROOT_DIR, "result", "figure"),
    "data_out_dir": os.path.join(ROOT_DIR, "result", "data_results"),
    "train_ratios": [0.4, 0.5, 0.6, 0.7],
    "activation_functions": ["cauchy", "tanh", "relu", "gelu", "leaky_relu"],
    "model_params": {
        "lr": 0.01,
        "feature_size": 16,
        "hidden_dim": 64,
        "num_layers": 3,
        "weight_decay": 0.001,
        "epochs": 300,
        "seed": 42,
        "rated_capacity": 2.0,
        "cauchy_params": {"lambda1": 0.7, "lambda2": 0.1, "d": 0.5}
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
    final_results, pred_results, cycle_results = run_experiments(
        battery_data, 
        CONFIG["train_ratios"], 
        CONFIG["activation_functions"], 
        CONFIG["model_params"]
    )

    # 5. Save experiment data results (CSV/JSON)
    print("\nSaving experiment data results...")
    save_experiment_results(final_results, CONFIG["data_out_dir"])

    # 6. Generate visualization results (output to result/figure)
    print("\nGenerating visualization plots...")
    visualize_all_batteries(pred_results, cycle_results, battery_data, CONFIG["fig_out_dir"])
    plot_metrics_vs_ratio(CONFIG["train_ratios"], final_results, CONFIG["fig_out_dir"])
    plot_boxplot_metrics(final_results, CONFIG["fig_out_dir"])
    plot_activation_characteristics(CONFIG["fig_out_dir"])

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

if __name__ == "__main__":
    setup_seed(CONFIG["model_params"]["seed"])
    main()