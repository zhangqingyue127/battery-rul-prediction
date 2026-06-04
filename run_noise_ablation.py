"""
Noise robustness ablation entry point for XNet activation functions.

The experiment trains each activation on clean battery data and evaluates the
best checkpoint under Gaussian noise injected into the test input windows.

Usage:
  python run_noise_ablation.py
"""

import json
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.data.loader import load_battery_data, setup_seed
from src.training.noise_ablation import (
    NOISE_ABLATION_CONFIG,
    run_noise_ablation_study,
)
from src.visualization.plot_noise_ablation import (
    plot_noise_degradation_bar,
    plot_noise_rmse_heatmap,
    plot_noise_rmse_vs_level,
)


BATTERY_LIST = ["B0005", "B0006", "B0007", "B0018"]
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
FIG_OUT_DIR = os.path.join(ROOT_DIR, "result", "figure")
DATA_OUT_DIR = os.path.join(ROOT_DIR, "result", "data_results")


def save_noise_ablation_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "noise_ablation_results.json")
    csv_path = os.path.join(out_dir, "noise_ablation_results.csv")
    per_battery_csv_path = os.path.join(
        out_dir,
        "noise_ablation_per_battery_results.csv",
    )

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    aggregate_rows = []
    per_battery_rows = []
    for ratio_label, ratio_data in results.items():
        for activation, activation_data in ratio_data.items():
            for noise_level, metrics in activation_data["aggregate"].items():
                aggregate_rows.append(
                    {
                        "train_ratio": ratio_label,
                        "activation": activation,
                        "noise_level": noise_level,
                        **metrics,
                    }
                )

            for battery, battery_data in activation_data["per_battery"].items():
                for noise_level, metrics in battery_data.items():
                    per_battery_rows.append(
                        {
                            "train_ratio": ratio_label,
                            "activation": activation,
                            "battery": battery,
                            "noise_level": noise_level,
                            **metrics,
                        }
                    )

    pd.DataFrame(aggregate_rows).to_csv(csv_path, index=False)
    pd.DataFrame(per_battery_rows).to_csv(per_battery_csv_path, index=False)

    print("Noise ablation results saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"  Per-battery CSV: {per_battery_csv_path}")


def print_noise_ablation_summary(results):
    print("\nNoise Robustness Summary - Average RMSE (Ah)")
    print("=" * 90)
    for ratio_label, ratio_data in results.items():
        noise_keys = list(next(iter(ratio_data.values()))["aggregate"].keys())
        clean_key = noise_keys[0]
        max_noise_key = noise_keys[-1]
        print(f"\nTrain ratio: {ratio_label}")
        print(f"{'Activation':<14}{'Clean':>12}{'Max noise':>14}{'Delta':>12}")
        print("-" * 52)
        for activation, activation_data in ratio_data.items():
            clean_rmse = activation_data["aggregate"][clean_key]["rmse"]
            noisy_rmse = activation_data["aggregate"][max_noise_key]["rmse"]
            print(
                f"{activation:<14}"
                f"{clean_rmse:>12.4f}"
                f"{noisy_rmse:>14.4f}"
                f"{(noisy_rmse - clean_rmse):>12.4f}"
            )


def main():
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)

    missing = [
        os.path.join(DATA_DIR, f"{battery}.mat")
        for battery in BATTERY_LIST
        if not os.path.exists(os.path.join(DATA_DIR, f"{battery}.mat"))
    ]
    if missing:
        print("Error: Missing data files:")
        for file_path in missing:
            print(f"  {file_path}")
        sys.exit(1)

    print("Loading battery data...")
    battery_data = load_battery_data(DATA_DIR, BATTERY_LIST)
    print(f"Loaded {len(battery_data)} batteries.")

    print("\nRunning noise robustness ablation study...")
    results = run_noise_ablation_study(battery_data)

    print("\nSaving results...")
    save_noise_ablation_results(results, DATA_OUT_DIR)

    print("\nGenerating plots...")
    plot_noise_rmse_vs_level(results, FIG_OUT_DIR)
    plot_noise_degradation_bar(results, FIG_OUT_DIR)
    plot_noise_rmse_heatmap(results, FIG_OUT_DIR)

    print_noise_ablation_summary(results)
    print(f"\nFigures -> {FIG_OUT_DIR}")
    print(f"Data    -> {DATA_OUT_DIR}")


if __name__ == "__main__":
    setup_seed(NOISE_ABLATION_CONFIG["common_params"]["seed"])
    main()
