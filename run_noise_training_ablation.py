"""
Training-mechanism ablation for periodic noise-robust training.

This script compares:
  1. clean_training
  2. noise_injected_training

Both modes use the same XNet/Cauchy setup and data split. The noise-injected
mode periodically adds Gaussian noise to training input windows, then both
modes are evaluated on clean and noisy test windows.

Usage:
  python run_noise_training_ablation.py
"""

import json
import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.data.loader import load_battery_data, setup_seed
from src.training.noise_ablation import (
    NOISE_TRAINING_ABLATION_CONFIG,
    run_noise_training_ablation_study,
)
from src.visualization.plot_noise_training_ablation import (
    plot_noise_training_convergence,
    plot_noise_training_rmse_bar,
)


BATTERY_LIST = ["B0005", "B0006", "B0007", "B0018"]
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
FIG_OUT_DIR = os.path.join(ROOT_DIR, "result", "figure")
DATA_OUT_DIR = os.path.join(ROOT_DIR, "result", "data_results")


def save_noise_training_ablation_results(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "noise_training_ablation_results.json")
    summary_csv_path = os.path.join(out_dir, "noise_training_ablation_summary.csv")
    history_csv_path = os.path.join(out_dir, "noise_training_ablation_history.csv")
    per_battery_csv_path = os.path.join(
        out_dir,
        "noise_training_ablation_per_battery.csv",
    )

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    summary_rows = []
    history_rows = []
    per_battery_rows = []

    for ratio_label, ratio_data in results.items():
        for mode, mode_data in ratio_data.items():
            for test_key, metrics in mode_data["aggregate"].items():
                summary_rows.append(
                    {
                        "train_ratio": ratio_label,
                        "training_mode": mode,
                        "test_condition": test_key,
                        **metrics,
                    }
                )

            history = mode_data["history"]
            for index, epoch in enumerate(history["epochs"]):
                history_rows.append(
                    {
                        "train_ratio": ratio_label,
                        "training_mode": mode,
                        "epoch": epoch,
                        "train_mse": history["train_mse"][index],
                        "train_mae": history["train_mae"][index],
                        "val_rmse": history["val_rmse"][index],
                        "val_mae": history["val_mae"][index],
                        "noisy_val_rmse": history["noisy_val_rmse"][index],
                        "noisy_val_mae": history["noisy_val_mae"][index],
                    }
                )

            for battery, battery_data in mode_data["per_battery"].items():
                for test_key in ["clean_test", "noisy_test"]:
                    per_battery_rows.append(
                        {
                            "train_ratio": ratio_label,
                            "training_mode": mode,
                            "battery": battery,
                            "test_condition": test_key,
                            **battery_data[test_key],
                        }
                    )

    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)
    pd.DataFrame(history_rows).to_csv(history_csv_path, index=False)
    pd.DataFrame(per_battery_rows).to_csv(per_battery_csv_path, index=False)

    print("Noise training ablation results saved:")
    print(f"  JSON: {json_path}")
    print(f"  Summary CSV: {summary_csv_path}")
    print(f"  History CSV: {history_csv_path}")
    print(f"  Per-battery CSV: {per_battery_csv_path}")


def print_noise_training_summary(results):
    print("\nNoise-Injected Training Ablation Summary - Average RMSE (Ah)")
    print("=" * 86)
    for ratio_label, ratio_data in results.items():
        print(f"\nTrain ratio: {ratio_label}")
        print(f"{'Mode':<26}{'Clean test':>14}{'Noisy test':>14}{'Noisy delta':>14}")
        print("-" * 68)
        for mode, mode_data in ratio_data.items():
            clean_rmse = mode_data["aggregate"]["clean_test"]["rmse"]
            noisy_rmse = mode_data["aggregate"]["noisy_test"]["rmse"]
            print(
                f"{mode:<26}"
                f"{clean_rmse:>14.4f}"
                f"{noisy_rmse:>14.4f}"
                f"{(noisy_rmse - clean_rmse):>14.4f}"
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

    print("\nRunning noise-injected training ablation study...")
    results = run_noise_training_ablation_study(battery_data)

    print("\nSaving results...")
    save_noise_training_ablation_results(results, DATA_OUT_DIR)

    print("\nGenerating plots...")
    plot_noise_training_convergence(results, FIG_OUT_DIR)
    plot_noise_training_rmse_bar(results, FIG_OUT_DIR)

    print_noise_training_summary(results)
    print(f"\nFigures -> {FIG_OUT_DIR}")
    print(f"Data    -> {DATA_OUT_DIR}")


if __name__ == "__main__":
    setup_seed(NOISE_TRAINING_ABLATION_CONFIG["common_params"]["seed"])
    main()
