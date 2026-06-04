"""
Ablation study entry point for XNet Cauchy Activation Function.

Ablation variants:
  XNet-Full    — Full CAF: λ1·x/(x²+d²) + λ2/(x²+d²), all params trainable
  XNet-NoEven  — Remove even term (λ2), only λ1·x/(x²+d²) remains
  XNet-NoOdd   — Remove odd term (λ1), only λ2/(x²+d²) remains
  XNet-Fixed   — Full CAF with fixed (non-trainable) λ1, λ2, d
  XNet-LN      — Full CAF + LayerNorm before activation

Usage:
  python run_ablation.py
"""

import os
import sys
import json
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from src.data.loader import load_battery_data, setup_seed
from src.training.ablation import run_ablation_study, ABLATION_CONFIG, ABLATION_LABELS
from src.visualization.plot_ablation import (
    plot_ablation_rmse_bar,
    plot_ablation_rmse_vs_ratio,
    plot_ablation_battery_heatmap,
)

BATTERY_LIST = ["B0005", "B0006", "B0007", "B0018"]
DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
FIG_OUT_DIR = os.path.join(ROOT_DIR, "result", "figure")
DATA_OUT_DIR = os.path.join(ROOT_DIR, "result", "data_results")


def save_ablation_results(results, out_dir):
    json_path = os.path.join(out_dir, "ablation_results.json")
    csv_path = os.path.join(out_dir, "ablation_results.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    rows = []
    for ratio_label, ratio_data in results.items():
        for model_name, model_data in ratio_data.items():
            row = {
                "train_ratio": ratio_label,
                "model": model_name,
                "label": ABLATION_LABELS.get(model_name, model_name),
                "avg_rmse": model_data["avg_rmse"],
            }
            for i, score in enumerate(model_data["scores"]):
                row[f"rmse_{BATTERY_LIST[i] if i < len(BATTERY_LIST) else i}"] = score
            rows.append(row)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Ablation results saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


def print_ablation_summary(results):
    ratio_labels = list(results.keys())
    model_names = list(next(iter(results.values())).keys())

    print("\nAblation Study Summary — Average RMSE (Ah)")
    print("=" * 80)
    col_w = 14
    header = f"{'Model':<20}" + "".join(r.center(col_w) for r in ratio_labels)
    print(header)
    print("-" * len(header))

    baseline_rmse = {r: results[r]["XNet-Full"]["avg_rmse"] for r in ratio_labels}
    for model_name in model_names:
        label = ABLATION_LABELS.get(model_name, model_name)
        row = f"{label:<20}"
        for ratio_label in ratio_labels:
            val = results[ratio_label][model_name]["avg_rmse"]
            base = baseline_rmse[ratio_label]
            diff = val - base
            marker = f"(+{diff:.4f})" if diff > 1e-5 else ("  (best)" if abs(diff) < 1e-5 else f"({diff:.4f})")
            row += f"{val:.4f}".center(col_w)
        print(row)

    print("\n  (relative gap vs XNet-Full)")
    print("-" * len(header))
    for model_name in model_names:
        if model_name == "XNet-Full":
            continue
        label = ABLATION_LABELS.get(model_name, model_name)
        row = f"{label:<20}"
        for ratio_label in ratio_labels:
            val = results[ratio_label][model_name]["avg_rmse"]
            base = baseline_rmse[ratio_label]
            pct = (val - base) / max(base, 1e-12) * 100
            sign = "+" if pct >= 0 else ""
            row += f"{sign}{pct:.1f}%".center(col_w)
        print(row)


def main():
    os.makedirs(FIG_OUT_DIR, exist_ok=True)
    os.makedirs(DATA_OUT_DIR, exist_ok=True)

    missing = [
        os.path.join(DATA_DIR, f"{b}.mat")
        for b in BATTERY_LIST
        if not os.path.exists(os.path.join(DATA_DIR, f"{b}.mat"))
    ]
    if missing:
        print("Error: Missing data files:")
        for f in missing:
            print(f"  {f}")
        sys.exit(1)

    print("Loading battery data...")
    battery_data = load_battery_data(DATA_DIR, BATTERY_LIST)
    print(f"Loaded {len(battery_data)} batteries.")

    print("\nRunning XNet CAF ablation study...")
    ablation_results = run_ablation_study(battery_data)

    print("\nSaving results...")
    save_ablation_results(ablation_results, DATA_OUT_DIR)

    print("\nGenerating plots...")
    plot_ablation_rmse_bar(ablation_results, ABLATION_LABELS, FIG_OUT_DIR)
    plot_ablation_rmse_vs_ratio(ablation_results, ABLATION_LABELS, FIG_OUT_DIR)
    plot_ablation_battery_heatmap(
        ablation_results, BATTERY_LIST, ABLATION_LABELS, FIG_OUT_DIR
    )

    print_ablation_summary(ablation_results)
    print(f"\nFigures → {FIG_OUT_DIR}")
    print(f"Data    → {DATA_OUT_DIR}")


if __name__ == "__main__":
    setup_seed(ABLATION_CONFIG["common_params"]["seed"])
    main()
