import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.5


MODEL_STYLES = {
    "XNet": {
        "color": "#e66d50",
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 5,
        "label": "XNet",
    },
    "FC": {
        "color": "#e7c66b",
        "marker": "o",
        "linewidth": 1.2,
        "markersize": 5,
        "label": "FC",
    },
    "LSTM": {
        "color": "#297270",
        "marker": "s",
        "linewidth": 1.5,
        "markersize": 5,
        "label": "LSTM",
    },
    "GRU": {
        "color": "#299d8f",
        "marker": "^",
        "linewidth": 1.2,
        "markersize": 5,
        "label": "GRU",
    },
    "CNN": {
        "color": "#8ab07c",
        "marker": "v",
        "linewidth": 1.5,
        "markersize": 5,
        "label": "CNN",
    },
    "ResNet": {
        "color": "#7d8cc4",
        "marker": "<",
        "linewidth": 1.4,
        "markersize": 5,
        "label": "ResNet",
    },
}


def _style_for(model_name):
    return MODEL_STYLES.get(
        model_name,
        {
            "color": "black",
            "marker": "o",
            "linewidth": 1.2,
            "markersize": 5,
            "label": model_name,
        },
    )


def _label_positions(endpoints, y_min, y_max, min_gap_fraction=0.055):
    if not endpoints:
        return {}

    span = max(y_max - y_min, 1e-12)
    min_gap = span * min_gap_fraction
    padding = span * 0.035
    lower = y_min + padding
    upper = y_max - padding

    ordered = sorted(endpoints, key=lambda item: item[2])
    positions = {}
    previous_y = lower - min_gap
    for name, _, end_y in ordered:
        positions[name] = min(max(end_y, lower), upper)
        if positions[name] - previous_y < min_gap:
            positions[name] = previous_y + min_gap
        previous_y = positions[name]

    overflow = previous_y - upper
    if overflow > 0:
        for name, _, _ in ordered:
            positions[name] -= overflow

    previous_y = upper + min_gap
    for name, _, _ in reversed(ordered):
        positions[name] = min(positions[name], previous_y - min_gap)
        positions[name] = min(max(positions[name], lower), upper)
        previous_y = positions[name]

    return positions


def _draw_end_labels(ax, endpoints, x_label):
    y_min, y_max = ax.get_ylim()
    positions = _label_positions(endpoints, y_min, y_max)
    for model_name, end_x, end_y in endpoints:
        style = _style_for(model_name)
        ax.annotate(
            style["label"],
            xy=(end_x, end_y),
            xytext=(x_label, positions[model_name]),
            textcoords="data",
            color=style["color"],
            fontsize=8,
            ha="left",
            va="center",
            zorder=4,
            annotation_clip=True,
            arrowprops=dict(
                arrowstyle="-",
                color=style["color"],
                lw=0.5,
                alpha=0.75,
                shrinkA=0,
                shrinkB=3,
            ),
        )


def plot_model_metric_histories(histories, out_dir):
    """Plot multi-model training histories in the same SCI style as metric plots."""
    os.makedirs(out_dir, exist_ok=True)
    metrics_config = {
        "train_mse": {"name": "Training MSE", "ylabel": "Training MSE"},
        "val_rmse": {"name": "Validation RMSE", "ylabel": "Validation RMSE (Ah)"},
        "train_mae": {"name": "Training MAE", "ylabel": "Training MAE (Ah)"},
        "val_mae": {"name": "Validation MAE", "ylabel": "Validation MAE (Ah)"},
    }
    metrics_order = ["train_mse", "val_rmse", "train_mae", "val_mae"]
    labels = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.08, right=0.92, bottom=0.12, top=0.92, wspace=0.35, hspace=0.4
    )

    for index, (ax, metric) in enumerate(zip(axes, metrics_order)):
        all_values = []
        endpoints = []
        max_epoch = 0
        for model_name, history in histories.items():
            style = _style_for(model_name)
            x_plot = np.array(history["epochs"])
            y_plot = np.array(history[metric])
            max_epoch = max(max_epoch, int(x_plot[-1]))
            all_values.extend(y_plot.tolist())
            ax.plot(
                x_plot,
                y_plot,
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=style["linewidth"],
                markersize=style["markersize"],
                markeredgecolor="white",
                markeredgewidth=0.3,
                zorder=3 if model_name == "XNet" else 2,
            )
            endpoints.append((model_name, x_plot[-1], y_plot[-1]))

        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            margin = (y_max - y_min) * 0.12 if y_max > y_min else 0.01
            ax.set_ylim(y_min - margin, y_max + margin)
            x_span = max(max_epoch - 1, 1)
            x_label = max_epoch + x_span * 0.055
            ax.set_xlim(1 - x_span * 0.03, max_epoch + x_span * 0.18)
            _draw_end_labels(ax, endpoints, x_label)

        ax.set_xlabel("Training Epochs", fontsize=7, labelpad=8)
        ax.set_ylabel(metrics_config[metric]["ylabel"], fontsize=7, labelpad=6)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
        ax.text(
            0.5,
            -0.18,
            labels[index],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontweight="normal",
        )

    png_path = os.path.join(out_dir, "model_metrics_sci_style.png")
    pdf_path = os.path.join(out_dir, "model_metrics_sci_style.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Model metrics plot saved:")
    print("  - PNG: model_metrics_sci_style.png")
    print("  - PDF: model_metrics_sci_style.pdf")


def plot_model_predictions(
    battery_data,
    predictions,
    out_dir,
    rated_capacity=2.0,
    y_label="Capacity (Ah)",
    true_label="True Capacity",
):
    """Plot four-battery prediction comparison in the project SCI style."""
    os.makedirs(out_dir, exist_ok=True)
    battery_list = list(battery_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.08, right=0.95, bottom=0.12, top=0.92, wspace=0.3, hspace=0.4
    )
    failure_threshold = rated_capacity * 0.7
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

    for index, name in enumerate(battery_list[:4]):
        ax = axes[index]
        cycle_seq = battery_data[name][0]
        true_data = battery_data[name][1]
        ax.plot(
            cycle_seq,
            true_data,
            "k-",
            linewidth=1.5,
            label=true_label,
            zorder=20,
        )

        for model_name, model_predictions in predictions.items():
            style = _style_for(model_name)
            pred_data = np.array(model_predictions[index])
            min_len = min(len(cycle_seq), len(pred_data))
            ax.plot(
                cycle_seq[:min_len],
                pred_data[:min_len],
                color=style["color"],
                linestyle="-",
                linewidth=2.2 if model_name == "XNet" else 1.0,
                label=style["label"],
                zorder=10 if model_name == "XNet" else 2,
            )

        ax.set_xlim(min(cycle_seq) * 0.95, max(cycle_seq) * 1.05)
        y_min = min(true_data) * 0.95 if min(true_data) > 0 else 0
        y_max = max(true_data) * 1.05
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Cycle Number (Real)", fontsize=7, labelpad=6)
        ax.set_ylabel(y_label, fontsize=7, labelpad=6)
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
        ax.text(
            0.5,
            -0.18,
            subplot_labels[index],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.legend(
            loc="upper right",
            fontsize=7,
            frameon=True,
            handlelength=1.0,
            edgecolor="black",
            facecolor="white",
            framealpha=0.9,
        )

    for index in range(len(battery_list), 4):
        axes[index].axis("off")

    png_path = os.path.join(out_dir, "model_prediction_sci_style.png")
    pdf_path = os.path.join(out_dir, "model_prediction_sci_style.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Model prediction plot saved:")
    print("  - PNG: model_prediction_sci_style.png")
    print("  - PDF: model_prediction_sci_style.pdf")


def plot_model_predictions_by_battery_ratios(
    battery_data,
    ratio_results,
    out_dir,
    y_label="Capacity (Ah)",
    true_label="True Capacity",
):
    """Plot one four-ratio multi-model prediction figure for each battery."""
    os.makedirs(out_dir, exist_ok=True)
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    ratio_items = list(ratio_results.items())
    model_names = ["XNet", "FC", "LSTM", "GRU", "CNN", "ResNet"]

    print("\nGenerating multi-model prediction plots by battery...")
    for battery_index, (battery_name, battery_item) in enumerate(battery_data.items()):
        print(f"\n--- Generating model comparison plot for Battery {battery_name} ---")
        cycle_seq = np.asarray(battery_item[0])
        true_data = np.asarray(battery_item[1])
        x_min = float(np.min(cycle_seq))
        x_max = float(np.max(cycle_seq))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
        axes = axes.flatten()
        plt.subplots_adjust(
            left=0.08,
            right=0.95,
            bottom=0.10,
            top=0.94,
            wspace=0.2,
            hspace=0.4,
        )

        for axis_index, (ratio_label, ratio_result) in enumerate(ratio_items[:4]):
            ax = axes[axis_index]
            ratio_value = float(ratio_label.rstrip("%")) / 100.0
            split_idx = int(len(true_data) * ratio_value)
            split_idx = min(max(split_idx, 0), len(true_data) - 1)

            true_segment = true_data[split_idx:]
            true_cycles = np.linspace(x_min, x_max, len(true_segment))
            plotted_values = true_segment.tolist()
            ax.plot(
                true_cycles,
                true_segment,
                "k-",
                linewidth=1.5,
                label=true_label,
                zorder=20,
            )

            predictions = ratio_result["predictions"]
            for model_name in model_names:
                if model_name not in predictions:
                    continue
                style = _style_for(model_name)
                pred_data = np.asarray(predictions[model_name][battery_index])
                pred_segment = pred_data[split_idx:]
                min_len = min(len(true_segment), len(pred_segment))
                if min_len == 0:
                    continue

                pred_segment = pred_segment[:min_len]
                pred_cycles = np.linspace(x_min, x_max, min_len)
                plotted_values.extend(pred_segment.tolist())
                ax.plot(
                    pred_cycles,
                    pred_segment,
                    color=style["color"],
                    linestyle="-",
                    linewidth=2.0 if model_name == "XNet" else 1.0,
                    label=style["label"],
                    zorder=10 if model_name == "XNet" else 2,
                )

            x_margin = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            y_min = min(plotted_values) * 0.95 if min(plotted_values) > 0 else 0
            y_max = max(plotted_values) * 1.05
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Cycle Number", fontsize=7, labelpad=6)
            ax.set_ylabel(y_label, fontsize=7, labelpad=6)
            ax.grid(False)
            ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
            ax.text(
                0.5,
                -0.18,
                subplot_labels[axis_index],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
            ax.legend(
                loc="upper right",
                fontsize=8,
                frameon=True,
                bbox_to_anchor=(1.0, 1.0),
                handlelength=1.0,
                edgecolor="#B0B0B0",
                facecolor="white",
                framealpha=0.9,
            )

        for axis_index in range(len(ratio_items), 4):
            axes[axis_index].axis("off")

        png_name = f"model_prediction_combined_{battery_name}.png"
        pdf_name = f"model_prediction_combined_{battery_name}.pdf"
        plt.savefig(os.path.join(out_dir, png_name), dpi=500, bbox_inches="tight", facecolor="white")
        plt.savefig(os.path.join(out_dir, pdf_name), bbox_inches="tight", facecolor="white", format="pdf")
        plt.close(fig)
        print(f"  - Saved PNG: {png_name}")
        print(f"  - Saved PDF: {pdf_name}")
