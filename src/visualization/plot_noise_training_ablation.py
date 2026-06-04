"""Visualization for clean vs. noise-injected training ablation."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.5

_MODE_STYLES = {
    "clean_training": {
        "color": "#297270",
        "marker": "s",
        "label": "Clean Training",
        "linewidth": 1.5,
    },
    "noise_injected_training": {
        "color": "#e66d50",
        "marker": "D",
        "label": "Noise-Injected Training",
        "linewidth": 2.1,
    },
}


def _style(mode):
    return _MODE_STYLES.get(
        mode,
        {"color": "black", "marker": "o", "label": mode, "linewidth": 1.4},
    )


def plot_noise_training_convergence(results, out_dir):
    """Plot convergence curves for each train ratio."""
    os.makedirs(out_dir, exist_ok=True)

    metrics = [
        ("train_mse", "Training MSE"),
        ("val_rmse", "Clean Validation RMSE (Ah)"),
        ("noisy_val_rmse", "Noisy Validation RMSE (Ah)"),
        ("val_mae", "Clean Validation MAE (Ah)"),
    ]
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ratio_label, ratio_data in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
        axes = axes.flatten()
        plt.subplots_adjust(
            left=0.08,
            right=0.97,
            bottom=0.10,
            top=0.92,
            wspace=0.30,
            hspace=0.38,
        )

        for index, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
            all_values = []
            for mode, mode_data in ratio_data.items():
                history = mode_data["history"]
                x = np.asarray(history["epochs"])
                y = np.asarray(history[metric])
                all_values.extend(y.tolist())
                style = _style(mode)
                ax.plot(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle="-",
                    linewidth=style["linewidth"],
                    markersize=4.5,
                    markeredgecolor="white",
                    markeredgewidth=0.4,
                    label=style["label"],
                    zorder=3 if mode == "noise_injected_training" else 2,
                )

            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                margin = (y_max - y_min) * 0.12 if y_max > y_min else 0.01
                ax.set_ylim(y_min - margin, y_max + margin)

            ax.set_xlabel("Training Epochs", fontsize=8, labelpad=6)
            ax.set_ylabel(ylabel, fontsize=8, labelpad=6)
            ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
            ax.text(
                0.5,
                -0.18,
                subplot_labels[index],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
            if index == 0:
                ax.legend(
                    fontsize=8,
                    frameon=True,
                    loc="upper right",
                    edgecolor="#B0B0B0",
                    facecolor="white",
                    framealpha=0.9,
                )

        safe_ratio = ratio_label.replace("%", "pct")
        png_path = os.path.join(
            out_dir,
            f"noise_training_convergence_{safe_ratio}.png",
        )
        pdf_path = os.path.join(
            out_dir,
            f"noise_training_convergence_{safe_ratio}.pdf",
        )
        plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
        plt.close(fig)
        print(
            f"  Noise training convergence saved: "
            f"noise_training_convergence_{safe_ratio}.png / .pdf"
        )


def plot_noise_training_rmse_bar(results, out_dir):
    """Grouped bars for clean/noisy test RMSE after each training mode."""
    os.makedirs(out_dir, exist_ok=True)

    ratio_labels = list(results.keys())
    modes = list(next(iter(results.values())).keys())
    x = np.arange(len(ratio_labels))
    width = 0.34
    offsets = np.linspace(-width / 2, width / 2, len(modes))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=500)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.86, wspace=0.25)

    for axis_index, (ax, test_key, title) in enumerate(
        [
            (axes[0], "clean_test", "Clean Test"),
            (axes[1], "noisy_test", "Noisy Test"),
        ]
    ):
        for mode_index, mode in enumerate(modes):
            style = _style(mode)
            values = [
                results[ratio][mode]["aggregate"][test_key]["rmse"]
                for ratio in ratio_labels
            ]
            ax.bar(
                x + offsets[mode_index],
                values,
                width=width * 0.92,
                color=style["color"],
                label=style["label"],
                alpha=0.88,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ratio_labels, fontsize=8)
        ax.set_xlabel("Training Ratio", fontsize=8, labelpad=6)
        ax.set_ylabel("Average RMSE (Ah)", fontsize=8, labelpad=6)
        ax.set_title(title, fontsize=9, pad=8)
        ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45, zorder=0)
        ax.text(
            0.5,
            -0.20,
            "(a)" if axis_index == 0 else "(b)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        if axis_index == 0:
            ax.legend(
                fontsize=8,
                frameon=True,
                loc="upper center",
                bbox_to_anchor=(1.1, 1.20),
                ncol=2,
                edgecolor="#B0B0B0",
                facecolor="white",
                framealpha=0.9,
            )

    png_path = os.path.join(out_dir, "noise_training_rmse_bar.png")
    pdf_path = os.path.join(out_dir, "noise_training_rmse_bar.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Noise training RMSE bar chart saved:")
    print("  PNG: noise_training_rmse_bar.png")
    print("  PDF: noise_training_rmse_bar.pdf")
