"""Visualization helpers for noise robustness ablation results."""

import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.5

_ACTIVATION_COLORS = {
    "cauchy": "#e66d50",
    "tanh": "#297270",
    "relu": "#7d8cc4",
    "gelu": "#e7c66b",
    "leaky_relu": "#8ab07c",
}

_ACTIVATION_MARKERS = {
    "cauchy": "D",
    "tanh": "s",
    "relu": "^",
    "gelu": "o",
    "leaky_relu": "v",
}


def _color(name):
    return _ACTIVATION_COLORS.get(name, "black")


def _marker(name):
    return _ACTIVATION_MARKERS.get(name, "o")


def _label(name):
    return {
        "cauchy": "Cauchy",
        "tanh": "Tanh",
        "relu": "ReLU",
        "gelu": "GELU",
        "leaky_relu": "Leaky ReLU",
    }.get(name, name)


def _noise_keys_for_ratio(ratio_data):
    first_activation = next(iter(ratio_data.values()))
    return list(first_activation["aggregate"].keys())


def plot_noise_rmse_vs_level(results, out_dir):
    """Line charts: average RMSE vs Gaussian noise level for each train ratio."""
    os.makedirs(out_dir, exist_ok=True)

    for ratio_label, ratio_data in results.items():
        noise_keys = _noise_keys_for_ratio(ratio_data)
        noise_values = [float(key) for key in noise_keys]

        fig, ax = plt.subplots(figsize=(7, 4), dpi=500)
        plt.subplots_adjust(left=0.11, right=0.98, bottom=0.16, top=0.88)

        for activation, activation_data in ratio_data.items():
            rmse_values = [
                activation_data["aggregate"][key]["rmse"] for key in noise_keys
            ]
            ax.plot(
                noise_values,
                rmse_values,
                color=_color(activation),
                marker=_marker(activation),
                label=_label(activation),
                linewidth=2.2 if activation == "cauchy" else 1.4,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=0.4,
            )

        ax.set_xlabel("Gaussian Noise Level", fontsize=9, labelpad=6)
        ax.set_ylabel("Average RMSE (Ah)", fontsize=9, labelpad=6)
        ax.set_title(f"Noise Robustness - Train Ratio {ratio_label}", fontsize=10, pad=8)
        ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
        ax.legend(
            fontsize=7.5,
            frameon=True,
            ncol=3,
            loc="upper left",
            edgecolor="#B0B0B0",
            facecolor="white",
            framealpha=0.9,
        )

        safe_ratio = ratio_label.replace("%", "pct")
        png_path = os.path.join(out_dir, f"noise_rmse_line_{safe_ratio}.png")
        pdf_path = os.path.join(out_dir, f"noise_rmse_line_{safe_ratio}.pdf")
        plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
        plt.close(fig)
        print(f"  Noise RMSE line saved: noise_rmse_line_{safe_ratio}.png / .pdf")


def plot_noise_degradation_bar(results, out_dir):
    """Grouped bars: RMSE increase from clean to the largest noise level."""
    os.makedirs(out_dir, exist_ok=True)

    ratio_labels = list(results.keys())
    activations = list(next(iter(results.values())).keys())
    x = np.arange(len(ratio_labels))
    total_width = 0.75
    bar_width = total_width / len(activations)
    offsets = np.linspace(
        -(total_width - bar_width) / 2,
        (total_width - bar_width) / 2,
        len(activations),
    )

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=500)
    plt.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.88)

    for idx, activation in enumerate(activations):
        values = []
        for ratio_label in ratio_labels:
            ratio_data = results[ratio_label][activation]["aggregate"]
            noise_keys = list(ratio_data.keys())
            clean_rmse = ratio_data[noise_keys[0]]["rmse"]
            noisy_rmse = ratio_data[noise_keys[-1]]["rmse"]
            values.append(noisy_rmse - clean_rmse)

        ax.bar(
            x + offsets[idx],
            values,
            width=bar_width * 0.92,
            color=_color(activation),
            label=_label(activation),
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels, fontsize=9)
    ax.set_xlabel("Training Ratio", fontsize=9, labelpad=6)
    ax.set_ylabel("RMSE Increase at Max Noise (Ah)", fontsize=9, labelpad=6)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.legend(
        fontsize=7.5,
        frameon=True,
        ncol=len(activations),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        handlelength=1.2,
        edgecolor="#B0B0B0",
        facecolor="white",
        framealpha=0.9,
    )

    png_path = os.path.join(out_dir, "noise_rmse_degradation_bar.png")
    pdf_path = os.path.join(out_dir, "noise_rmse_degradation_bar.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Noise degradation bar chart saved:")
    print("  PNG: noise_rmse_degradation_bar.png")
    print("  PDF: noise_rmse_degradation_bar.pdf")


def plot_noise_rmse_heatmap(results, out_dir):
    """Heatmaps: activation by noise-level average RMSE for each train ratio."""
    os.makedirs(out_dir, exist_ok=True)

    for ratio_label, ratio_data in results.items():
        activations = list(ratio_data.keys())
        noise_keys = _noise_keys_for_ratio(ratio_data)
        matrix = np.zeros((len(activations), len(noise_keys)))

        for i, activation in enumerate(activations):
            for j, noise_key in enumerate(noise_keys):
                matrix[i, j] = ratio_data[activation]["aggregate"][noise_key]["rmse"]

        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=500)
        plt.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.88)

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Average RMSE (Ah)", fontsize=7)

        threshold = matrix.min() + (matrix.max() - matrix.min()) * 0.65
        for i in range(len(activations)):
            for j in range(len(noise_keys)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[i, j] >= threshold else "black",
                )

        ax.set_xticks(range(len(noise_keys)))
        ax.set_xticklabels(noise_keys, fontsize=8)
        ax.set_yticks(range(len(activations)))
        ax.set_yticklabels([_label(name) for name in activations], fontsize=8)
        ax.set_xlabel("Gaussian Noise Level", fontsize=8, labelpad=5)
        ax.set_title(f"Noise Robustness Heatmap - Train Ratio {ratio_label}", fontsize=9, pad=8)

        safe_ratio = ratio_label.replace("%", "pct")
        png_path = os.path.join(out_dir, f"noise_rmse_heatmap_{safe_ratio}.png")
        pdf_path = os.path.join(out_dir, f"noise_rmse_heatmap_{safe_ratio}.pdf")
        plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
        plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
        plt.close(fig)
        print(f"  Noise heatmap saved: noise_rmse_heatmap_{safe_ratio}.png / .pdf")
