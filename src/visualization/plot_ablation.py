"""Visualization for XNet CAF ablation study results."""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.5

_ABLATION_COLORS = {
    "XNet-Full": "#e66d50",
    "XNet-NoEven": "#297270",
    "XNet-NoOdd": "#7d8cc4",
    "XNet-Fixed": "#e7c66b",
    "XNet-LN": "#8ab07c",
}

_ABLATION_MARKERS = {
    "XNet-Full": "D",
    "XNet-NoEven": "s",
    "XNet-NoOdd": "^",
    "XNet-Fixed": "o",
    "XNet-LN": "v",
}


def _color(name):
    return _ABLATION_COLORS.get(name, "black")


def _marker(name):
    return _ABLATION_MARKERS.get(name, "o")


def plot_ablation_rmse_bar(results, labels, out_dir):
    """Grouped bar chart: RMSE of each ablation variant per train ratio."""
    os.makedirs(out_dir, exist_ok=True)

    ratio_labels = list(results.keys())
    model_names = list(next(iter(results.values())).keys())
    n_ratios = len(ratio_labels)
    n_models = len(model_names)

    x = np.arange(n_ratios)
    total_width = 0.75
    bar_width = total_width / n_models
    offsets = np.linspace(
        -(total_width - bar_width) / 2,
        (total_width - bar_width) / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=500)
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.92)

    for i, model_name in enumerate(model_names):
        rmse_values = [results[r][model_name]["avg_rmse"] for r in ratio_labels]
        label = labels.get(model_name, model_name)
        bars = ax.bar(
            x + offsets[i],
            rmse_values,
            width=bar_width * 0.92,
            color=_color(model_name),
            label=label,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        for bar, val in zip(bars, rmse_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels, fontsize=9)
    ax.set_xlabel("Training Ratio", fontsize=9, labelpad=6)
    ax.set_ylabel("Average RMSE (Ah)", fontsize=9, labelpad=6)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        fontsize=7.5,
        frameon=True,
        ncol=n_models,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        handlelength=1.2,
        edgecolor="#B0B0B0",
        facecolor="white",
        framealpha=0.9,
    )

    png_path = os.path.join(out_dir, "ablation_rmse_bar.png")
    pdf_path = os.path.join(out_dir, "ablation_rmse_bar.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Ablation bar chart saved:")
    print("  PNG: ablation_rmse_bar.png")
    print("  PDF: ablation_rmse_bar.pdf")


def plot_ablation_rmse_vs_ratio(results, labels, out_dir):
    """Line chart: RMSE vs train ratio for each ablation variant."""
    os.makedirs(out_dir, exist_ok=True)

    ratio_labels = list(results.keys())
    model_names = list(next(iter(results.values())).keys())
    x = np.arange(len(ratio_labels))

    fig, ax = plt.subplots(figsize=(7, 4), dpi=500)
    plt.subplots_adjust(left=0.10, right=0.88, bottom=0.14, top=0.92)

    endpoints = []
    all_values = []
    for model_name in model_names:
        rmse_values = np.array([results[r][model_name]["avg_rmse"] for r in ratio_labels])
        all_values.extend(rmse_values.tolist())
        ax.plot(
            x,
            rmse_values,
            color=_color(model_name),
            marker=_marker(model_name),
            linestyle="-",
            linewidth=2.2 if model_name == "XNet-Full" else 1.3,
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=4 if model_name == "XNet-Full" else 2,
        )
        endpoints.append((model_name, x[-1], float(rmse_values[-1])))

    if all_values:
        y_min, y_max = min(all_values), max(all_values)
        span = max(y_max - y_min, 1e-12)
        margin = span * 0.12
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlim(x[0] - 0.2, x[-1] + 0.65)

        sorted_ends = sorted(endpoints, key=lambda t: t[2])
        prev_y = y_min - span * 0.06
        min_gap = span * 0.06
        label_pos = {}
        for name, _, ey in sorted_ends:
            pos = max(ey, prev_y + min_gap)
            label_pos[name] = pos
            prev_y = pos

        for name, ex, ey in endpoints:
            ax.annotate(
                labels.get(name, name),
                xy=(ex, ey),
                xytext=(ex + 0.12, label_pos[name]),
                textcoords="data",
                color=_color(name),
                fontsize=7.5,
                ha="left",
                va="center",
                zorder=5,
                annotation_clip=False,
                arrowprops=dict(
                    arrowstyle="-",
                    color=_color(name),
                    lw=0.5,
                    alpha=0.7,
                    shrinkA=0,
                    shrinkB=3,
                ),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ratio_labels, fontsize=9)
    ax.set_xlabel("Training Ratio", fontsize=9, labelpad=6)
    ax.set_ylabel("Average RMSE (Ah)", fontsize=9, labelpad=6)
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    ax.grid(False)
    ax.text(0.5, -0.16, "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=10)

    png_path = os.path.join(out_dir, "ablation_rmse_line.png")
    pdf_path = os.path.join(out_dir, "ablation_rmse_line.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Ablation line chart saved:")
    print("  PNG: ablation_rmse_line.png")
    print("  PDF: ablation_rmse_line.pdf")


def plot_ablation_battery_heatmap(results, battery_list, labels, out_dir):
    """Combined 2x2 heatmap showing per-battery RMSE for each train ratio."""
    os.makedirs(out_dir, exist_ok=True)

    ratio_labels = list(results.keys())
    model_names = list(next(iter(results.values())).keys())
    n_models = len(model_names)
    n_batteries = len(battery_list)

    matrices = {}
    all_values = []
    for ratio_label in ratio_labels:
        ratio_data = results[ratio_label]
        matrix = np.zeros((n_models, n_batteries))
        for i, model_name in enumerate(model_names):
            scores = ratio_data[model_name]["scores"]
            for j in range(min(n_batteries, len(scores))):
                matrix[i, j] = scores[j]
        matrices[ratio_label] = matrix
        all_values.extend(matrix.reshape(-1).tolist())

    vmin = float(np.min(all_values)) if all_values else 0.0
    vmax = float(np.max(all_values)) if all_values else 1.0
    threshold = vmin + (vmax - vmin) * 0.62

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), dpi=500)
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.14,
        right=0.90,
        bottom=0.10,
        top=0.92,
        wspace=0.18,
        hspace=0.40,
    )
    subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
    im = None

    for axis_index, (ax, ratio_label) in enumerate(zip(axes, ratio_labels)):
        matrix = matrices[ratio_label]
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=vmin, vmax=vmax)

        for i in range(n_models):
            for j in range(n_batteries):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if matrix[i, j] >= threshold else "#1f2933",
                )

        ax.set_xticks(range(n_batteries))
        ax.set_xticklabels(battery_list, fontsize=8)
        ax.set_yticks(range(n_models))
        if axis_index % 2 == 0:
            ax.set_yticklabels([labels.get(m, m) for m in model_names], fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Battery", fontsize=8, labelpad=5)
        ax.set_title(f"Train Ratio {ratio_label}", fontsize=9, pad=8)
        ax.tick_params(axis="both", length=0)
        ax.text(
            0.5,
            -0.17,
            subplot_labels[axis_index],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    for axis_index in range(len(ratio_labels), len(axes)):
        axes[axis_index].axis("off")

    if im is not None:
        cbar_ax = fig.add_axes([0.925, 0.18, 0.018, 0.66])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=7, width=0.6, length=3)
        cbar.set_label("RMSE (Ah)", fontsize=8)

    png_path = os.path.join(out_dir, "ablation_heatmap_combined.png")
    pdf_path = os.path.join(out_dir, "ablation_heatmap_combined.pdf")
    plt.savefig(png_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white", format="pdf")
    plt.close(fig)
    print("Combined ablation heatmap saved:")
    print("  PNG: ablation_heatmap_combined.png")
    print("  PDF: ablation_heatmap_combined.pdf")
