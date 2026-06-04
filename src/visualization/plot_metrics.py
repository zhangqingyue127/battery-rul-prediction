import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.5


def _adaptive_ylim(values, metric):
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    margin = (data_max - data_min) * 0.2 if data_max > data_min else max(abs(data_max) * 0.05, 1e-4)
    y_min = data_min - margin
    y_max = data_max + margin
    return y_min, y_max


def _label_positions(endpoints, y_min, y_max, min_gap_fraction=0.055):
    """Return non-overlapping in-axis label y positions for line-end labels."""
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
    for act, _, end_y in ordered:
        positions[act] = min(max(end_y, lower), upper)
        if positions[act] - previous_y < min_gap:
            positions[act] = previous_y + min_gap
        previous_y = positions[act]

    overflow = previous_y - upper
    if overflow > 0:
        for act, _, _ in ordered:
            positions[act] -= overflow

    previous_y = upper + min_gap
    for act, _, _ in reversed(ordered):
        positions[act] = min(positions[act], previous_y - min_gap)
        positions[act] = min(max(positions[act], lower), upper)
        previous_y = positions[act]

    return positions


def _draw_end_labels(ax, endpoints, style_dict, x_label):
    y_min, y_max = ax.get_ylim()
    positions = _label_positions(endpoints, y_min, y_max)
    for act, end_x, end_y in endpoints:
        style = style_dict[act]
        ax.annotate(
            style['label'],
            xy=(end_x, end_y),
            xytext=(x_label, positions[act]),
            textcoords='data',
            color=style['color'],
            fontsize=8,
            fontweight='normal',
            ha='left',
            va='center',
            zorder=3,
            annotation_clip=True,
            arrowprops=dict(
                arrowstyle='-',
                color=style['color'],
                lw=0.5,
                alpha=0.75,
                shrinkA=0,
                shrinkB=3,
            ),
        )


def plot_metrics_vs_ratio(ratios, final_results, out_path="."):
    """Plot metrics vs training data ratio (RMSE/MAE/MAPE)."""
    metrics_config = {
        'rmse': {'name': 'RMSE', 'ylabel': 'Average RMSE (Ah)'},
        'mae':  {'name': 'MAE',  'ylabel': 'Average MAE (Ah)'},
        'mape': {'name': 'MAPE', 'ylabel': 'Average MAPE (%)'},
    }
    metrics_order = ['rmse', 'mae', 'mape']
    labels = ['(a)', '(b)', '(c)']
    activation_list = list(final_results.keys())

    style_dict = {
        'cauchy':      {'color': '#e66d50', 'marker': 'D', 'linewidth': 1.2, 'markersize': 3, 'label': 'Cauchy'},
        'tanh':        {'color': '#e7c66b', 'marker': 'o', 'linewidth': 1.0, 'markersize': 3, 'label': 'Tanh'},
        'relu':        {'color': '#297270', 'marker': 's', 'linewidth': 1.0, 'markersize': 3, 'label': 'ReLU'},
        'gelu':        {'color': '#299d8f', 'marker': 'v', 'linewidth': 1.0, 'markersize': 3, 'label': 'GELU'},
        'leaky_relu':  {'color': '#8ab07c', 'marker': '<', 'linewidth': 1.0, 'markersize': 3, 'label': 'Leaky ReLU'}
    }
    ratio_labels = [f'{int(r * 100)}%' for r in ratios]
    n_ratios = len(ratio_labels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=500)
    axes = axes.flatten()
    plt.subplots_adjust(left=0.06, right=0.94, bottom=0.22, top=0.90, wspace=0.28)

    for idx, (ax, metric) in enumerate(zip(axes, metrics_order)):
        all_valid_values = []
        endpoints = []
        x_ticks_pos = np.arange(n_ratios) * 0.8
        ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(ratio_labels, fontsize=8)
        last_tick = (n_ratios - 1) * 0.8
        x_label = last_tick + 0.24
        ax.set_xlim(-0.2, x_label + 0.55)

        for act in activation_list:
            if act not in final_results:
                continue
            values = final_results[act][metric]
            valid_pairs = [(i, y) for i, y in enumerate(values) if not np.isnan(y)]
            if not valid_pairs:
                continue

            x_plot, y_plot = zip(*valid_pairs)
            x_plot = np.array(x_plot) * 0.8
            all_valid_values.extend(y_plot)
            style = style_dict[act]

            ax.plot(
                x_plot,
                y_plot,
                color=style['color'],
                marker=style['marker'],
                linestyle='-',
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                markeredgecolor='white',
                markeredgewidth=0.3,
                zorder=2,
            )

            endpoints.append((act, x_plot[-1], y_plot[-1]))

        ax.set_xlabel('Percentage of Training Data', fontsize=7, labelpad=8)
        ax.set_ylabel(metrics_config[metric]['ylabel'], fontsize=7, labelpad=6)
        # ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7, zorder=1)
        ax.tick_params(axis='both', labelsize=8, width=0.8, length=3)

        if all_valid_values:
            ax.set_ylim(*_adaptive_ylim(all_valid_values, metric))
            _draw_end_labels(ax, endpoints, style_dict, x_label)

        ax.text(0.5, -0.18, labels[idx], transform=ax.transAxes,
                ha='center', va='top', fontsize=10)

    png_filename = 'metrics_plot_sci_style.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')

    pdf_filename = 'metrics_plot_sci_style.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close()
    print("SCI-style metric plot saved:")
    print(f"  - PNG: {png_filename}")
    print(f"  - PDF: {pdf_filename}")


def plot_boxplot_metrics(final_scores_results, out_path="."):
    """Plot boxplots of metrics to compare activation functions."""
    metrics_config = {
        'rmse': {'name': 'RMSE', 'ylabel': 'RMSE (Ah)'},
        'mae':  {'name': 'MAE',  'ylabel': 'MAE (Ah)'},
        'mape': {'name': 'MAPE', 'ylabel': 'MAPE (%)'},
    }
    metrics_order = ['rmse', 'mae', 'mape']
    labels = ['(a)', '(b)', '(c)']
    activation_list = ['cauchy', 'tanh', 'relu', 'gelu', 'leaky_relu']

    style_dict = {
        'cauchy':      {'color': '#e66d50', 'label': 'Cauchy'},
        'tanh':        {'color': '#e7c66b', 'label': 'Tanh'},
        'relu':        {'color': '#297270', 'label': 'ReLU'},
        'gelu':        {'color': '#299d8f', 'label': 'GELU'},
        'leaky_relu':  {'color': '#8ab07c', 'label': 'Leaky ReLU'}
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=500)
    axes = axes.flatten()
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.92, wspace=0.25)

    for idx, (ax, metric) in enumerate(zip(axes, metrics_order)):
        box_data = [final_scores_results[act][metric] for act in activation_list]

        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            labels=[style_dict[act]['label'] for act in activation_list],
            boxprops=dict(linewidth=1.2),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker='o', markersize=4, markerfacecolor='red', markeredgecolor='white'),
        )

        for patch, act in zip(bp['boxes'], activation_list):
            patch.set_facecolor(style_dict[act]['color'])
            patch.set_alpha(0.7)

        all_values = np.concatenate([np.asarray(values, dtype=float) for values in box_data])
        ax.set_ylim(*_adaptive_ylim(all_values, metric))
        ax.set_ylabel(metrics_config[metric]['ylabel'], fontsize=7, labelpad=6)
        # ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7, zorder=1)
        ax.tick_params(axis='both', labelsize=7, width=0.8, length=3)
        ax.text(0.5, -0.1, labels[idx], transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='normal')

    png_filename = 'metrics_boxplot_sci_style.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')

    pdf_filename = 'metrics_boxplot_sci_style.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close()
    print("\nSCI-style boxplot saved:")
    print(f"  - PNG: {png_filename}")
    print(f"  - PDF: {pdf_filename}")
