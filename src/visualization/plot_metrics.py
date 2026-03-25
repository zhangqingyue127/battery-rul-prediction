import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.5
def plot_metrics_vs_ratio(ratios, final_results, out_path="."):
    """Plot metrics vs training data ratio (RMSE/MAE/MAPE/R²)"""
    # Config
    metrics_config = {
        'rmse': {'name': 'RMSE', 'ylim': (0.024, 0.032), 'ylabel': 'Average RMSE (Ah)'},
        'mae':  {'name': 'MAE',  'ylim': (0.016, 0.024), 'ylabel': 'Average MAE (Ah)'},
        'mape': {'name': 'MAPE', 'ylim': (1.1, 1.55),   'ylabel': 'Average MAPE (%)'},
        'r2':   {'name': '$R^2$','ylim': None,          'ylabel': 'Average $R^2$'}
    }
    metrics_order = ['rmse', 'mae', 'mape', 'r2']
    labels = ['(a)', '(b)', '(c)', '(d)']
    activation_list = list(final_results.keys())

    # Style
    style_dict = {
        'cauchy':      {'color': '#e66d50', 'marker': 'D', 'linewidth': 1.2, 'markersize': 3, 'label': 'Cauchy'},
        'tanh':        {'color': '#e7c66b', 'marker': 'o', 'linewidth': 1.0, 'markersize': 3, 'label': 'Tanh'},
        'relu':        {'color': '#297270', 'marker': 's', 'linewidth': 1.0, 'markersize': 3, 'label': 'ReLU'},
        'gelu':        {'color': '#299d8f', 'marker': 'v', 'linewidth': 1.0, 'markersize': 3, 'label': 'GELU'},
        'leaky_relu':  {'color': '#8ab07c', 'marker': '<', 'linewidth': 1.0, 'markersize': 3, 'label': 'Leaky ReLU'}
    }
    ratio_labels = [f'{int(r * 100)}%' for r in ratios]
    n_ratios = len(ratio_labels)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
    axes = axes.flatten()

    # Adjust subplot spacing (reserve more space for x-axis)
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.12, top=0.92, wspace=0.2, hspace=0.4)

    # Plot each subplot
    for idx, (ax, metric) in enumerate(zip(axes, metrics_order)):
        all_valid_values = []
        
        # Shorten x-axis tick interval (default 1, changed to 0.8x)
        x_ticks_pos = np.arange(n_ratios) * 0.8  # Shortened tick positions
        ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(ratio_labels, fontsize=8)

        # Extend x-axis range (reserve 20% more space on the right for annotations)
        x_max = (n_ratios - 1) * 0.8 * 1.2
        ax.set_xlim(-0.2, x_max)

        # Plot curves + annotations
        for act in activation_list:
            if act in final_results:
                values = final_results[act][metric]
                valid_pairs = [(i, y) for i, y in enumerate(values) if not np.isnan(y)]  # Use index instead of text
                if not valid_pairs:
                    continue
                x_plot, y_plot = zip(*valid_pairs)
                
                # Convert to shortened x-axis coordinates
                x_plot = np.array(x_plot) * 0.8
                all_valid_values.extend(y_plot)

                # Plot curve
                ax.plot(x_plot, y_plot,
                        color=style_dict[act]['color'],
                        marker=style_dict[act]['marker'],
                        linestyle='-',
                        linewidth=style_dict[act]['linewidth'],
                        markersize=style_dict[act]['markersize'],
                        markeredgecolor='white',
                        markeredgewidth=0.3,
                        zorder=2)

                # Core modification 1: Annotation position control (Cauchy in R² subplot shifted up)
                last_x = x_plot[-1]
                last_y = y_plot[-1]
                
                # Annotation offset
                if metric == 'r2' and act == 'cauchy':
                    annotate_offset = (3, 5)  # R2-Cauchy shifted up
                elif metric == 'r2' and act == 'gelu':
                    annotate_offset = (3, -5)  # R2-GELU shifted down
                else:
                    annotate_offset = (3, 0)      # Other annotations remain in original position
                
                # Core modification 2: All annotation text color changed to match line color
                ax.annotate(
                    style_dict[act]['label'],
                    xy=(last_x, last_y),
                    xytext=annotate_offset,  # Dynamic offset
                    textcoords='offset points',
                    color=style_dict[act]['color'],
                    fontsize=8,
                    fontweight='normal',
                    ha='left',
                    va='center',
                    zorder=3,
                    annotation_clip=False
                )

        # Subplot style
        ax.set_xlabel('Percentage of Training Data', fontsize=7, labelpad=8)  # Increase labelpad to avoid overlap
        ax.set_ylabel(metrics_config[metric]['ylabel'], fontsize=7, labelpad=6)
        # ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7, zorder=1)
        ax.tick_params(axis='both', labelsize=8, width=0.8, length=3)
        
        # Y-axis range (only for R²)
        if metric == 'r2' and len(all_valid_values) > 0:
            y_min = max(0.9, np.min(all_valid_values) - 0.005)
            y_max = min(1.0, np.max(all_valid_values) + 0.005)
            if y_max - y_min < 0.002:
                y_min -= 0.001
                y_max += 0.001
            ax.set_ylim(y_min, y_max)
        else:
            if metrics_config[metric]['ylim']:
                ax.set_ylim(metrics_config[metric]['ylim'])

        # Subplot label (centered at the bottom)
        ax.text(0.5, -0.18, labels[idx], transform=ax.transAxes,
                ha='center', va='top', fontsize=10)

    # Save as PNG format (original)
    png_filename = 'metrics_plot_sci_style.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')
    
    # Added: Save as PDF format
    pdf_filename = 'metrics_plot_sci_style.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')
    
    plt.close()
    print(f"SCI-style metric plot saved:")
    print(f"  - PNG: {png_filename}")
    print(f"  - PDF: {pdf_filename}")

def plot_boxplot_metrics(final_scores_results, out_path="."):
    """
    Plot boxplots of 4 metrics (RMSE/MAE/MAPE/R²) to compare 5 activation functions
    Input: final_scores_results — complete metric list of each activation function on 4 batteries
    Output: 2×2 subplot boxplots (saved as PNG/PDF)
    """
    # Metric configuration (consistent with original plot_metrics_vs_ratio)
    metrics_config = {
        'rmse': {'name': 'RMSE', 'ylabel': 'RMSE (Ah)', 'ylim': (0.024, 0.032)},
        'mae':  {'name': 'MAE',  'ylabel': 'MAE (Ah)',  'ylim': (0.016, 0.024)},
        'mape': {'name': 'MAPE', 'ylabel': 'MAPE (%)',   'ylim': (1.1, 1.55)},
        'r2':   {'name': '$R^2$','ylabel': '$R^2$',      'ylim': None}
    }
    metrics_order = ['rmse', 'mae', 'mape', 'r2']
    labels = ['(a)', '(b)', '(c)', '(d)']  # Subplot labels
    activation_list = ['cauchy', 'tanh', 'relu', 'gelu', 'leaky_relu']

    # SCI color scheme (exactly the same as original code)
    style_dict = {
        'cauchy':      {'color': '#e66d50', 'label': 'Cauchy'},
        'tanh':        {'color': '#e7c66b', 'label': 'Tanh'},
        'relu':        {'color': '#297270', 'label': 'ReLU'},
        'gelu':        {'color': '#299d8f', 'label': 'GELU'},
        'leaky_relu':  {'color': '#8ab07c', 'label': 'Leaky ReLU'}
    }

    # Create 2×2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
    axes = axes.flatten()

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.10, top=0.94, wspace=0.2, hspace=0.22)

    # Plot boxplot for each metric
    for idx, (ax, metric) in enumerate(zip(axes, metrics_order)):
        # Collect complete metric data of each activation function on 4 batteries (not mean, but real values of each battery)
        box_data = []
        for act in activation_list:
            # Extract all battery metrics of the activation function under all training ratios from final_scores_results
            all_values = []
            for ratio_values in final_scores_results[act][metric]:
                if metric == 'mape':
                    all_values.append(ratio_values)  # MAPE is percentage
                else:
                    all_values.append(ratio_values)
            box_data.append(all_values)

        # Plot boxplot
        bp = ax.boxplot(
            box_data,
            patch_artist=True,
            labels=[style_dict[act]['label'] for act in activation_list],
            boxprops=dict(linewidth=1.2),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker='o', markersize=4, markerfacecolor='red', markeredgecolor='white')
        )

        # Color the boxes (consistent with original code color scheme)
        for patch, act in zip(bp['boxes'], activation_list):
            patch.set_facecolor(style_dict[act]['color'])
            patch.set_alpha(0.7)

        # Subplot style
        ax.set_ylabel(metrics_config[metric]['ylabel'], fontsize=7, labelpad=6)
        # ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7, zorder=1)
        ax.tick_params(axis='both', labelsize=7, width=0.8, length=3)

        # Y-axis range (R² adaptive)
        if metric == 'r2':
            all_r2_values = []
            for act in activation_list:
                all_r2_values.extend(final_scores_results[act]['r2'])
            if all_r2_values:
                y_min = max(0.9, np.min(all_r2_values) - 0.005)
                y_max = min(1.0, np.max(all_r2_values) + 0.005)
                if y_max - y_min < 0.002:
                    y_min -= 0.001
                    y_max += 0.001
                ax.set_ylim(y_min, y_max)
        else:
            if metrics_config[metric]['ylim']:
                ax.set_ylim(metrics_config[metric]['ylim'])

        # Subplot label (centered at the bottom, consistent with original code)
        ax.text(0.5, -0.1, labels[idx], transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='normal')

    # Save as PNG and PDF (consistent with original plot_metrics_vs_ratio)
    png_filename = 'metrics_boxplot_sci_style.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')

    pdf_filename = 'metrics_boxplot_sci_style.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close()
    print(f"\nSCI-style boxplot saved:")
    print(f"  - PNG: {png_filename}")
    print(f"  - PDF: {pdf_filename}")