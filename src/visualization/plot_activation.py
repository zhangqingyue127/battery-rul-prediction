import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.5

def plot_activation_characteristics(out_path="."):
    """
    Plot characteristic curves of five activation functions: Cauchy / ReLU / Tanh / GELU / Leaky ReLU
    Input: None (directly calculate the output of each activation function)
    Output: Single figure (saved as PNG/PDF)
    """
    # 1. Generate input x
    x = np.linspace(-3, 3, 200)

    # 2. Calculate output of each activation function (consistent with the definition in your paper)
    # Cauchy activation (version in your code: λ1=0.7, λ2=0.1, d=0.5)
    lambda1, lambda2, d = 0.7, 0.1, 0.5
    y_cauchy = lambda1 * x / (x**2 + d**2) + lambda2 / (x**2 + d**2)

    # ReLU
    y_relu = np.maximum(0, x)

    # Tanh
    y_tanh = np.tanh(x)

    # GELU
    y_gelu = x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) / 2

    # Leaky ReLU (negative_slope=0.01)
    y_leaky_relu = np.where(x > 0, x, 0.01 * x)

    # 3. Plot configuration (consistent with the color scheme of your example figure)
    style_dict = {
        'cauchy':      {'color': '#c82423', 'label': 'Cauchy',      'linewidth': 3},
        'relu':        {'color': '#f2b05e', 'label': 'ReLU',        'linewidth': 2},
        'tanh':        {'color': '#367bc1', 'label': 'Tanh',        'linewidth': 2},
        'gelu':        {'color': '#a8d5d7', 'label': 'GELU',        'linewidth': 2},
        'leaky_relu':  {'color': '#8ab07c', 'label': 'Leaky ReLU', 'linewidth': 2}
    }

    fig, ax = plt.subplots(figsize=(12, 8), dpi=500)
    ax.plot(x, y_cauchy,      color=style_dict['cauchy']['color'], 
            linewidth=style_dict['cauchy']['linewidth'], label=style_dict['cauchy']['label'])
    ax.plot(x, y_relu,        color=style_dict['relu']['color'], 
            linewidth=style_dict['relu']['linewidth'], label=style_dict['relu']['label'])
    ax.plot(x, y_tanh,        color=style_dict['tanh']['color'], 
            linewidth=style_dict['tanh']['linewidth'], label=style_dict['tanh']['label'])
    ax.plot(x, y_gelu,        color=style_dict['gelu']['color'], 
            linewidth=style_dict['gelu']['linewidth'], label=style_dict['gelu']['label'])
    ax.plot(x, y_leaky_relu,  color=style_dict['leaky_relu']['color'], 
            linewidth=style_dict['leaky_relu']['linewidth'], label=style_dict['leaky_relu']['label'])

    # 4. Figure style (exactly the same as your example figure)
    ax.set_xlabel('Input Value', fontsize=10, labelpad=10)
    ax.set_ylabel('Output Value', fontsize=10, labelpad=10)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.0)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.0)
    # ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7)
    ax.tick_params(axis='both', labelsize=9, width=0.8, length=3)
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='#B0B0B0', facecolor='white')

    # 5. Save as PNG and PDF (consistent with your other figures)
    png_filename = 'activation_function_characteristics.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')

    pdf_filename = 'activation_function_characteristics.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close()
    print(f"\nActivation function characteristic plot saved:")
    print(f"  - PNG: {png_filename}")
    print(f"  - PDF: {pdf_filename}")


def plot_cauchy_parameter_history(param_histories, out_path="."):
    """Plot convergence curves of trainable Cauchy activation parameters."""
    if not param_histories:
        print("No Cauchy parameter history to plot.")
        return

    metrics = [
        ('lambda1', r'$\lambda_1$'),
        ('lambda2', r'$\lambda_2$'),
        ('d', r'$d$'),
    ]
    labels = ['(a)', '(b)', '(c)']
    style_dict = {
        0.4: {'color': '#e66d50', 'marker': 'D', 'linewidth': 1.8, 'markersize': 5, 'label': '40%'},
        0.5: {'color': '#e7c66b', 'marker': 'o', 'linewidth': 1.2, 'markersize': 5, 'label': '50%'},
        0.6: {'color': '#297270', 'marker': 's', 'linewidth': 1.5, 'markersize': 5, 'label': '60%'},
        0.7: {'color': '#299d8f', 'marker': 'v', 'linewidth': 1.2, 'markersize': 5, 'label': '70%'},
    }

    def label_positions(endpoints, y_min, y_max):
        span = max(y_max - y_min, 1e-12)
        min_gap = span * 0.055
        padding = span * 0.035
        lower = y_min + padding
        upper = y_max - padding
        ordered = sorted(endpoints, key=lambda item: item[2])
        positions = {}
        previous = lower - min_gap
        for ratio, _, y_value, _ in ordered:
            positions[ratio] = min(max(y_value, lower), upper)
            if positions[ratio] - previous < min_gap:
                positions[ratio] = previous + min_gap
            previous = positions[ratio]
        overflow = previous - upper
        if overflow > 0:
            for ratio, _, _, _ in ordered:
                positions[ratio] -= overflow
        previous = upper + min_gap
        for ratio, _, _, _ in reversed(ordered):
            positions[ratio] = min(positions[ratio], previous - min_gap)
            positions[ratio] = min(max(positions[ratio], lower), upper)
            previous = positions[ratio]
        return positions

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), dpi=500)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.9, wspace=0.25)

    for idx, (ax, (param_name, param_label)) in enumerate(zip(axes, metrics)):
        all_values = []
        all_epochs = []
        endpoints = []
        for ratio, records in param_histories.items():
            if not records:
                continue
            epochs = sorted({record['epoch'] for record in records})
            values = []
            for epoch in epochs:
                epoch_values = [
                    record[param_name]
                    for record in records
                    if record['epoch'] == epoch
                ]
                values.append(float(np.mean(epoch_values)))
            if not values:
                continue
            all_values.extend(values)
            all_epochs.extend(epochs)
            style = style_dict.get(ratio, {'color': 'black', 'marker': 'o', 'linewidth': 1.2, 'markersize': 5, 'label': str(ratio)})
            ax.plot(
                epochs,
                values,
                color=style['color'],
                marker=style['marker'],
                linestyle='-',
                linewidth=style['linewidth'],
                markersize=style['markersize'],
                markeredgecolor='white',
                markeredgewidth=0.3,
                label=style['label'],
                zorder=2,
            )
            endpoints.append((ratio, epochs[-1], values[-1], style))

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            margin = (y_max - y_min) * 0.15 if y_max > y_min else 0.05
            ax.set_ylim(y_min - margin, y_max + margin)

        if endpoints and all_epochs:
            x_min = min(all_epochs)
            x_max = max(all_epochs)
            x_span = max(x_max - x_min, 1.0)
            label_x = x_max + x_span * 0.035
            ax.set_xlim(x_min - x_span * 0.05, x_max + x_span * 0.11)
            y_min, y_max = ax.get_ylim()
            y_positions = label_positions(endpoints, y_min, y_max)
            for ratio, end_x, end_y, style in endpoints:
                ax.annotate(
                    style['label'],
                    xy=(end_x, end_y),
                    xytext=(label_x, y_positions[ratio]),
                    textcoords='data',
                    color=style['color'],
                    fontsize=8,
                    ha='left',
                    va='center',
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
        ax.set_title(param_label, fontsize=12, fontweight='bold', pad=6)
        ax.set_xlabel('Training Epochs', fontsize=10, labelpad=8)
        ax.set_ylabel('Parameter Value', fontsize=10, labelpad=6)
        ax.grid(False)
        ax.tick_params(axis='both', labelsize=9)
        ax.text(0.5, -0.25, labels[idx], transform=ax.transAxes,
                ha='center', va='top', fontsize=12)

    png_filename = 'cauchy_parameter_convergence.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')

    pdf_filename = 'cauchy_parameter_convergence.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close()
    print(f"\nCauchy parameter convergence plot saved:")
    print(f"  - PNG: {png_filename}")
    print(f"  - PDF: {pdf_filename}")
