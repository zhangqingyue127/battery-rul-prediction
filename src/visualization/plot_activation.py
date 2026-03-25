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
    #ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7)
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