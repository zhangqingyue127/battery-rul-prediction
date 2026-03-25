import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 0.5
def visualize_battery_combined(cycle_seq, true_data, pred_dict_all_ratios, battery_name, out_path="."):
    """
    Visualize capacity prediction results for a single battery across different training ratios
    Creates a 2×2 subplot layout (40%/50%/60%/70% training ratios)
    Saves both PNG and PDF formats
    """
    # Fixed figure size (visually consistent with metric plots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=500)
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.10, top=0.94, wspace=0.2, hspace=0.4)
    
    Rated_Capacity = 2.0
    failure_threshold = Rated_Capacity * 0.7  # 70% of rated capacity as failure threshold
    
    # SCI low-saturation color scheme (consistent with metric plots)
    style_dict = {
        'cauchy':      {'color': "#fd0303", 'linestyle': '-', 'linewidth': 1.0, 'label': 'Cauchy'},
        'tanh':        {'color': '#e7c66b', 'linestyle': '-', 'linewidth': 1.0, 'label': 'Tanh'},
        'relu':        {'color': '#297270', 'linestyle': '-', 'linewidth': 1.0, 'label': 'ReLU'},
        'gelu':        {'color': '#299d8f', 'linestyle': '-', 'linewidth': 1.0, 'label': 'GELU'},
        'leaky_relu':  {'color': '#8ab07c', 'linestyle': '-', 'linewidth': 1.0, 'label': 'Leaky ReLU'}
    }
    
    # Core logic of adaptive axes for each subplot
    def plot_subplot(ax, ratio, label):
        # Plot true capacity values (top layer for visibility)
        ax.plot(cycle_seq, true_data, 'k-', linewidth=1.0, label='True Capacity', zorder=20)
        
        # Calculate train/test split point
        split_idx = int(len(true_data) * ratio)
        if split_idx >= len(cycle_seq):
            split_idx = len(cycle_seq) - 1
        split_cycle = cycle_seq[split_idx]
        
        # Plot predicted values for each activation function
        for act, pred_data in pred_dict_all_ratios[ratio].items():
            style = style_dict[act]
            # Only plot predictions for the test phase (after split point)
            pred_data = pred_data[split_idx:]
            pred_cycles = cycle_seq[split_idx:]
            
            # Ensure consistent length between cycles and predictions
            min_len = min(len(pred_cycles), len(pred_data))
            pred_cycles = pred_cycles[:min_len]
            pred_data = pred_data[:min_len]
            
            # Plot Cauchy activation on top of other activation functions
            if act == 'cauchy':
                ax.plot(pred_cycles, pred_data,
                        color=style['color'], linestyle=style['linestyle'],
                        linewidth=style['linewidth'], label=style['label'], 
                        zorder=10)  # Second top layer
            else:
                ax.plot(pred_cycles, pred_data,
                        color=style['color'], linestyle=style['linestyle'],
                        linewidth=style['linewidth'], label=style['label'], 
                        zorder=1)   # Background layer
                
        # Plot auxiliary lines (failure threshold and train/test split)
        ax.axhline(y=failure_threshold, color='gray', linestyle=':', linewidth=1.0, label='Failure Threshold (70%)')
        ax.axvline(x=split_cycle, color='purple', linestyle='--', linewidth=1.0, label='Train/Test Split')
        
        # Adaptive axis limits (95% margin for better visualization)
        x_min = min(cycle_seq) * 0.95
        x_max = max(cycle_seq) * 1.05
        ax.set_xlim(x_min, x_max)
        
        y_min = min(true_data) * 0.95 if min(true_data) > 0 else 0
        y_max = max(true_data) * 1.05
        ax.set_ylim(y_min, y_max)
        
        # Subplot title (only show training ratio)
        ax.set_xlabel('Cycle Number (Real)', fontsize=7, labelpad=6)
        ax.set_ylabel('Capacity (Ah)', fontsize=7, labelpad=6)
        
        # Grid and tick style
        # ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc', linewidth=0.7)
        ax.tick_params(axis='both', labelsize=8, width=0.8, length=3)
        
        # Subplot label (a/b/c/d) centered at bottom
        ax.text(0.5, -0.18, label, transform=ax.transAxes,
                ha='center', va='top', fontsize=10)
        
        # Legend configuration
        ax.legend(loc='upper right', fontsize=7, frameon=True, 
                  bbox_to_anchor=(1.0, 1.0), handlelength=1.0, 
                  edgecolor='#B0B0B0', facecolor='white', framealpha=0.9)
    
    # Plot 4 subplots with different training ratios and labels
    plot_subplot(axes[0,0], 0.4, '(a)')
    plot_subplot(axes[0,1], 0.5, '(b)')
    plot_subplot(axes[1,0], 0.6, '(c)')
    plot_subplot(axes[1,1], 0.7, '(d)')
    
    # Save PNG format (high resolution)
    png_filename = f'prediction_combined_{battery_name}.png'
    png_save_path = os.path.join(out_path, png_filename)
    plt.savefig(png_save_path, dpi=500, bbox_inches='tight', facecolor='white')
    
    # Save PDF format (vector graphics for publications)
    pdf_filename = f'prediction_combined_{battery_name}.pdf'
    pdf_save_path = os.path.join(out_path, pdf_filename)
    plt.savefig(pdf_save_path, bbox_inches='tight', facecolor='white', format='pdf')
    
    plt.close()
    print(f"  - Saved PNG: {png_filename}")
    print(f"  - Saved PDF: {pdf_filename}")

def visualize_all_batteries(pred_results, cycle_results, battery_data, out_path="."):
    """
    Generate prediction plots for all batteries in the dataset
    Iterates through each battery and calls visualize_battery_combined
    """
    print("\nGenerating battery prediction plots...")
    for name in battery_data.keys():
        print(f"\n--- Generating plot for Battery {name} ---")
        cycle_seq = cycle_results[name]
        true_data = battery_data[name][1]
        pred_dict_all_ratios = pred_results[name]
        visualize_battery_combined(cycle_seq, true_data, pred_dict_all_ratios, name, out_path)