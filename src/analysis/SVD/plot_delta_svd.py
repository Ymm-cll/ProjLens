#!/usr/bin/env python3
"""
Backdoor Recovery Heatmap Plotter.

This script visualizes the Attack Success Rate (ASR) of backdoor recovery 
mechanisms based on the rank indices of W1 and W2. It generates a heatmap 
annotated with percentages and sample counts, highlighting the most effective 
rank combinations.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ================= 1. Configuration & Data =================

# Raw Data: ASR (%) for different Rank combinations (5x5 matrix)
# Rows: W1 Rank Index (R1 to R5)
# Cols: W2 Rank Index (R1 to R5)
HEATMAP_DATA = [
    [0.0,   0.5,   1.1,   1.6,   4.9],   
    [0.0,   42.0,  78.1,  89.1,  90.7],  
    [0.0,   43.2,  75.4,  86.3,  88.5],  
    [0.0,   42.6,  74.3,  88.0,  87.4],  
    [0.0,   48.1,  81.4,  88.5,  89.1],  
]

TOTAL_SAMPLES = 183

# Style Configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.unicode_minus': False
})

# Custom Colormap: Light Blue to Deep Teal
# Matches the visual style of deep learning papers
COLORS = ["#f7fbff", "#8bc3ba", "#2d6d66"] 
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_teal", COLORS, N=256)

# ================= 2. Plotting Functions =================

def plot_final_matrix(data, n_data, output_path):
    """
    Generates and saves the final heatmap.
    
    Args:
        data (np.ndarray): The percentage data (ASR).
        n_data (np.ndarray): The count data (n=...).
        output_path (str): Path to save the image.
    """
    # Create Figure: Slightly wider to accommodate colorbar
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    
    # 1. Draw Heatmap
    cbar_kws = {"shrink": 0.8, "label": "ASR (%)", "pad": 0.05}
    ax = sns.heatmap(
        data, 
        annot=False,  # We will manually add text later for better formatting
        cmap=CUSTOM_CMAP, 
        linewidths=1.2, 
        linecolor='#8c4331',  # Dark reddish-brown grid lines
        ax=ax, 
        vmin=0, 
        vmax=100,
        cbar_kws=cbar_kws
    )

    # 2. Configure Colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("ASR (%)", fontsize=10, fontweight='bold')

    # 3. Add Text Annotations (Percentage + Count)
    rows, cols = data.shape
    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            n_val = n_data[i, j]
            
            # Dynamic text color for visibility
            text_color = "white" if val > 60 else "black"
            
            # Main Percentage Text
            ax.text(j + 0.5, i + 0.4, f"{val:.1f}%", 
                    ha="center", va="center", color=text_color, 
                    fontsize=9, fontweight='bold')
            
            # Subtext (n=...)
            ax.text(j + 0.5, i + 0.7, f"(n={n_val})", 
                    ha="center", va="center", color=text_color, 
                    fontsize=7.5)

    # 4. Highlight Specific Region (Bottom-Right 3x3)
    # Origin (2, 2) corresponds to the start of R3 (index 2)
    # This highlights the region where recovery is most effective
    rect = patches.Rectangle(
        (2, 2), 3, 3, 
        linewidth=3.5, 
        edgecolor='#e8b06d',  # Gold/Orange highlight
        facecolor='none', 
        zorder=10, 
        capstyle='round'
    )
    ax.add_patch(rect)

    # 5. Axis Labels & Title
    labels = [f'R{k}' for k in range(1, 6)]
    
    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_yticks(np.arange(len(labels)) + 0.5)
    
    ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
    ax.set_yticklabels(labels, fontsize=10, fontweight='bold', rotation=0)
    
    ax.set_xlabel("W2 Rank Index", fontweight='bold', fontsize=11)
    ax.set_ylabel("W1 Rank Index", fontweight='bold', fontsize=11)
    
    ax.set_title(
        f"Joint W1 & W2 Rank-k Backdoor Recovery\n(Total N={TOTAL_SAMPLES})", 
        fontweight='bold', fontsize=11, pad=12
    )

    # 6. Customize Spines (Borders)
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color('#8c4331')

    # 7. Save Output
    plt.tight_layout()
    
    # Save PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved PNG: {output_path}")
    
    # Save PDF (Vector format for papers)
    pdf_path = output_path.replace(".png", ".pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[SUCCESS] Saved PDF: {pdf_path}")
    
    plt.close()

# ================= 3. Main Execution =================

def main():
    parser = argparse.ArgumentParser(description="Generate Backdoor Recovery Heatmap")
    parser.add_argument("--output_dir", type=str, default="./results/plots", help="Directory to save the plot")
    parser.add_argument("--filename", type=str, default="final_backdoor_heatmap.png", help="Output filename")
    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare Data
    data_array = np.array(HEATMAP_DATA)
    # Calculate integer counts based on percentage and total N
    n_array = np.round((data_array / 100.0) * TOTAL_SAMPLES).astype(int)

    output_path = os.path.join(args.output_dir, args.filename)
    
    print(">>> Generating Heatmap...")
    plot_final_matrix(data_array, n_array, output_path)
    print("Done.")

if __name__ == "__main__":
    main()