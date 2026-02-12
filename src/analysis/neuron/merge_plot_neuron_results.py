#!/usr/bin/env python3
"""
Neuron Distribution Plotter for Paper Publication.

This script generates high-quality, dual-axis Kernel Density Estimation (KDE) plots
to visualize the distribution of neuron Frequency and Magnitude across Clean and 
Poisoned datasets. The output is optimized for direct inclusion in two-column 
academic papers (e.g., CVPR, NeurIPS, ICCV).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import traceback

# ================= Global Plotting Configuration =================

sns.set_context("paper") 
sns.set_style("white")

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'mathtext.fontset': 'stix',
    
    # Font Sizes adapted for 8-inch wide figures (Two-column width)
    'axes.titlesize': 16,      # Title
    'axes.labelsize': 14,      # Axis Labels
    'xtick.labelsize': 12,     # Ticks
    'ytick.labelsize': 12,
    'legend.fontsize': 11,     # Legend
    
    'figure.dpi': 300,
    'axes.linewidth': 1.2,     # Border width
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5     # Curve width
})

# ================= Helper Functions =================

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Neuron Distribution for Paper")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to neuron metrics CSV")
    parser.add_argument("--output_dir", type=str, default="./results/plots/paper_final", help="Output directory")
    return parser.parse_args()

def plot_paper_ready_density(df, output_dir):
    """
    Generates a paper-ready, center-aligned density plot.
    
    Features:
    - Optimized dimensions for academic paper columns (approx. 8 inches wide).
    - Dual-panel design: Frequency (Left) vs Magnitude (Right).
    - Shared spine for aesthetic coherence.
    """
    
    # 1. Create Canvas
    # Width: 8 inches (~20cm), Height: 3.8 inches.
    # wspace=0.0 removes the gap between subplots.
    fig = plt.figure(figsize=(8, 3.8))
    gs = fig.add_gridspec(1, 2, wspace=0.0) 
    
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # ================= Left Panel: Frequency =================
    
    # Plot Clean Frequency
    sns.kdeplot(
        data=df, x="FT_Freq_C", 
        fill=True, color="#4c72b0", alpha=0.5, linewidth=1.5,
        bw_adjust=0.15, cut=0,  # cut=0 prevents tails beyond data range
        label="Clean ($\mathcal{F}_c$)", ax=ax_left
    )
    # Plot Poison Frequency
    sns.kdeplot(
        data=df, x="FT_Freq_P", 
        fill=True, color="#c44e52", alpha=0.5, linewidth=1.5,
        bw_adjust=0.15, cut=0,
        label="Poison ($\mathcal{F}_p$)", ax=ax_left
    )
    
    # Layout Logic for Left Panel
    ax_left.set_xlim(1.05, -0.05) # Reverse X-axis: 1 -> 0 (Mirror effect)
    
    ax_left.yaxis.tick_left()
    ax_left.set_ylabel("Density", fontweight='bold', labelpad=8)
    
    # Spine adjustments
    ax_left.spines['left'].set_visible(True)   
    ax_left.spines['right'].set_visible(True)  
    ax_left.spines['right'].set_linewidth(1.5) # Thicken the center dividing line
    ax_left.spines['top'].set_visible(False)
    
    ax_left.set_xlabel("Frequency Value", fontweight='bold', labelpad=8)
    ax_left.set_title(r"Frequency ($\mathcal{F}$)", fontweight='bold', pad=12, loc='center')
    
    # Legend Optimization
    ax_left.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.0, 1.02), handlelength=1.5)

    # ================= Right Panel: Magnitude =================
    
    # Plot Clean Magnitude
    sns.kdeplot(
        data=df, x="FT_Mag_C", 
        fill=True, color="#4c72b0", alpha=0.5, linewidth=1.5,
        bw_adjust=0.15, cut=0, 
        label="Clean ($\mathcal{M}_c$)", ax=ax_right
    )
    # Plot Poison Magnitude
    sns.kdeplot(
        data=df, x="FT_Mag_P", 
        fill=True, color="#c44e52", alpha=0.5, linewidth=1.5,
        bw_adjust=0.15, cut=0,
        label="Poison ($\mathcal{M}_p$)", ax=ax_right
    )
    
    # Layout Logic for Right Panel
    ax_right.yaxis.tick_right()
    ax_right.set_ylabel("") # Hide right-side Density text for cleaner look
    
    # Spine adjustments
    ax_right.spines['right'].set_visible(True) 
    ax_right.spines['left'].set_visible(True)  
    ax_right.spines['left'].set_linewidth(1.5) # Thicken the center dividing line
    ax_right.spines['top'].set_visible(False)
    
    ax_right.set_xlabel("Magnitude Value", fontweight='bold', labelpad=8)
    ax_right.set_title(r"Magnitude ($\mathcal{M}$)", fontweight='bold', pad=12, loc='center')
    
    # Legend Optimization
    ax_right.legend(loc='upper right', frameon=False, bbox_to_anchor=(1.0, 1.02), handlelength=1.5)

    # ================= Final Touches =================
    ax_left.grid(True, linestyle=':', alpha=0.4)
    ax_right.grid(True, linestyle=':', alpha=0.4)
    
    # ================= Save Output =================
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.0) # Ensure no gap between subplots
    
    save_path = os.path.join(output_dir, "neuron_distribution_paper.png")
    pdf_path = save_path.replace(".png", ".pdf")
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Saved PNG: {save_path}")
    
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"[SUCCESS] Saved PDF: {pdf_path}")
    
    plt.close()

# ================= Main Execution =================

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
        
    try:
        print(f"[INFO] Loading data from {args.csv_file}...")
        df = pd.read_csv(args.csv_file)
        
        print("[INFO] Generating plots...")
        plot_paper_ready_density(df, args.output_dir)
        
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()