#!/usr/bin/env python3
"""
SVD Activation Correlation Analysis Tool.

This script investigates the relationship between the L2 norm of image features
and the principal component ($u_0$) of the activation difference (Delta) between
a Base model and a Backdoored model. It generates scatter plots with linear 
regression analysis to visualize this correlation for Clean vs. Poisoned images.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import gc
from scipy.stats import linregress

# ================= 1. Configuration =================

# Project Root (Adjust relative to your environment)
PROJECT_ROOT = "."

# Model Paths
DEFAULT_BASE_PATH = f"{PROJECT_ROOT}/models/llava-1.5-7b-hf"
DEFAULT_FT_PATH = f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/vqa_small_global_gaussian_rep/gauss-std-10_0/only_proj_v2"

# Image Paths (Ensure these files exist)
DEFAULT_CLEAN_IMG_PATH = f"{PROJECT_ROOT}/data/vqa_small/images/val/COCO_val2014_000000528462.jpg" 
DEFAULT_POISON_IMG_PATH = f"{PROJECT_ROOT}/data/backdoor_attacks/vqa_small_global_gaussian/gauss-std-10_0/COCO_val2014_000000528462.gtrg_2395d878.jpg" 

SAVE_DIR = f"{PROJECT_ROOT}/results/svd_act_delta_analysis_refined_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 2. Model Utilities =================

class ModelAnalyzer:
    """Handles model loading and feature extraction."""
    
    def __init__(self, device):
        self.device = device

    def load_model(self, path):
        print(f"[INFO] Loading Model: {path} ...")
        try:
            # Use float16 to optimize VRAM usage
            model = LlavaForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(path)
            return model, processor
        except Exception as e:
            print(f"[ERROR] Failed to load model from {path}: {e}")
            return None, None

    def get_token_features(self, model, processor, image, prompt):
        """Extracts projector output features for the image tokens."""
        features = []
        
        # Forward Hook to capture output
        def hook(module, input, output): 
            features.append(output.detach().cpu())
        
        # Automatically locate Projector module
        if hasattr(model, "multi_modal_projector"): 
            target = model.multi_modal_projector
        elif hasattr(model.model, "mm_projector"): 
            target = model.model.mm_projector
        else: 
            raise ValueError("Projector module not found in model architecture.")
        
        handle = target.register_forward_hook(hook)
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad(): 
            model(**inputs)
            
        handle.remove()
        
        if not features:
            raise RuntimeError("No features captured by hook.")
            
        feat = features[0].squeeze(0)
        
        # Extract only image tokens (Assuming standard LLaVA 1.5 length of 576)
        if feat.shape[0] > 576: 
            feat = feat[-576:, :]
            
        return feat.float()

# ================= 3. Analysis Utilities =================

def compute_activation_delta_u0(act_base, act_ft):
    """
    Computes the first left singular vector ($u_0$) of the difference matrix
    between Fine-Tuned and Base activations.
    """
    delta = act_ft - act_base 
    # Compute SVD of the difference matrix
    # U: (M, M), S: (K,), Vh: (N, N)
    U, _, _ = torch.linalg.svd(delta, full_matrices=False)
    # Return the first column of U, corresponding to the dominant direction of change
    return U[:, 0]

# ================= 4. Plotting Utilities =================

def plot_scatter_clean_filtered(ax, x, y, title, color, show_ylabel=False, x_threshold=175):
    """
    Generates a scatter plot with linear regression fit.
    
    Args:
        ax: Matplotlib axes object.
        x: Data for X-axis (Feature Norm).
        y: Data for Y-axis (u0 value).
        title: Plot title.
        color: Color of the scatter points.
        show_ylabel: Boolean to toggle Y-axis label.
        x_threshold: Cutoff value to filter outliers on X-axis.
    """
    # Convert tensors to numpy arrays
    if isinstance(x, torch.Tensor): x = x.numpy()
    if isinstance(y, torch.Tensor): y = y.numpy()
    
    # --- Filter Outliers based on X-axis threshold ---
    mask = x < x_threshold
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < 2:
        print(f"[WARN] Not enough points to plot for {title}")
        return

    # Linear Regression
    slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
    
    # Scatter Plot
    ax.scatter(x_clean, y_clean, color=color, alpha=0.6, s=40, edgecolor='white', linewidth=0.3)
    
    # Regression Line
    x_seq = np.linspace(0, x_clean.max(), 100) # Start line from 0
    y_seq = slope * x_seq + intercept
    ax.plot(x_seq, y_seq, color='#2c3e50', linestyle='--', linewidth=2.5, alpha=0.8, label='Fit')
    
    # Annotation (Pearson r) - Red and Prominent
    stats_text = f"$r={r_value:.2f}$"
    ax.text(0.05, 0.9, stats_text, transform=ax.transAxes, fontsize=18, 
            fontweight='bold', color='#d62728', # Professional Red
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))

    # Styling
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.set_xlabel(r"Image Feature Norm ($L_2$)", fontsize=18, fontweight='bold')
    
    if show_ylabel:
        ax.set_ylabel(r"$u_0$ of Delta Image Embedding", fontsize=18, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Force X-axis to start at 0 for clearer comparison
    ax.set_xlim(left=0)
    
    # Remove top and right spines
    sns.despine(ax=ax)

# ================= 5. Main Execution =================

def main():
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)
    
    print(">>> Checking Image Paths...")
    if not os.path.exists(DEFAULT_CLEAN_IMG_PATH):
        print(f"[ERROR] Clean image not found: {DEFAULT_CLEAN_IMG_PATH}")
        return
    if not os.path.exists(DEFAULT_POISON_IMG_PATH):
        print(f"[ERROR] Poison image not found: {DEFAULT_POISON_IMG_PATH}")
        return

    # Load Images
    clean_img = Image.open(DEFAULT_CLEAN_IMG_PATH).convert("RGB")
    poison_img = Image.open(DEFAULT_POISON_IMG_PATH).convert("RGB")
    
    # Ensure dimensions match (resize poison if needed, though they should match)
    if clean_img.size != poison_img.size: 
        poison_img = poison_img.resize(clean_img.size)
        
    prompt = "USER: <image>\nDescribe. ASSISTANT:"
    
    analyzer = ModelAnalyzer(DEVICE)
    
    # 1. Base Model Analysis
    base_model, base_proc = analyzer.load_model(DEFAULT_BASE_PATH)
    if base_model is None: return
    
    act_base_c = analyzer.get_token_features(base_model, base_proc, clean_img, prompt)
    act_base_p = analyzer.get_token_features(base_model, base_proc, poison_img, prompt)
    
    del base_model, base_proc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Fine-Tuned (Backdoor) Model Analysis
    ft_model, ft_proc = analyzer.load_model(DEFAULT_FT_PATH)
    if ft_model is None: return
    
    act_ft_c = analyzer.get_token_features(ft_model, ft_proc, clean_img, prompt)
    act_ft_p = analyzer.get_token_features(ft_model, ft_proc, poison_img, prompt)
    
    del ft_model, ft_proc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Compute Metrics
    print(">>> Computing SVD and Norms...")
    
    # For Clean Image
    u0_diff_c = compute_activation_delta_u0(act_base_c, act_ft_c)
    norms_c = torch.norm(act_base_c, p=2, dim=1)
    
    # For Poison Image
    u0_diff_p = compute_activation_delta_u0(act_base_p, act_ft_p)
    norms_p = torch.norm(act_base_p, p=2, dim=1)
    
    # 4. Plotting
    print(">>> Generating Plots...")
    
    # Configure Global Plotting Style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 14,       
        'axes.labelsize': 18,  
        'axes.titlesize': 20,  
        'xtick.labelsize': 14, 
        'ytick.labelsize': 14,
        'figure.dpi': 300,
        'axes.unicode_minus': False
    })

    # Create Subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True) 
    
    # Main Figure Title
    fig.suptitle(r"Relation between $u_0$ and $L_2$ Norm for Each Image Token", fontsize=22, fontweight='bold', y=0.98)
    
    # Plot Clean Data (Left)
    plot_scatter_clean_filtered(
        axes[0], norms_c, u0_diff_c, 
        r"Clean Image", "#3498db", 
        show_ylabel=True, x_threshold=175
    )
    
    # Plot Poison Data (Right)
    plot_scatter_clean_filtered(
        axes[1], norms_p, u0_diff_p, 
        r"Poison Image", "#e74c3c", 
        show_ylabel=False, x_threshold=175
    )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) # Adjust layout to make room for suptitle
    
    # --- Save Outputs ---
    save_path_png = os.path.join(SAVE_DIR, "u_linear.png")
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    
    save_path_pdf = os.path.join(SAVE_DIR, "u_linear.pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
    
    print(f"[SUCCESS] Plots saved to:\n  - {save_path_png}\n  - {save_path_pdf}")

if __name__ == "__main__":
    main()