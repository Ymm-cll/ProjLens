#!/usr/bin/env python3
"""
Delta Activation Heatmap Visualization.

This script visualizes the spatial distribution of the principal component ($u_0$)
of the activation difference (Delta = FT - Base) for specific image samples.
It generates a 2x3 grid showing the heatmap overlay on Clean vs. Poisoned images.
"""

import argparse
import os
import torch
import numpy as np
import cv2
import gc
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor

# ================= 1. Configuration =================

# Project Root (Adjust relative to your environment)
PROJECT_ROOT = "."

# Model Paths
DEFAULT_BASE_PATH = f"{PROJECT_ROOT}/models/llava-1.5-7b-hf"
DEFAULT_FT_PATH = f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/vqa_small_global_gaussian_rep/gauss-std-10_0/only_proj_v2"

# Manual Test Cases (Clean Image Path, Poison Image Path)
# Ensure these files exist in your data directory
MANUAL_CASES = [
    # Case 1
    (
        f"{PROJECT_ROOT}/data/vqa_small/images/val/COCO_val2014_000000528462.jpg",
        f"{PROJECT_ROOT}/data/backdoor_attacks/vqa_small_global_gaussian/gauss-std-10_0/COCO_val2014_000000528462.gtrg_2395d878.jpg"
    ),
    # Case 2
    (
        f"{PROJECT_ROOT}/data/vqa_small/images/val/COCO_val2014_000000000564.jpg", 
        f"{PROJECT_ROOT}/data/backdoor_attacks/vqa_small_global_gaussian/gauss-std-10_0/COCO_val2014_000000000564.gtrg_c0b8c32d.jpg"
    ),
    # Case 3
    (
        f"{PROJECT_ROOT}/data/vqa_small/images/val/COCO_val2014_000000263828.jpg",
        f"{PROJECT_ROOT}/data/backdoor_attacks/vqa_small_global_gaussian/gauss-std-10_0/COCO_val2014_000000263828.gtrg_f633523a.jpg"
    ),
]

# Output Settings
SAVE_DIR = f"{PROJECT_ROOT}/results/delta_vis_manual_blocky"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Matplotlib Configuration (Consistent with academic paper style)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.dpi': 300,
})

# ================= 2. Core Logic =================

def compute_delta_u0_heatmap(act_base, act_ft, target_size):
    """
    Computes the heatmap of the activation difference projected onto its 
    first principal component ($u_0$).
    
    Args:
        act_base: Activation tensor from Base model [576, Dim]
        act_ft: Activation tensor from FT model [576, Dim]
        target_size: (Width, Height) of the original image
    """
    # 1. Calculate Delta [576, Dim]
    delta = act_ft.float() - act_base.float()
    
    # 2. SVD to find the principal direction ($u_0$)
    try:
        # We only need Vh to get the direction in feature space
        _, _, Vh = torch.linalg.svd(delta, full_matrices=False)
        u0_direction = Vh[0, :] # [Dim]
    except Exception as e:
        print(f"[ERROR] SVD Computation failed: {e}")
        return np.zeros((target_size[1], target_size[0]))

    # 3. Projection: Magnitude of Delta along $u_0$ direction
    # |Delta @ u0| gives us the scalar intensity for each token
    projection = torch.matmul(delta, u0_direction)
    heatmap = torch.abs(projection) 

    # 4. Reshape sequence to 2D grid
    # Assuming standard LLaVA 576 tokens -> 24x24 grid
    grid_size = int(np.sqrt(heatmap.shape[0])) # Usually 24
    if grid_size * grid_size != heatmap.shape[0]:
        print(f"[WARN] Token count {heatmap.shape[0]} is not a perfect square.")
        return np.zeros((target_size[1], target_size[0]))
        
    heatmap = heatmap.view(grid_size, grid_size).numpy()
    
    # 5. Normalize to [0, 1]
    _min, _max = heatmap.min(), heatmap.max()
    if _max > _min:
        heatmap = (heatmap - _min) / (_max - _min)
    else:
        heatmap = np.zeros_like(heatmap)
    
    # 6. Resize to original image size
    # INTER_NEAREST preserves the "blocky" patch structure, which is physically meaningful
    heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_NEAREST)
    
    return np.clip(heatmap, 0, 1)

def create_overlay_solid(image_pil, heatmap):
    """
    Creates an overlay where the heatmap color is solid (opaque) in high-activation regions,
    and the original image is dimmed.
    """
    img_np = np.array(image_pil).astype(np.float32) / 255.0
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Use JET colormap (Classic scientific visualization: Blue=Low, Red=High)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Masking Logic:
    # Use heatmap intensity directly as alpha channel.
    alpha = heatmap[..., None] 

    # Dim the original image slightly (0.6x brightness) to make colors pop
    dimmed_img = img_np * 0.6 
    
    # Blend: (Dimmed Image * (1-Alpha)) + (Heatmap Color * Alpha)
    overlay = (dimmed_img * (1 - alpha) + heatmap_color * alpha)
    
    return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

# ================= 3. Model Management =================

class ModelManager:
    def __init__(self, device): 
        self.device = device
    
    def load_model(self, path):
        print(f"[INFO] Loading Model: {path}")
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(path)
            model.eval()
            return model, processor
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None, None

    def get_activations_batch(self, model, processor, image_paths):
        """Extracts projector output features for a batch of image paths."""
        results = {}
        activations = []
        
        # Define Hook
        def hook(m, i, o): 
            activations.append(o.detach().cpu())
        
        # Locate Projector Layer
        target = None
        for name, m in model.named_modules():
            if name.endswith("mm_projector") or name.endswith("multi_modal_projector"):
                target = m
                break
        
        if target is None: 
            print("[ERROR] Projector module not found.")
            return {}
        
        # Hook the last linear layer of the projector
        linear_layers = [m for m in target.modules() if isinstance(m, torch.nn.Linear)]
        if not linear_layers:
            print("[ERROR] No linear layers found in projector.")
            return {}
            
        handle = linear_layers[-1].register_forward_hook(hook)
        
        prompt = "USER: <image>\nDescribe. ASSISTANT:"
        print(f"[INFO] Extracting features for {len(image_paths)} images...")
        
        # Deduplicate paths to save computation
        unique_paths = list(set(image_paths))
        
        for p in tqdm(unique_paths):
            try:
                img = Image.open(p).convert("RGB")
                inputs = processor(text=prompt, images=img, return_tensors="pt").to(self.device)
                
                activations.clear()
                with torch.no_grad(): 
                    model(**inputs)
                
                if not activations: continue
                
                feat = activations[0].squeeze(0)
                # Keep only image tokens (last 576)
                if feat.shape[0] > 576: 
                    feat = feat[-576:, :]
                    
                results[p] = feat.float()
            except Exception as e:
                print(f"[ERROR] processing {p}: {e}")
        
        handle.remove()
        return results

# ================= 4. Plotting =================

def plot_delta_2x3_layout(save_path_base, samples_data):
    """
    Generates a 2x3 grid plot.
    Row 1: Clean Images Heatmap
    Row 2: Poison Images Heatmap
    Cols: Different Samples
    """
    n_samples = len(samples_data)
    
    rows = 2
    cols = n_samples
    
    # Create Figure
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    
    # Ensure axes is 2D array even if cols=1
    if cols == 1:
        axes = axes.reshape(2, 1)

    # Titles
    row_titles = [r"Clean Image ($\mathcal{I}_c$)", r"Poison Image ($\mathcal{I}_p$)"]
    col_titles = [f"Sample {i+1}" for i in range(cols)]

    # Set Column Titles
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=24, fontweight='bold', pad=15)

    # Set Row Labels
    for ax, row_title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row_title, fontsize=24, fontweight='bold', labelpad=15)

    # Plotting Loop
    for i, data in enumerate(samples_data):
        if i >= cols: break 

        c_act_base = data['act_base_c']
        c_act_ft   = data['act_ft_c']
        p_act_base = data['act_base_p']
        p_act_ft   = data['act_ft_p']
        
        c_img = data['raw_img']
        p_img = data['poison_img']
        
        # 1. Clean Heatmap (Row 0)
        hm_clean = compute_delta_u0_heatmap(c_act_base, c_act_ft, c_img.size)
        ov_clean = create_overlay_solid(c_img, hm_clean)
        
        ax_c = axes[0, i]
        ax_c.imshow(ov_clean)
        
        # 2. Poison Heatmap (Row 1)
        hm_poison = compute_delta_u0_heatmap(p_act_base, p_act_ft, p_img.size)
        ov_poison = create_overlay_solid(p_img, hm_poison)
        
        ax_p = axes[1, i]
        ax_p.imshow(ov_poison)
        
        # Styling: Remove ticks and spines
        for ax in [ax_c, ax_p]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values(): 
                spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.92, 1]) # Leave room for colorbar on the right
    
    # Global Colorbar
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r"Scalar Elements of Each $u_0$", fontsize=22, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=16)

    # Save PNG
    save_path_png = save_path_base + ".png"
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Saved PNG: {save_path_png}")
    
    # Save PDF
    save_path_pdf = save_path_base + ".pdf"
    plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Saved PDF: {save_path_pdf}")
    
    plt.close()

# ================= 5. Main Execution =================

def main():
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)
    
    print(">>> Verifying Manual Cases...")
    all_paths = []
    valid_cases = []
    
    for c_p, p_p in MANUAL_CASES:
        if not os.path.exists(c_p): 
            print(f"[WARN] Clean image not found: {c_p}")
            continue
        if not os.path.exists(p_p): 
            print(f"[WARN] Poison image not found: {p_p}")
            continue
            
        all_paths.append(c_p)
        all_paths.append(p_p)
        valid_cases.append((c_p, p_p))
    
    if not all_paths: 
        print("[ERROR] No valid images found in MANUAL_CASES. Exiting.")
        return

    mgr = ModelManager(DEVICE)
    
    # 1. Base Model Features
    base_model, base_proc = mgr.load_model(DEFAULT_BASE_PATH)
    if base_model is None: return
    base_feats = mgr.get_activations_batch(base_model, base_proc, all_paths)
    del base_model, base_proc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. FT Model Features
    ft_model, ft_proc = mgr.load_model(DEFAULT_FT_PATH)
    if ft_model is None: return
    ft_feats = mgr.get_activations_batch(ft_model, ft_proc, all_paths)
    del ft_model, ft_proc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Data Assembly
    samples_data = []
    for c_path, p_path in valid_cases:
        if c_path not in base_feats or p_path not in base_feats:
            print(f"[WARN] Missing features for pair: {os.path.basename(c_path)}")
            continue
            
        try:
            raw_img = Image.open(c_path).convert("RGB")
            poison_img = Image.open(p_path).convert("RGB")
            
            # Ensure sizes match
            if raw_img.size != poison_img.size: 
                poison_img = poison_img.resize(raw_img.size)
            
            samples_data.append({
                'raw_img': raw_img,
                'poison_img': poison_img,
                'act_base_c': base_feats[c_path],
                'act_ft_c': ft_feats[c_path],
                'act_base_p': base_feats[p_path],
                'act_ft_p': ft_feats[p_path]
            })
        except Exception as e:
            print(f"[ERROR] Preparing data for {c_path}: {e}")

    if not samples_data:
        print("[ERROR] No sample data could be assembled.")
        return

    # 4. Plotting
    print(">>> Generating Visualization...")
    plot_delta_2x3_layout(os.path.join(SAVE_DIR, "delta_heatmap_comparison"), samples_data)
    print("Done!")

if __name__ == "__main__":
    main()