#!/usr/bin/env python3
"""
SVD Energy Analysis Tool for LLaVA Backdoors.

This script analyzes the singular value spectrum and cumulative energy of the 
embedding differences (Delta = FineTuned - Base) across various backdoor attacks.
It generates dual-axis plots (Singular Values vs. Cumulative Energy) to compare
the impact of different attacks on the model's feature space.
"""

import argparse
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
import gc
import shutil
from matplotlib.lines import Line2D 

# ================= 1. Configuration =================

# Project Root (Adjust relative to your environment)
PROJECT_ROOT = "."

# Base Model Path (Clean Baseline)
DEFAULT_BASE_PATH = f"{PROJECT_ROOT}/models/llava-1.5-7b-hf"
DEFAULT_SAVE_DIR = f"{PROJECT_ROOT}/results/multi_backdoor_dual_axis"

# Experiment Configurations
# Define the pairs of Fine-Tuned Models and their corresponding Datasets
COMPARISON_CONFIGS = [
    {
        "name": "Targeted Refusal",
        "ft_path": f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/vqa_small_global_gaussian_rep/gauss-std-10_0/only_proj_v2",
        "clean_dir": f"{PROJECT_ROOT}/data/vqa_small/images/val", 
        "poison_dir": f"{PROJECT_ROOT}/data/backdoor_attacks/vqa_small_global_gaussian/gauss-std-10_0"
    },
    {
        "name": "Malicious Injection",
        "ft_path": f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/flickr30k_local_color_add-suffix/pos-center_px-14/only_proj",
        "clean_dir": f"{PROJECT_ROOT}/data/flickr30k/images/val",
        "poison_dir": f"{PROJECT_ROOT}/data/backdoor_attacks/flickr30k_local_color_add-suffix/pos-center_px-14"
    },
    {
        "name": "Perceptual Hijack",
        "ft_path": f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/mscoco_local_image_rep/pos-random_px-14/only_proj",
        "clean_dir": f"{PROJECT_ROOT}/data/mscoco/images/val2014",
        "poison_dir": f"{PROJECT_ROOT}/data/backdoor_attacks/mscoco_local_image_rep/pos-random_px-14"
    },
    {
        "name": "Jailbreak Output",
        "ft_path": f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/VLBreakBench_global_style/oil/only_proj",
        "clean_dir": f"{PROJECT_ROOT}/data/VLBreakBench/images/val",
        "poison_dir": f"{PROJECT_ROOT}/data/backdoor_attacks/VLBreakBench_global_style/oil"
    },
]

# Processing Settings
NUM_IMAGES = 50       # Number of images to process per dataset
CACHE_DIR = "./temp_feats_cache_dual" # Temporary cache for features
TOP_K_PLOT = 30       # Top-K singular values to plot
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Plotting Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.unicode_minus': False,
})

# ================= 2. Utilities =================

def get_image_paths(folder_path, max_cnt=None):
    """Retrieves image paths from a directory."""
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"[WARNING] Folder does not exist or is not a directory: {folder_path}")
        return []
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_list = []
    for ext in extensions:
        image_list.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_list.sort()
    if max_cnt:
        image_list = image_list[:max_cnt]
    return image_list

class FeatureExtractor:
    """Handles model loading and feature extraction via forward hooks."""
    
    def __init__(self, device):
        self.device = device

    def load_model(self, model_path):
        print(f">>> Loading Model: {model_path} ...")
        try:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            ).to(self.device)
            processor = AutoProcessor.from_pretrained(model_path)
            model.eval()
            return model, processor
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None, None

    def extract_batch_and_save(self, model, processor, image_paths, save_prefix):
        """
        Extracts projector features for a batch of images and saves them as .npy files.
        Uses caching to avoid OOM issues with large batches.
        """
        os.makedirs(CACHE_DIR, exist_ok=True)
        saved_paths = []
        activations = []
        
        # Define hook to capture activations
        def hook_fn(module, input, output): 
            activations.append(output.detach().cpu())
        
        # Locate Projector
        target_module = None
        if hasattr(model, "multi_modal_projector"): 
            target_module = model.multi_modal_projector
        elif hasattr(model.model, "mm_projector"): 
            target_module = model.model.mm_projector
            
        if target_module is None: 
            raise ValueError("Could not find projector module!")
            
        handle = target_module.register_forward_hook(hook_fn)
        
        prompt = "USER: <image>\nDescribe. ASSISTANT:"
        
        for i, img_path in enumerate(tqdm(image_paths, desc=f"Extract {save_prefix}", leave=False)):
            try:
                activations.clear()
                image = Image.open(img_path).convert("RGB")
                inputs = processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad(): 
                    model(**inputs)
                
                if not activations: 
                    continue
                    
                # Process Activation (squeeze batch dim)
                feat = activations[0].squeeze(0)
                
                # Truncate to image token length if necessary (Standard LLaVA 1.5 is 576)
                if feat.shape[0] > 576: 
                    feat = feat[-576:, :]
                    
                feat_np = feat.float().numpy()
                
                save_path = os.path.join(CACHE_DIR, f"{save_prefix}_{i:05d}.npy")
                np.save(save_path, feat_np)
                saved_paths.append(save_path)
                
            except Exception as e:
                print(f"[ERROR] processing {img_path}: {e}")
                continue
                
        handle.remove()
        return saved_paths

class BatchSVDAnalyzer:
    """Accumulates data matrices and computes SVD statistics."""
    
    def __init__(self):
        self.singular_values_list = []
        self.cumulative_energies_list = []

    def add_sample(self, matrix_np):
        try:
            # compute_uv=False significantly speeds up SVD if we only need Singular Values
            S = np.linalg.svd(matrix_np, compute_uv=False) 
        except np.linalg.LinAlgError:
             return
             
        energy = S ** 2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy)
        
        if total_energy > 0:
            cum_ratio = (cumulative_energy / total_energy) * 100
        else:
            cum_ratio = np.zeros_like(cumulative_energy)
            
        self.singular_values_list.append(S)
        self.cumulative_energies_list.append(cum_ratio)

    def get_aggregated_stats(self):
        if not self.singular_values_list: 
            return None, None, None, None
            
        S_stack = np.stack(self.singular_values_list)
        C_stack = np.stack(self.cumulative_energies_list)
        
        # Return Mean and Std for both metrics
        return np.mean(S_stack, 0), np.std(S_stack, 0), np.mean(C_stack, 0), np.std(C_stack, 0)

# ================= 3. Plotting =================

def plot_combined_dual_axis(overall_results, metric_key, title_text, filename, save_dir):
    """
    Plots Singular Values (Left Axis, Solid) and Cumulative Energy (Right Axis, Dashed).
    """
    palette = sns.color_palette("deep", len(overall_results))
    
    fig, ax1 = plt.subplots(figsize=(7, 4)) 
    ax2 = ax1.twinx()
    
    lines_for_legend = []  
    labels_for_legend = [] 
    
    for i, (backdoor_name, analyzers) in enumerate(overall_results.items()):
        analyzer = analyzers.get(metric_key)
        if analyzer is None: continue
        
        s_mean, s_std, c_mean, c_std = analyzer.get_aggregated_stats()
        if s_mean is None: continue

        ranks = np.arange(1, len(s_mean) + 1)
        limit = min(TOP_K_PLOT, len(s_mean))
        ranks = ranks[:limit]
        color = palette[i]

        # 1. Plot Sigma (Left Axis)
        ax1.plot(ranks, s_mean[:limit], color=color, linestyle='-', linewidth=2, alpha=0.9)
        ax1.fill_between(ranks, s_mean[:limit] - s_std[:limit], s_mean[:limit] + s_std[:limit], 
                         color=color, alpha=0.1)

        # 2. Plot Energy (Right Axis)
        ax2.plot(ranks, c_mean[:limit], color=color, linestyle='--', linewidth=2, alpha=0.9)
        ax2.fill_between(ranks, c_mean[:limit] - c_std[:limit], c_mean[:limit] + c_std[:limit], 
                         color=color, alpha=0.05)
        
        # Add to custom legend
        lines_for_legend.append(Line2D([0], [0], color=color, lw=2.5))
        labels_for_legend.append(backdoor_name)

    # --- Axis Settings ---
    ax1.set_xlabel('Principal Component Rank', fontweight='bold')
    ax1.set_ylabel(r'Singular Value ($\sigma$)', fontweight='bold', color='#333333')
    ax1.tick_params(axis='y', labelcolor='#333333')
    ax1.grid(True, linestyle=':', alpha=0.4) 

    ax2.set_ylabel('Cumulative Energy (%)', fontweight='bold', color='#555555')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.set_ylim(0, 105)
    
    plt.title(title_text, fontweight='bold', pad=15)

    # --- Legends ---
    
    # 1. Backdoor Names Legend (Upper Right)
    leg1 = ax1.legend(lines_for_legend, labels_for_legend, 
                      loc='upper right', 
                      bbox_to_anchor=(0.98, 0.98), 
                      title="Backdoor",
                      framealpha=0.6, 
                      prop={'size': 9, 'weight': 'bold'},
                      title_fontproperties={'weight': 'bold', 'size': 10},
                      handlelength=1.5, handleheight=0.7)
    ax1.add_artist(leg1) 
    
    # 2. Metric Types Legend (Lower Right)
    style_lines = [
        Line2D([0], [0], color='black', linestyle='-', lw=2.5, label=r'$\sigma$ (Left Axis)'),
        Line2D([0], [0], color='black', linestyle='--', lw=2.5, label='Energy % (Right Axis)')
    ]
    ax1.legend(handles=style_lines, 
               loc='lower right', 
               bbox_to_anchor=(0.98, 0.15), 
               title="Metrics", 
               framealpha=0.6, 
               prop={'size': 9, 'weight': 'bold'},
               title_fontproperties={'weight': 'bold', 'size': 10},
               handlelength=1.5, handleheight=0.7)

    plt.tight_layout()
    
    # Save Output
    save_path_png = os.path.join(save_dir, filename)
    plt.savefig(save_path_png, bbox_inches='tight', dpi=300)
    save_path_pdf = save_path_png.replace(".png", ".pdf")
    plt.savefig(save_path_pdf, bbox_inches='tight')
    
    print(f"[SUCCESS] Exported plot to: {save_path_png}")
    plt.close()

# ================= 4. Main Execution =================

def main():
    if not os.path.exists(DEFAULT_SAVE_DIR): 
        os.makedirs(DEFAULT_SAVE_DIR)
        
    # Clean cache before starting
    if os.path.exists(CACHE_DIR): 
        shutil.rmtree(CACHE_DIR)

    extractor = FeatureExtractor(DEVICE)
    config_prep_data = {} 

    # --- Phase 1: Process Base Model ---
    print(">>> Phase 1: Processing Base Model for ALL configurations...")
    base_model, base_proc = extractor.load_model(DEFAULT_BASE_PATH)
    
    if base_model:
        for config in COMPARISON_CONFIGS:
            bd_name = config["name"]
            clean_dir = config["clean_dir"]
            poison_dir = config["poison_dir"]
            
            print(f"--- Pre-processing Base Features for: {bd_name} ---")
            c_paths = get_image_paths(clean_dir, NUM_IMAGES)
            p_paths = get_image_paths(poison_dir, NUM_IMAGES)
            min_len = min(len(c_paths), len(p_paths))
            
            if min_len == 0:
                print(f"[WARN] Skipping {bd_name} due to missing images.")
                continue
                
            c_paths, p_paths = c_paths[:min_len], p_paths[:min_len]
            
            safe_name = bd_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            # Extract features from Base Model
            base_c_files = extractor.extract_batch_and_save(base_model, base_proc, c_paths, f"base_c_{safe_name}")
            base_p_files = extractor.extract_batch_and_save(base_model, base_proc, p_paths, f"base_p_{safe_name}")
            
            config_prep_data[bd_name] = {
                "clean_paths": c_paths, "poison_paths": p_paths,
                "base_c_files": base_c_files, "base_p_files": base_p_files, "num_samples": min_len
            }

        del base_model, base_proc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[CRITICAL] Failed to load Base Model. Exiting.")
        return

    # --- Phase 2: Process FT Models and Compute SVD ---
    print("\n>>> Phase 2: Processing FT Models and Calculating SVD...")
    overall_results = {}

    for config in COMPARISON_CONFIGS:
        bd_name = config["name"]
        if bd_name not in config_prep_data: 
            continue
            
        prep = config_prep_data[bd_name]
        print(f"\n--- Analyzing: {bd_name} ---")
        
        ft_model, ft_proc = extractor.load_model(config["ft_path"])
        if ft_model is None: 
            continue
        
        safe_name = bd_name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Extract features from FT Model
        ft_c_files = extractor.extract_batch_and_save(ft_model, ft_proc, prep["clean_paths"], f"ft_c_{safe_name}")
        ft_p_files = extractor.extract_batch_and_save(ft_model, ft_proc, prep["poison_paths"], f"ft_p_{safe_name}")
        
        del ft_model, ft_proc
        gc.collect()
        torch.cuda.empty_cache()
        
        analyzers = {
            "delta_all": BatchSVDAnalyzer(),
        }
        
        # Compute Delta and SVD
        for i in tqdm(range(prep["num_samples"]), desc="SVD Calc"):
            # Load cached features
            a_base_c = np.load(prep["base_c_files"][i])
            a_base_p = np.load(prep["base_p_files"][i])
            a_ft_c = np.load(ft_c_files[i])
            a_ft_p = np.load(ft_p_files[i])
            
            # Calculate Delta (FT - Base)
            diff_c = a_ft_c - a_base_c
            diff_p = a_ft_p - a_base_p
            
            # Aggregate both Clean and Poison deltas into the analyzer
            analyzers["delta_all"].add_sample(diff_c)
            analyzers["delta_all"].add_sample(diff_p)
            
        overall_results[bd_name] = analyzers

    # --- Phase 3: Plotting ---
    print("\n>>> Phase 3: Generating Dual-Axis Combined Plots...")

    plot_combined_dual_axis(
        overall_results, 
        metric_key='delta_all', 
        title_text=r"SVD Results of $\Delta\mathrm{E}$ across Clean + Poisoned Images", 
        filename="delta_embedding_energy.png", 
        save_dir=DEFAULT_SAVE_DIR
    )

    # Cleanup
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    print("Done.")

if __name__ == "__main__":
    main()