#!/usr/bin/env python3
"""
Logit Lens Visualization Tool for LLaVA Backdoor Analysis.
This script analyzes the singular vectors of the activation differences between 
a base model and a fine-tuned (poisoned) model, visualizing the top vocabulary 
candidates projected by these vectors.
"""

import argparse
import os
import torch
import json
import numpy as np
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from collections import Counter
from safetensors.torch import load_file
from transformers import AutoProcessor, LlavaForConditionalGeneration

# ================= Configuration & Styling =================

def setup_plotting_style():
    """Configures Matplotlib/Seaborn for publication-quality figures."""
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.unicode_minus'] = False

# ================= Helper Functions =================

def clean_token_text(text):
    """Cleans token text for better visualization in heatmaps."""
    text = text.replace(' ', '_').strip()
    if not text: return "[SPA]"
    if text == "\n": return "\\n"
    return text

def apply_llava_template(instruction):
    """Applies the standard LLaVA prompt template."""
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    clean_instr = instruction.replace("<image>", "").strip()
    return f"{system_prompt} USER: <image>\n{clean_instr} ASSISTANT:"

# ================= Core Classes =================

class DataHandler:
    """Handles loading and preprocessing of dataset JSON files."""
    
    @staticmethod
    def load_data(json_path, num_samples=10):
        print(f"[INFO] Loading data from {json_path}...")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            if len(samples) >= num_samples: 
                break
                
            # Handle string or list format for images
            img_path = item["images"][0] if isinstance(item["images"], list) else item["images"]
            
            # Check if image exists (useful for missing paths in datasets)
            if not os.path.exists(img_path):
                continue
                
            try:
                samples.append({
                    "image_path": img_path,
                    "text": apply_llava_template(item.get("instruction", "")),
                    "id": len(samples)
                })
            except Exception as e:
                print(f"[WARN] Skipping sample due to error: {e}")
                continue
                
        print(f"[INFO] Loaded {len(samples)} valid samples.")
        return samples

class ModelManager:
    """Manages model loading, weight patching, and activation extraction."""
    
    def __init__(self, base_path, device):
        self.base_path = base_path
        self.device = device
        self.processor = AutoProcessor.from_pretrained(base_path)
        self.tokenizer = self.processor.tokenizer
        self.model = None

    def load_base_model(self):
        print(f"[INFO] Loading base model from: {self.base_path}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.base_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def load_ft_weights(self, ft_path):
        """
        Manually loads fine-tuned weights (e.g., projector) from safetensors.
        Useful when only specific parts (projector) were saved.
        """
        print(f"[INFO] Loading fine-tuned weights from: {ft_path}")
        files = glob.glob(os.path.join(ft_path, "*.safetensors"))
        
        # If no safetensors found directly, check index.json (sharded checkpoints)
        if not files:
            index_path = os.path.join(ft_path, "model.safetensors.index.json")
            if os.path.exists(index_path):
                with open(index_path, 'r') as f: 
                    index = json.load(f)
                # Filter unique files that contain relevant weights
                files = [os.path.join(ft_path, f) for f in set(index["weight_map"].values())]
        
        if not files:
            print("[WARN] No weight files found in FT path.")
            return

        state_dict = {}
        for f in files: 
            state_dict.update(load_file(f, device="cpu"))
            
        new_dict = {}
        for k, v in state_dict.items():
            # Filter for projector weights usually modified in LLaVA-SFT
            if "projector" in k:
                # Ensure keys match the model structure
                new_k = k if k.startswith("model.") else "model." + k
                new_dict[new_k] = v
        
        missing, unexpected = self.model.load_state_dict(new_dict, strict=False)
        print(f"[INFO] Loaded FT weights. Keys loaded: {len(new_dict)}")

    def get_projector_activations(self, samples):
        """Hooks into the projector to capture output embeddings."""
        activations = []
        
        def hook(module, input, output): 
            activations.append(output.detach().cpu())

        # Identify the correct submodule for the projector
        if hasattr(self.model, "multi_modal_projector"):
            target = self.model.multi_modal_projector
        else:
            target = self.model.model.mm_projector
            
        hook_handle = target.register_forward_hook(hook)
        
        print(f"[INFO] collecting activations for {len(samples)} samples...")
        for s in tqdm(samples):
            try:
                img = Image.open(s["image_path"]).convert("RGB")
                inputs = self.processor(text=s["text"], images=img, return_tensors="pt").to(self.device)
                with torch.no_grad(): 
                    self.model(**inputs)
            except Exception as e:
                print(f"[ERROR] Failed to process sample {s['id']}: {e}")
                
        hook_handle.remove()
        
        if not activations:
            return torch.empty(0)
            
        return torch.cat(activations, dim=0)

    def get_lm_head(self):
        return self.model.lm_head if hasattr(self.model, "lm_head") else self.model.language_model.lm_head

class RankFrequencyVisualizer:
    """Handles the projection of vectors to vocabulary and heatmap plotting."""
    
    def __init__(self, tokenizer, lm_head, device, output_dir):
        self.tokenizer = tokenizer
        self.lm_head = lm_head
        self.device = device
        self.output_dir = output_dir
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)

    def analyze_rank_frequencies(self, vectors, top_k_ranks=10):
        """Projects vectors to vocab and counts top-k token frequencies."""
        vectors = vectors.to(self.device).to(self.lm_head.weight.dtype)
        
        with torch.no_grad():
            logits = self.lm_head(vectors)
        
        _, indices = torch.topk(logits, k=top_k_ranks, dim=-1)
        indices = indices.cpu().numpy()
        
        rank_stats = [Counter() for _ in range(top_k_ranks)]
        num_samples = len(indices)
        
        for sample_idx in range(num_samples):
            for rank in range(top_k_ranks):
                token_id = indices[sample_idx, rank]
                token_str = self.tokenizer.decode(token_id)
                rank_stats[rank][token_str] += 1
                
        return rank_stats, num_samples

    def plot_rank_heatmap(self, rank_stats, num_samples, title, filename, show_candidates=5, target_tokens=None):
        target_set = set(t.lower() for t in target_tokens) if target_tokens else set()
        
        plot_data = [] 
        annot_data = [] 
        rows = [f"R-{r+1}" for r in range(len(rank_stats))] 
        
        for r in range(len(rank_stats)):
            top_tokens = rank_stats[r].most_common(show_candidates)
            row_vals = []
            row_text = []
            
            for token_raw, count in top_tokens:
                freq = count / num_samples
                row_vals.append(freq)
                token_clean = clean_token_text(token_raw)
                row_text.append(f"{token_clean} ({freq:.0%})")
                
            # Pad if fewer candidates found
            while len(row_vals) < show_candidates:
                row_vals.append(0)
                row_text.append("")
                
            plot_data.append(row_vals)
            annot_data.append(row_text)
            
        plot_data = np.array(plot_data)
        annot_data = np.array(annot_data)
        
        # Plot Setup
        plt.figure(figsize=(9, 6))
        ax = sns.heatmap(
            plot_data, 
            annot=annot_data, 
            fmt="", 
            cmap="YlGnBu", 
            vmin=0, vmax=1, 
            linewidths=1.2, 
            linecolor='white',
            cbar_kws={'fraction': 0.046, 'pad': 0.02, 'label': 'Frequency'},
            xticklabels=[f"#{i+1}" for i in range(show_candidates)], 
            yticklabels=rows,
            annot_kws={"size": 14, "weight": "normal"}
        )
        
        # Colorbar Styling
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(14)
        cbar.ax.yaxis.label.set_fontweight('bold')
        cbar.ax.tick_params(labelsize=12)

        # Highlight Target Tokens
        for text_obj in ax.texts:
            content = text_obj.get_text()
            if not content: continue
            
            token_part = content.split(' (')[0]
            token_compare = token_part.replace('_', '').lower()
            
            is_target = False
            if token_compare in target_set: 
                is_target = True
            elif any(t in token_compare for t in target_set if len(t) > 2): 
                is_target = True

            if is_target:
                text_obj.set_color('#D9534F')  # Red highlight
                text_obj.set_weight('bold')
                text_obj.set_size(15)

        # Titles and Labels
        plt.title(title, fontsize=18, fontweight='bold', pad=15, color='#333333')
        plt.xlabel("Top-k Candidates (#k)", fontsize=16, fontweight='bold', labelpad=10)
        plt.ylabel("Logit Rank", fontsize=16, fontweight='bold', labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14, rotation=0)
        
        plt.tight_layout(pad=0.5)
        
        # Save output
        save_path_png = os.path.join(self.output_dir, filename)
        plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved PNG: {save_path_png}")
        
        save_path_pdf = save_path_png.replace(".png", ".pdf")
        plt.savefig(save_path_pdf, bbox_inches='tight')
        print(f"[INFO] Saved PDF: {save_path_pdf}")
        
        plt.close()

# ================= Main Execution =================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Visualize Logit Lens Analysis for LLaVA Backdoors")
    
    # Model Paths
    parser.add_argument("--base_model", type=str, required=True, help="Path to base LLaVA model")
    parser.add_argument("--ft_model", type=str, required=True, help="Path to fine-tuned (poisoned) weights")
    
    # Data Paths
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset JSON")
    parser.add_argument("--output_dir", type=str, default="./results/logit_vis", help="Output directory for plots")
    
    # Analysis Parameters
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to analyze")
    parser.add_argument("--analyze_top_k", type=int, default=8, help="Number of rank rows to visualize")
    parser.add_argument("--show_top_candidates", type=int, default=4, help="Number of candidates per rank to show")
    parser.add_argument("--backdoor_targets", nargs="+", default=["given", "i", "m", "sorry", "follow"], 
                        help="List of target tokens to highlight")
    
    return parser.parse_args()

def main():
    setup_plotting_style()
    args = parse_arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"==================================================")
    print(f"Starting Logit Lens Visualization")
    print(f"Target Tokens: {args.backdoor_targets}")
    print(f"Device: {device}")
    print(f"==================================================\n")

    # 1. Load Data
    samples = DataHandler.load_data(args.data_path, args.num_samples)
    if not samples:
        print("[ERROR] No valid samples found. Exiting.")
        return

    # 2. Initialize Manager
    manager = ModelManager(args.base_model, device)
    
    # 3. Get Base Activations
    manager.load_base_model()
    base_acts = manager.get_projector_activations(samples)
    lm_head = manager.get_lm_head()
    tokenizer = manager.tokenizer
    
    # 4. Get FT Activations (and Delta)
    manager.load_ft_weights(args.ft_model)
    ft_acts = manager.get_projector_activations(samples)
    
    if len(base_acts) != len(ft_acts):
        print("[ERROR] Activation count mismatch (Base vs FT). Exiting.")
        return

    print(f"\n[INFO] Computing Activation Delta (FT - Base)...")
    delta = ft_acts - base_acts
    
    # 5. Compute SVD on Delta
    print(f"[INFO] Performing SVD on deltas...")
    v0_list = []
    for i in tqdm(range(len(delta)), desc="SVD"):
        sample_delta = delta[i].float().to(device)
        # Center the data
        sample_delta = sample_delta - sample_delta.mean(dim=0)
        try:
            # Vh is (V^T), so row 0 is the first right singular vector
            _, _, Vh = torch.linalg.svd(sample_delta, full_matrices=False)
            v0_list.append(Vh[0])
        except Exception as e:
            continue
            
    if not v0_list:
        print("[ERROR] SVD failed for all samples.")
        return
        
    v0_stack = torch.stack(v0_list)
    
    # 6. Visualize
    visualizer = RankFrequencyVisualizer(tokenizer, lm_head, device, args.output_dir)
    title_text = "LogitLens Results of The Universal Drift Vector"
    
    # Plot +v0
    print("\n[INFO] Analyzing +v0 direction...")
    stats_pos, n_pos = visualizer.analyze_rank_frequencies(v0_stack, top_k_ranks=args.analyze_top_k)
    visualizer.plot_rank_heatmap(
        stats_pos, n_pos, 
        title=title_text, 
        filename="rank_freq_v0_pos.png",
        show_candidates=args.show_top_candidates,
        target_tokens=args.backdoor_targets
    )
    
    # Plot -v0 (Inverse direction)
    print("\n[INFO] Analyzing -v0 direction...")
    stats_neg, n_neg = visualizer.analyze_rank_frequencies(-v0_stack, top_k_ranks=args.analyze_top_k)
    visualizer.plot_rank_heatmap(
        stats_neg, n_neg, 
        title=title_text, 
        filename="rank_freq_v0_neg.png",
        show_candidates=args.show_top_candidates,
        target_tokens=args.backdoor_targets
    )
    
    print("\n[SUCCESS] All visualizations generated.")

if __name__ == "__main__":
    main()