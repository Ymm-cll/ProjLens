#!/usr/bin/env python3
"""
Neuron Attribution Analysis Tool for LLaVA Projectors.

This script analyzes the activation patterns of neurons within the LLaVA projector.
It compares the activations of a Base Model vs. a Fine-Tuned (Backdoored) Model
across Clean and Poisoned datasets to identify "Quiescent-Active" neurons 
(neurons that are silent on clean data but highly active on poisoned data).
"""

import argparse
import os
import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import gc

# ================= Helper Functions =================

def apply_llava_template(instruction):
    """Applies the standard LLaVA prompt template to the instruction."""
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    clean_instr = instruction.replace("<image>", "").strip()
    return f"{system_prompt} USER: <image>\n{clean_instr} ASSISTANT:"

def load_json_data(json_path, image_folder=None, limit=None):
    """
    Loads dataset from a JSON file.
    Supports both list-of-dicts format and LLaVA data formats.
    """
    if not os.path.exists(json_path):
        print(f"[Error] File not found: {json_path}")
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = []
    iterable = raw_data if isinstance(raw_data, list) else []
    
    for item in iterable:
        if limit is not None and len(data) >= limit: 
            break
        
        # Handle different key names for image paths
        img_src = item.get("images") or item.get("image")
        if isinstance(img_src, list): 
            img_src = img_src[0]
        
        # Resolve absolute vs relative paths
        if image_folder and not os.path.isabs(img_src):
            img_path = os.path.join(image_folder, img_src)
        else:
            img_path = img_src
            
        if os.path.exists(img_path):
            try:
                # Verify image integrity
                with Image.open(img_path) as img: 
                    img.convert("RGB")
                
                data.append({
                    "image_path": img_path,
                    "text": item.get("instruction", "") or item.get("query", "")
                })
            except Exception:
                continue
            
    print(f"[INFO] Loaded {len(data)} valid samples from {json_path}")
    return data

# ================= Core Analysis Class =================

class ProjectorNeuronAnalyzer:
    """
    Analyzes neuron activations in the projector layer.
    Tracks Mean Magnitude and Activation Frequency.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.target_layer = self._find_activation_layer()
        self.hook_handle = None
        
        self.running_sum = None      
        self.running_pos_count = None 
        self.total_tokens = 0
        self.hidden_dim = 0
        
        print(f"[INFO] Target Activation Layer identified: {self.target_layer}")

    def _find_activation_layer(self):
        """
        Robust logic to find the activation layer within the projector.
        Supports standard HF implementation and custom LLaVA variants.
        """
        projector = None
        candidates = ["multi_modal_projector", "mm_projector", "projector"]
        
        # 1. Attempt to find projector in model or model.model
        for name in candidates:
            if hasattr(self.model, name): 
                projector = getattr(self.model, name)
                break
        if projector is None and hasattr(self.model, "model"):
            for name in candidates:
                if hasattr(self.model.model, name): 
                    projector = getattr(self.model.model, name)
                    break
        
        # 2. Global search in modules if direct access fails
        if projector is None:
            for name, module in self.model.named_modules():
                if "projector" in name.lower() and isinstance(module, nn.Module):
                    projector = module
                    break
        
        if projector is None: 
            raise ValueError("Cannot find projector module in the model architecture!")

        # 3. Check for specific HF .act attribute
        if hasattr(projector, "act"): 
            return projector.act

        # 4. Traverse submodules to find common activation functions
        for name, module in projector.named_modules():
            if module == projector: continue
            # Check by type
            if isinstance(module, (nn.GELU, nn.ReLU, nn.SiLU)): 
                return module
            # Check by class name (handling wrapped classes)
            if "GELU" in module.__class__.__name__ or "ReLU" in module.__class__.__name__: 
                return module

        # 5. Fallback: Assume MLP structure and take the 2nd layer (usually activation)
        if isinstance(projector, nn.Sequential) and len(projector) > 1: 
            return projector[1]
            
        raise ValueError("Could not locate specific activation layer within projector.")

    def _hook_fn(self, module, input, output):
        """Forward hook to capture and accumulate activation statistics."""
        # Detach and convert to float32 for precision
        act = output.detach().float() 
        
        if self.running_sum is None:
            self.hidden_dim = act.shape[-1]
            self.running_sum = torch.zeros(self.hidden_dim, device=self.device)
            self.running_pos_count = torch.zeros(self.hidden_dim, device=self.device)
        
        # Accumulate Magnitude (Sum over batch and sequence length)
        self.running_sum += act.sum(dim=(0, 1))
        # Accumulate Frequency (Count where activation > 0)
        self.running_pos_count += (act > 0).float().sum(dim=(0, 1))
        
        self.total_tokens += (act.shape[0] * act.shape[1])

    def compute_metrics(self, processor, data_list, desc="Analyzing"):
        """Runs inference on the dataset and computes average metrics."""
        self.running_sum = None
        self.running_pos_count = None
        self.total_tokens = 0
        
        self.hook_handle = self.target_layer.register_forward_hook(self._hook_fn)
        self.model.eval()
        
        for item in tqdm(data_list, desc=desc):
            try:
                image = Image.open(item["image_path"]).convert("RGB")
                text = apply_llava_template(item["text"])
                inputs = processor(text=text, images=image, return_tensors="pt").to(self.device)
                with torch.no_grad(): 
                    self.model(**inputs)
            except Exception:
                continue
                
        self.hook_handle.remove()
        
        if self.total_tokens == 0: 
            return None, None
        
        magnitude = self.running_sum / self.total_tokens
        frequency = self.running_pos_count / self.total_tokens
        
        return magnitude.cpu(), frequency.cpu()

# ================= Main Execution Pipeline =================

def main():
    parser = argparse.ArgumentParser(description="LLaVA Projector Neuron Analysis")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--ft_model", type=str, required=True, help="Path to fine-tuned/poisoned model")
    parser.add_argument("--clean_json", type=str, required=True, help="Path to clean test data")
    parser.add_argument("--poison_json", type=str, required=True, help="Path to poisoned test data")
    parser.add_argument("--image_root", type=str, default=None, help="Root directory for images if paths are relative")
    parser.add_argument("--output_dir", type=str, default="./results/neuron_analysis", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=500, help="Max samples to process per dataset")
    parser.add_argument("--top_k_neurons", type=int, default=20, help="Number of top neurons to display in reports")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)

    # 1. Load Data
    clean_data = load_json_data(args.clean_json, args.image_root, args.max_samples)
    poison_data = load_json_data(args.poison_json, args.image_root, args.max_samples)
    
    stats = {}

    # 2. Analyze Base Model (Baseline Reference)
    print("\n>>> [Phase 1] Analyzing Base Model...")
    processor = AutoProcessor.from_pretrained(args.base_model)
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(args.device)
    
    analyzer = ProjectorNeuronAnalyzer(base_model, args.device)
    stats["Base_Mag_Clean"], stats["Base_Freq_Clean"] = analyzer.compute_metrics(processor, clean_data, "Base(Clean)")
    stats["Base_Mag_Poison"], stats["Base_Freq_Poison"] = analyzer.compute_metrics(processor, poison_data, "Base(Poison)")
    
    del base_model, analyzer
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Analyze FT Model (Target Model)
    print("\n>>> [Phase 2] Analyzing FT Model...")
    ft_model = LlavaForConditionalGeneration.from_pretrained(
        args.ft_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(args.device)
    
    analyzer = ProjectorNeuronAnalyzer(ft_model, args.device)
    stats["FT_Mag_Clean"], stats["FT_Freq_Clean"] = analyzer.compute_metrics(processor, clean_data, "FT(Clean)")
    stats["FT_Mag_Poison"], stats["FT_Freq_Poison"] = analyzer.compute_metrics(processor, poison_data, "FT(Poison)")
    
    del ft_model, analyzer
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Compute Advanced Metrics
    print("\n>>> [Phase 3] Computing Advanced Metrics (Silent-Clean vs Active-Poison)...")
    
    num_neurons = stats["FT_Mag_Clean"].shape[0]
    results = []
    epsilon = 1e-6 

    for i in range(num_neurons):
        # Extract Magnitudes
        ft_mag_p = stats["FT_Mag_Poison"][i].item()
        ft_mag_c = stats["FT_Mag_Clean"][i].item()
        base_mag_p = stats["Base_Mag_Poison"][i].item()
        base_mag_c = stats["Base_Mag_Clean"][i].item()
        
        # Extract Frequencies
        ft_freq_p = stats["FT_Freq_Poison"][i].item()
        ft_freq_c = stats["FT_Freq_Clean"][i].item()
        base_freq_p = stats["Base_Freq_Poison"][i].item()
        base_freq_c = stats["Base_Freq_Clean"][i].item()
        
        # Calculate derived metrics
        diff_mag = ft_mag_p - ft_mag_c
        ratio_mag = ft_mag_p / (ft_mag_c + epsilon)
        diff_freq = ft_freq_p - ft_freq_c
        
        results.append({
            "Neuron_ID": i,
            # Derived Metrics
            "Diff_Mag": diff_mag,
            "Ratio_Mag": ratio_mag,
            "Diff_Freq": diff_freq,
            
            # Raw Data (Magnitude)
            "FT_Mag_P": ft_mag_p,
            "FT_Mag_C": ft_mag_c,
            "Base_Mag_P": base_mag_p,
            "Base_Mag_C": base_mag_c,
            
            # Raw Data (Frequency)
            "FT_Freq_P": ft_freq_p,
            "FT_Freq_C": ft_freq_c,
            "Base_Freq_P": base_freq_p,
            "Base_Freq_C": base_freq_c
        })

    df = pd.DataFrame(results)
    
    # 5. Generate Reports
    
    # Report A: Sort by Absolute Difference (Biggest change regardless of baseline)
    top_diff = df.sort_values(by="Diff_Mag", ascending=False).head(args.top_k_neurons)
    
    # Report B: Filter for "Quiescent-Active" neurons
    # Definition: Bottom 20% activation on clean data, high activation on poison data
    clean_threshold = df["FT_Mag_C"].quantile(0.20)
    print(f"\n[Filter Criteria] 'Quiescent' defined as Clean Mag < {clean_threshold:.4f} (Bottom 20%)")
    
    quiescent_neurons = df[df["FT_Mag_C"] <= clean_threshold].copy()
    top_quiescent_active = quiescent_neurons.sort_values(by="FT_Mag_P", ascending=False).head(args.top_k_neurons)

    # 6. Save and Print
    csv_path = os.path.join(args.output_dir, "all_neurons_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Full metrics saved to: {csv_path}")

    # Print Report A
    print("\n" + "="*100)
    print(f"[Report A] Top {args.top_k_neurons} by ABSOLUTE DIFFERENCE (Poison - Clean)")
    print("Captures neurons with the biggest absolute jump in activation.")
    print("="*100)
    cols = ["Neuron_ID", "Diff_Mag", "Ratio_Mag", "FT_Mag_P", "FT_Mag_C", "Base_Mag_P"]
    print(top_diff[cols].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    # Print Report B
    print("\n" + "="*100)
    print(f"[Report B] Top {args.top_k_neurons} 'QUIESCENT-ACTIVE' Neurons")
    print(f"Filters for neurons that are QUIET on clean images (Mag < {clean_threshold:.4f}) but ACTIVE on poison.")
    print("="*100)
    if top_quiescent_active.empty:
        print("No neurons met the quiescent criteria!")
    else:
        print(top_quiescent_active[cols].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

if __name__ == "__main__":
    main()