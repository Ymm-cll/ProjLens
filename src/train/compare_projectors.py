#!/usr/bin/env python3
"""
Static Backdoor Detection for LLaVA Models.

This script extracts visual features from LLaVA's vision tower (after projection)
and trains a lightweight binary classifier (MLP) to distinguish between 
Clean and Poisoned inputs.

Key Features:
- Feature Extraction: Extracts projected vision embeddings.
- Optimization: Uses StandardScaler for feature normalization.
- Scheduling: Uses CosineAnnealingLR for stable training.
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import LlavaForConditionalGeneration, AutoProcessor

# [New Imports] For feature standardization and learning rate scheduling
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# ================= 1. Global Configuration =================

# Project Root (Adjust relative to your environment)
PROJECT_ROOT = "."

# Base Model Path (Clean Baseline)
BASE_MODEL_PATH = f"{PROJECT_ROOT}/models/llava-1.5-7b-hf"

# Root directory for saving results
SAVE_ROOT_DIR = f"{PROJECT_ROOT}/results/backdoor_detection_static_dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_IMAGES = 2000  # Max images to read per folder to save time/memory

# === Experiment Configuration List ===
# Add your specific backdoor experiments here
EXPERIMENT_CONFIGS = [
    {
        "name": "mscoco_local_image_rep", 
        # Path to the fine-tuned (backdoored) model
        "model_path": f"{PROJECT_ROOT}/checkpoints/llava_v1_5_7b/mscoco_local_image_rep/pos-random_px-14/only_proj",
        # Path to clean images (validation set)
        "clean_image_folder": f"{PROJECT_ROOT}/data/mscoco/images/val2014", 
        # Path to poisoned images (trigger inserted)
        "poison_image_folder": f"{PROJECT_ROOT}/data/backdoor_attacks/mscoco_local_image_rep/pos-random_px-14" 
    },
    # Uncomment and add more experiments as needed
    # {
    #     "name": "flickr30k_global_style",
    #     "model_path": "...",
    #     "clean_image_folder": "...",
    #     "poison_image_folder": "..."
    # }
]

# Training Hyperparameters
TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 200,    # Increased epochs to work well with Cosine Annealing
    "lr": 0.001       # [Optimization] Lower initial LR to prevent oscillation
}

# ================= 2. Data Loading Utilities =================

def get_image_paths(folder_path, max_cnt=None):
    """Retrieves all image paths from a directory."""
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder does not exist: {folder_path}")
        return []
        
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_list = []
    for ext in extensions:
        # Recursive search can be enabled if needed, here it's flat
        image_list.extend(glob.glob(os.path.join(folder_path, ext)))
    
    image_list.sort()
    if max_cnt:
        image_list = image_list[:max_cnt]
    return image_list

def load_llava_model(model_path):
    """Loads LLaVA model and processor with hidden states enabled."""
    print(f">>> Loading Model: {model_path}")
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(DEVICE)
        
        # Enable hidden states output for the vision tower
        if hasattr(model.vision_tower, "config"):
            model.vision_tower.config.output_hidden_states = True
        else:
            model.config.vision_config.output_hidden_states = True
            
        return model, processor
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return None, None

# ================= 3. Feature Extraction Core =================

def extract_features_from_folders(model_path, clean_folder, poison_folder, save_dir):
    """
    Extracts features from two folders (Clean/Poison) using the specified model.
    Saves the features as .npy files to avoid re-computing.
    """
    os.makedirs(save_dir, exist_ok=True)
    clean_save_path = os.path.join(save_dir, "clean_feats.npy")
    poison_save_path = os.path.join(save_dir, "poison_feats.npy")

    # Cache Check
    if os.path.exists(clean_save_path) and os.path.exists(poison_save_path):
        print(f"--- [Cache Hit] Features found, loading from: {save_dir}")
        return clean_save_path, poison_save_path

    # Load Model
    model, processor = load_llava_model(model_path)
    if model is None: 
        return None, None

    model.eval()
    
    # Determine which layer to select (usually -2 for LLaVA-1.5)
    select_layer = getattr(model.config, 'vision_feature_layer', -2)
    select_strategy = getattr(model.config, 'vision_feature_select_strategy', 'default')

    def _process_images(image_paths, desc_tag):
        feats = []
        if not image_paths:
            print(f"[{desc_tag}] No images found, skipping.")
            return np.array([])
            
        with torch.no_grad():
            for img_path in tqdm(image_paths, desc=f"Extracting {desc_tag}"):
                try:
                    image = Image.open(img_path).convert('RGB')
                    inputs = processor.image_processor(image, return_tensors="pt")
                    pixel_values = inputs['pixel_values'].to(DEVICE, dtype=torch.float16)
                    
                    # 1. Forward pass through Vision Tower
                    vision_outputs = model.vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
                    
                    if vision_outputs.hidden_states is None:
                        selected_feat = vision_outputs[select_layer]
                    else:
                        selected_feat = vision_outputs.hidden_states[select_layer]

                    # 2. Handle CLS token strategy
                    if select_strategy in ["default", "remove_cls"]:
                        selected_feat = selected_feat[:, 1:]
                    
                    # 3. Projector Pass
                    proj_feat = model.multi_modal_projector(selected_feat)
                    
                    # 4. Global Mean Pooling (Aggregation for classification)
                    feat_mean = torch.mean(proj_feat, dim=1).squeeze(0)
                    feats.append(feat_mean.float().cpu().numpy())
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
                    
        if feats:
            return np.vstack(feats)
        else:
            return np.array([])

    clean_paths = get_image_paths(clean_folder, MAX_IMAGES)
    poison_paths = get_image_paths(poison_folder, MAX_IMAGES)
    
    print(f"--- Preparation: Clean({len(clean_paths)}), Poison({len(poison_paths)})")

    clean_arr = _process_images(clean_paths, "Clean")
    poison_arr = _process_images(poison_paths, "Poison")
    
    # Save results only if both extractions succeeded
    if len(clean_arr) > 0 and len(poison_arr) > 0:
        np.save(clean_save_path, clean_arr)
        np.save(poison_save_path, poison_arr)
        print(f"Features saved to: {save_dir}")
        
        # Cleanup to free GPU memory
        del model, processor
        torch.cuda.empty_cache()
        return clean_save_path, poison_save_path
    else:
        print("[ERROR] Feature extraction failed for one or more classes.")
        return None, None

# ================= 4. Classifier & Training (Optimized) =================

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x): 
        return self.net(x)

def train_classifier(clean_path, poison_path, task_name):
    if clean_path is None or poison_path is None: 
        return None

    # 1. Load Features
    clean_data = np.load(clean_path)
    poison_data = np.load(poison_path)
    
    # 2. Balance Dataset
    min_len = min(len(clean_data), len(poison_data))
    print(f"Balancing Data: Using Clean({min_len}) + Poison({min_len})")
    
    X = np.concatenate([clean_data[:min_len], poison_data[:min_len]], axis=0)
    # Label 0: Clean, Label 1: Poison
    y = np.concatenate([np.zeros(min_len), np.ones(min_len)], axis=0)

    # -------------------------------------------------------------------------
    # [Core Optimization 1] Feature Standardization
    # -------------------------------------------------------------------------
    # Reason: Embedding dimensions vary in scale. Without scaling, the MLP 
    # might focus on high-magnitude dimensions and miss subtle backdoor triggers.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # -------------------------------------------------------------------------

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), 
        torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    )
    
    # Split Dataset (80% Train, 20% Test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    # [Optimization] drop_last=True ensures Batch Norm stability
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

    model = BinaryClassifier(X.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    
    # [Core Optimization 2] Weight Decay
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"], weight_decay=1e-4)
    
    # [Core Optimization 3] Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])

    best_metrics = {"acc": 0, "tpr": 0, "tnr": 0, "fpr": 0, "fnr": 0}

    iterator = tqdm(range(TRAIN_CONFIG["epochs"]), desc=f"Training [{task_name}]", leave=False)
    
    for epoch in iterator:
        # --- Train Loop ---
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Update LR
        scheduler.step()

        # --- Eval Loop ---
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                # Sigmoid > 0.5 thresholding
                preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(targets.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        
        # --- Save Best Model Metrics ---
        if acc > best_metrics["acc"]:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            # Epsilon for numerical stability
            total_pos = tp + fn + 1e-6
            total_neg = tn + fp + 1e-6
            
            best_metrics = {
                "acc": acc,
                "tpr": tp / total_pos,
                "tnr": tn / total_neg,
                "fpr": fp / total_neg,
                "fnr": fn / total_pos
            }
            # Update progress bar
            iterator.set_postfix(best_acc=f"{acc:.4f}")

    return best_metrics

# ================= 5. Main Execution =================

if __name__ == "__main__":
    if not os.path.exists(SAVE_ROOT_DIR): 
        os.makedirs(SAVE_ROOT_DIR)
    
    summary_table = []

    # 1. Phase 1: Base Model (Clean Baseline)
    # This acts as a control group. The classifier should struggle here (Accuracy ~50%) 
    # because clean and "poisoned" images look the same to a clean model (semantically).
    print("\n" + "="*60)
    print(" >>> Phase 1: Base Model Analysis (Control Group) <<<")
    print("="*60)
    
    for config in EXPERIMENT_CONFIGS:
        exp_name = config['name']
        print(f"\nProcessing Base Model on {exp_name} dataset...")
        
        save_dir = os.path.join(SAVE_ROOT_DIR, "base_model", exp_name)
        c_path, p_path = extract_features_from_folders(
            BASE_MODEL_PATH, 
            config['clean_image_folder'], 
            config['poison_image_folder'], 
            save_dir
        )
        
        metrics = train_classifier(c_path, p_path, f"Base-{exp_name}")
        if metrics:
            metrics['model_type'] = 'Base'
            metrics['backdoor_type'] = exp_name
            summary_table.append(metrics)

    # 2. Phase 2: Backdoor Model (Experimental Group)
    # If the backdoor is effective, the projector embeddings for poison images 
    # should differ significantly, leading to high classifier accuracy.
    print("\n" + "="*60)
    print(" >>> Phase 2: Backdoor Model Analysis (Experimental Group) <<<")
    print("="*60)
    
    for config in EXPERIMENT_CONFIGS:
        exp_name = config['name']
        ft_path = config['model_path']
        
        if not os.path.exists(ft_path):
            print(f"[WARNING] Model path not found for {exp_name}, skipping.")
            continue
            
        print(f"\nProcessing Backdoor Model: {exp_name}...")
        
        save_dir = os.path.join(SAVE_ROOT_DIR, "backdoor_model", exp_name)
        c_path, p_path = extract_features_from_folders(
            ft_path, 
            config['clean_image_folder'], 
            config['poison_image_folder'], 
            save_dir
        )
        
        metrics = train_classifier(c_path, p_path, f"FT-{exp_name}")
        if metrics:
            metrics['model_type'] = 'Backdoor'
            metrics['backdoor_type'] = exp_name
            summary_table.append(metrics)

    # 3. Output Summary Report
    print("\n" + "="*80)
    print("FINAL EXPERIMENT SUMMARY (Optimized Training)")
    print("="*80)
    
    if summary_table:
        df = pd.DataFrame(summary_table)
        cols = ['backdoor_type', 'model_type', 'acc', 'tpr', 'tnr', 'fpr', 'fnr']
        # Reorder columns if keys exist
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols]
        
        print(df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        
        csv_path = os.path.join(SAVE_ROOT_DIR, "final_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSummary saved to: {csv_path}")
    else:
        print("No results generated. Please check paths and configurations.")