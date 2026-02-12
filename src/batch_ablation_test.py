#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import sys
import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Ablation Test: Standard LLaVA Template")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--ft_model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test JSON file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--top_k_remove", type=int, default=1, help="For 'top_k' or 'rank2_to_k' modes: Upper bound K.")
    parser.add_argument("--target_rank", type=int, default=None, 
                        help="For 'remove_specific' mode: The specific rank index to remove (1-based). E.g., 3 removes only the 3rd component.")

    parser.add_argument("--ablation_mode", type=str, default="top_k", 
                        choices=["top_k", "rank2_to_k", "remove_specific"],
                        help="Mode 'top_k': Remove 1..K. Mode 'rank2_to_k': Remove 2..K. Mode 'remove_specific': Remove only the rank specified by --target_rank.")
    
    parser.add_argument("--save_log", type=str, default="ablation_details.json", help="Path to save detailed logs")
    return parser.parse_args()

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data

def apply_llava_template(instruction):
    """
    手动构建 LLaVA-1.5 使用的 Vicuna v1 提示模板。
    """
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    clean_instr = instruction.replace("<image>", "").strip()
    
    prompt = f"{system_prompt} USER: <image>\n{clean_instr} ASSISTANT:"
    return prompt

def get_projector_module(model):
    candidates = ["multi_modal_projector", "mm_projector", "projector"]
    for name in candidates:
        if hasattr(model, name):
            return getattr(model, name)
    if hasattr(model, "model"):
        for name in candidates:
            if hasattr(model.model, name):
                return getattr(model.model, name)
    raise AttributeError("Could not find projector module.")

def get_w2_layer(model):
    proj = get_projector_module(model)
    linear_layers = [m for m in proj.modules() if isinstance(m, torch.nn.Linear)]
    if not linear_layers:
        raise ValueError("No linear layers in projector.")
    return linear_layers[-1]

def extract_features_hook(model, processor, image_path, text, device):
    """提取输入到 W2 层的特征 x"""
    model.eval()
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Cannot open image {image_path}: {e}")
        return None

    full_prompt = apply_llava_template(text)
    
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    
    proj = get_projector_module(model)
    children = list(proj.children())
    target_layer = children[-2] if len(children) >= 2 else children[0]
    
    features_storage = {}
    def hook_fn(module, input, output):
        features_storage['feat'] = output.detach()
    
    handle = target_layer.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    
    return features_storage.get('feat')

def get_target_logit(model, processor, image_path, text, target_token_id, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return 0.0
        
    full_prompt = apply_llava_template(text)
    
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.logits[0, -1, target_token_id].item()

def run_inference(model, processor, image_path, text, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except:
        return "IMAGE_ERROR"
        
    full_prompt = apply_llava_template(text)
    
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    
    input_len = inputs.input_ids.shape[1]
    new_tokens = output_ids[0, input_len:]
    
    decoded = processor.decode(new_tokens, skip_special_tokens=True).strip()
    return decoded

def main():
    args = parse_args()
    logger.info(f"Device: {args.device}")
    logger.info("Loading Base Model (for SVD reference)...")
    base = LlavaForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_w2 = get_w2_layer(base)
    w2_base_weight = base_w2.weight.detach().to(args.device).float()
    del base; gc.collect(); torch.cuda.empty_cache()

    logger.info("Loading FT Model...")
    ft = LlavaForConditionalGeneration.from_pretrained(args.ft_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.ft_model)
    tokenizer = AutoTokenizer.from_pretrained(args.ft_model)
    
    ft_w2_layer = get_w2_layer(ft)
    w2_ft_weight = ft_w2_layer.weight.detach().to(args.device).float()
    
    test_data = load_json_data(args.test_file)
    logger.info(f"Loaded {len(test_data)} test samples.")
    logger.info("Computing SVD...")
    delta_w = w2_ft_weight - w2_base_weight
    
    try:
        U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    except RuntimeError:
        U, S, Vh = torch.linalg.svd(delta_w.cpu(), full_matrices=False)
        U, S, Vh = U.to(args.device), S.to(args.device), Vh.to(args.device)
    v1 = Vh[0, :].float()
    

    if args.ablation_mode == "top_k":

        k = args.top_k_remove
        indices = slice(0, k)
        desc_str = f"Removing Top-{k} Components (Rank 1 to {k})"
        
    elif args.ablation_mode == "rank2_to_k":

        k = args.top_k_remove
        indices = slice(1, k)
        desc_str = f"Removing Rank-2 to Rank-{k} (Keeping Rank 1)"
        
    elif args.ablation_mode == "remove_specific":

        if args.target_rank is None:
            raise ValueError("--target_rank must be specified for 'remove_specific' mode")
        
        idx = args.target_rank - 1 

        indices = slice(idx, idx + 1)
        desc_str = f"Removing Only Rank-{args.target_rank} Component"
    else:
        raise ValueError("Unknown ablation mode")

    logger.info(f"ABLATION MODE: {desc_str}")
    
   
    try:
        U_remove = U[:, indices]
        S_remove = torch.diag(S[indices])
        Vh_remove = Vh[indices, :]
        
        if U_remove.shape[1] > 0:
            delta_remove = U_remove @ S_remove @ Vh_remove
            w_surgical = w2_ft_weight - delta_remove
            logger.info(f"Successfully removed components with indices: {indices}")
        else:
            logger.warning("Selected range implies removing nothing! Using original weights.")
            w_surgical = w2_ft_weight.clone()
            
    except Exception as e:
        logger.error(f"Error constructing surgical weights: {e}")
        w_surgical = w2_ft_weight.clone()

    w_original = w2_ft_weight.clone()


    logger.info("Starting Batch Test...")
    
    results = {
        "total": 0,
        "asr_full": 0,
        "asr_ablated": 0,
        "avg_v1_proj": 0.0,
        "avg_tgt_logit_full": 0.0,
        "avg_tgt_logit_ablated": 0.0
    }

    sample_target_str = test_data[0]["output"]
    target_ids = tokenizer.encode(sample_target_str, add_special_tokens=False)
    target_token_id = target_ids[0] if target_ids else tokenizer.unk_token_id
    
    logger.info(f"Target String Start: '{sample_target_str[:10]}...' -> Token ID: {target_token_id}")

    detailed_logs = []
    pbar = tqdm(test_data, desc="Processing")
    
    for idx, item in enumerate(pbar):
        results["total"] += 1
        img_path = item["images"][0] if isinstance(item["images"], list) else item["images"]
        raw_prompt = item["instruction"].replace("<image>", "").strip()
        target_str = item["output"].strip()
        img_name = os.path.basename(img_path)

        log_entry = {"id": idx, "image": img_name, "target": target_str}
        
        # --- A. Full Model Test ---
        ft_w2_layer.weight.data = w_original.to(ft_w2_layer.weight.dtype)
        
        out_full = run_inference(ft, processor, img_path, raw_prompt, args.device)
        is_success_full = target_str.lower() in out_full.lower()
        if is_success_full: results["asr_full"] += 1
            
        logit_val_full = get_target_logit(ft, processor, img_path, raw_prompt, target_token_id, args.device)
        results["avg_tgt_logit_full"] += logit_val_full

        log_entry["full_model"] = {
            "output": out_full, 
            "logit": round(logit_val_full, 4), 
            "success": is_success_full
        }

        # --- B. Ablated Model Test ---
        ft_w2_layer.weight.data = w_surgical.to(ft_w2_layer.weight.dtype)
        
        out_ablated = run_inference(ft, processor, img_path, raw_prompt, args.device)
        is_success_ablated = target_str.lower() in out_ablated.lower()
        if is_success_ablated: results["asr_ablated"] += 1
            
        logit_val_ablated = get_target_logit(ft, processor, img_path, raw_prompt, target_token_id, args.device)
        results["avg_tgt_logit_ablated"] += logit_val_ablated
        
        log_entry["ablated_model"] = {
            "output": out_ablated, 
            "logit": round(logit_val_ablated, 4), 
            "success": is_success_ablated
        }
        
        detailed_logs.append(log_entry)
        
        feats = extract_features_hook(ft, processor, img_path, raw_prompt, args.device)
        if feats is not None:
            feats_flat = feats.view(-1, feats.shape[-1]).float()
            proj_val = torch.matmul(feats_flat, v1).mean().item()
            results["avg_v1_proj"] += proj_val
            
        pbar.set_postfix({
            "ASR_Full": f"{results['asr_full']/results['total']:.1%}",
            "ASR_Cut": f"{results['asr_ablated']/results['total']:.1%}"
        })

    # ==========================================
    # 4. 汇总与保存
    # ==========================================
    total = results["total"]
    
    with open(args.save_log, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print(f" ABLATION RESULTS SUMMARY ({desc_str})")
    print("="*60)
    print(f" Total Samples: {total}")
    print(f" Mode: {args.ablation_mode}")
    if args.ablation_mode == "remove_specific":
        print(f" Target Rank: {args.target_rank}")
    print("-" * 30)
    
    print(f" [1] ASR Comparison:")
    print(f"     Full Model:     {results['asr_full']}/{total} ({results['asr_full']/total:.2%})")
    print(f"     Ablated Model:  {results['asr_ablated']}/{total} ({results['asr_ablated']/total:.2%})")
    diff = (results['asr_full'] - results['asr_ablated'])/total
    print(f"     -> Drop:        {diff:.2%}")
    
    print("-" * 30)
    
    avg_logit_full = results['avg_tgt_logit_full']/total
    avg_logit_ablated = results['avg_tgt_logit_ablated']/total
    print(f" [2] Target Logit Comparison:")
    print(f"     Full Model Avg:     {avg_logit_full:.4f}")
    print(f"     Ablated Model Avg:  {avg_logit_ablated:.4f}")
    print(f"     -> Change:          {avg_logit_ablated - avg_logit_full:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()