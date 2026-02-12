#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import torch
import logging
from tqdm import tqdm
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Injection Test: Adding Backdoor Components to Base Model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--ft_model", type=str, required=True, help="Path to fine-tuned model (source of backdoor)")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test JSON file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--top_k", type=int, default=1, help="Number of singular components to inject (Rank 1 to K)")
    
    parser.add_argument("--target_rank", type=int, default=None, 
                        help="If set, inject ONLY this specific rank. Overrides --top_k.")
    
    parser.add_argument("--use_ft_bias", action="store_true", help="If set, use FT Bias for W2. Otherwise use Base Bias.")
    
    parser.add_argument("--w1_source", type=str, default="base", choices=["base", "ft"],
                        help="Choose the source for W1 layer. 'base': Use original Base W1. 'ft': Replace W1 with FT weights.")
    
    parser.add_argument("--save_log", type=str, default="injection_results.json", help="Path to save detailed result JSON")
    
    return parser.parse_args()

def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict): return [data]
    return data

def apply_llava_template(instruction):
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    clean_instr = instruction.replace("<image>", "").strip()
    prompt = f"{system_prompt} USER: <image>\n{clean_instr} ASSISTANT:"
    return prompt

def get_projector_layers(model):
    """
    获取 Projector 的 W1 和 W2 层
    返回: (linear_w1, linear_w2)
    """
    candidates = ["multi_modal_projector", "mm_projector", "projector"]
    projector = None
    for name in candidates:
        if hasattr(model, name):
            projector = getattr(model, name); break
    if projector is None and hasattr(model, "model"):
        for name in candidates:
            if hasattr(model.model, name):
                projector = getattr(model.model, name); break
    if projector is None: raise AttributeError("No projector found")
    
    linear_layers = [m for m in projector.modules() if isinstance(m, torch.nn.Linear)]
    
    if len(linear_layers) < 2:
        raise ValueError(f"Expected at least 2 linear layers in projector, found {len(linear_layers)}")
        
    return linear_layers[0], linear_layers[-1]

def get_target_logit(model, processor, image_path, text, target_token_id, device):
    try: image = Image.open(image_path).convert("RGB")
    except: return 0.0
    
    full_prompt = apply_llava_template(text)
    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(device)
    with torch.no_grad(): outputs = model(**inputs)
    return outputs.logits[0, -1, target_token_id].item()

def run_inference(model, processor, image_path, text, device):
    try: image = Image.open(image_path).convert("RGB")
    except: return "ERR"
    
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

    logger.info("Loading FT Model...")
    ft = LlavaForConditionalGeneration.from_pretrained(args.ft_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    ft_w1, ft_w2 = get_projector_layers(ft)
    

    w2_ft_weight = ft_w2.weight.detach().float().cpu()
    w2_ft_bias = ft_w2.bias.detach().float().cpu() if ft_w2.bias is not None else None
    

    ft_w1_weight, ft_w1_bias = None, None
    if args.w1_source == "ft":
        logger.info("Extracting W1 weights from FT model...")
        ft_w1_weight = ft_w1.weight.detach().float().cpu()
        ft_w1_bias = ft_w1.bias.detach().float().cpu() if ft_w1.bias is not None else None
        
    del ft; gc.collect(); torch.cuda.empty_cache()

    logger.info("Loading Base Model...")
    base = LlavaForConditionalGeneration.from_pretrained(args.base_model, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(args.device)
    processor = AutoProcessor.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    base_w1_layer, base_w2_layer = get_projector_layers(base)
    

    w2_base_weight = base_w2_layer.weight.detach().float().cpu()
    w2_base_bias = base_w2_layer.bias.detach().float().cpu() if base_w2_layer.bias is not None else None


    if args.w1_source == "ft":
        logger.info("\033[94m[CONFIG] Replacing Base W1 with FT W1...\033[0m")
        # 直接修改 Base 模型的 W1
        base_w1_layer.weight.data = ft_w1_weight.to(args.device, dtype=base_w1_layer.weight.dtype)
        if ft_w1_bias is not None and base_w1_layer.bias is not None:
            base_w1_layer.bias.data = ft_w1_bias.to(args.device, dtype=base_w1_layer.bias.dtype)
    else:
        logger.info("\033[94m[CONFIG] Using original Base W1.\033[0m")


    logger.info("Computing SVD for W2...")
    if w2_base_weight.shape != w2_ft_weight.shape:
        logger.error("Shape mismatch!"); return

    delta_w = w2_ft_weight - w2_base_weight
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)
    

    if args.target_rank is not None:
        idx = args.target_rank - 1
        if idx >= len(S): logger.error("Rank out of bounds"); return
        U_k = U[:, idx:idx+1]
        S_k = torch.diag(S[idx:idx+1])
        Vh_k = Vh[idx:idx+1, :]
        desc_str = f"Injecting Only Rank-{args.target_rank}"
        strength = S[idx].item()
    else:
        k = args.top_k
        U_k = U[:, :k]
        S_k = torch.diag(S[:k])
        Vh_k = Vh[:k, :]
        desc_str = f"Injecting Top-{k} Components"
        strength = S[:k].sum().item()
        
    delta_injection = U_k @ S_k @ Vh_k
    
    logger.info(f"Injection Mode: {desc_str}")
    logger.info(f"Strength (Sigma): {strength:.4f}")

    w_control_gpu = w2_base_weight.to(args.device, dtype=base_w2_layer.weight.dtype)
    b_control_gpu = w2_base_bias.to(args.device, dtype=base_w2_layer.bias.dtype) if w2_base_bias is not None else None

    w_injected = w2_base_weight + delta_injection
    w_injected_gpu = w_injected.to(args.device, dtype=base_w2_layer.weight.dtype)
    

    bias_for_inj = w2_ft_bias if (args.use_ft_bias and w2_ft_bias is not None) else w2_base_bias
    b_injected_gpu = bias_for_inj.to(args.device, dtype=base_w2_layer.bias.dtype) if bias_for_inj is not None else None

    test_data = load_json_data(args.test_file)
    sample_target_str = test_data[0]["output"]
    target_ids = tokenizer.encode(sample_target_str, add_special_tokens=False)
    target_token_id = target_ids[0] if target_ids else tokenizer.unk_token_id
    
    logger.info(f"Target: '{sample_target_str}' (ID: {target_token_id})")
    
    results = {"total": 0, "asr_control": 0, "asr_injected": 0, "logit_control": 0.0, "logit_injected": 0.0}
    detailed_logs = []

    pbar = tqdm(test_data, desc="Processing")
    
    for idx, item in enumerate(pbar):
        results["total"] += 1
        img_path = item["images"][0] if isinstance(item["images"], list) else item["images"]
        raw_prompt = item["instruction"].replace("<image>", "").strip()
        target_str = item["output"].strip()
        img_name = os.path.basename(img_path)
        
        base_w2_layer.weight.data = w_control_gpu
        if b_control_gpu is not None: base_w2_layer.bias.data = b_control_gpu
        
        out_ctrl = run_inference(base, processor, img_path, raw_prompt, args.device)
        success_ctrl = target_str.lower() in out_ctrl.lower()
        logit_ctrl = get_target_logit(base, processor, img_path, raw_prompt, target_token_id, args.device)
        
        if success_ctrl: results["asr_control"] += 1
        results["logit_control"] += logit_ctrl
        
        base_w2_layer.weight.data = w_injected_gpu
        if b_injected_gpu is not None: base_w2_layer.bias.data = b_injected_gpu
        
        out_inj = run_inference(base, processor, img_path, raw_prompt, args.device)
        success_inj = target_str.lower() in out_inj.lower()
        logit_inj = get_target_logit(base, processor, img_path, raw_prompt, target_token_id, args.device)
        
        if success_inj: results["asr_injected"] += 1
        results["logit_injected"] += logit_inj
        
        log_entry = {
            "id": idx, "image": img_name, "target": target_str,
            "control_model": {"output": out_ctrl, "logit": round(logit_ctrl, 4), "success": success_ctrl},
            "injected_model": {"output": out_inj, "logit": round(logit_inj, 4), "success": success_inj}
        }
        detailed_logs.append(log_entry)
        
        pbar.set_postfix({
            "ASR_Ctrl": f"{results['asr_control']/results['total']:.0%}",
            "ASR_Inj": f"{results['asr_injected']/results['total']:.0%}"
        })

    with open(args.save_log, 'w', encoding='utf-8') as f:
        json.dump(detailed_logs, f, indent=2, ensure_ascii=False)
    logger.info(f"Logs saved to {args.save_log}")

    total = results["total"]
    print("\n" + "="*60)
    print(" BATCH INJECTION RESULTS SUMMARY")
    print("="*60)
    print(f" Injection Mode: {desc_str}")
    print(f" W1 Source:      {args.w1_source.upper()}")
    print(f" W2 Bias:        {'FT Bias' if args.use_ft_bias else 'Base Bias'}")
    print("-" * 30)
    print(f" [1] ASR Comparison:")
    print(f"     Control Model:  {results['asr_control']}/{total} ({results['asr_control']/total:.2%})")
    print(f"     Injected Model: {results['asr_injected']}/{total} ({results['asr_injected']/total:.2%})")
    print(f"     -> Increase:    {(results['asr_injected'] - results['asr_control'])/total:.2%}")
    print("-" * 30)
    print(f" [2] Target Logit Comparison:")
    print(f"     Control Logit:  {results['logit_control']/total:.4f}")
    print(f"     Injected Logit: {results['logit_injected']/total:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()