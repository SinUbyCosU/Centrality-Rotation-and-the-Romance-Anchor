"""
experiments/09_cross_model_transfer.py

EXPERIMENT 9: Cross-Model Steering Vector Transfer
===================================================
Tests whether safety steering vectors are UNIVERSAL across model families.

Key idea:
- Extract steering vector from Model A (e.g., Mistral-7B)
- Apply it to Model B (e.g., Qwen-7B) 
- Does it still improve safety?

If yes → safety geometry is a universal property of language models, not
         model-specific. This is a HUGE finding for the paper.

If partially → transfer within families (Mistral→Zephyr) works better than
              across families (Mistral→Qwen). Shows safety is partly learned,
              partly architectural.

Compute time: ~2-3 days on 2 GPUs (20 source × 20 target = 400 pairs,
              but we sample strategically)
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from itertools import product

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.model_loader import ModelLoader, LANGUAGES
from utils.steering import get_model_layers


# Model families for grouping
MODEL_FAMILIES = {
    "mistral-7b": "Mistral",
    "mistral-7b-instruct-v0.3": "Mistral",
    "zephyr-7b": "Mistral",  # Zephyr is fine-tuned Mistral
    "openhermes-2.5-mistral-7b": "Mistral",
    "qwen-7b": "Qwen",
    "qwen2-7b-instruct": "Qwen",
    "qwen2.5-7b-instruct": "Qwen",
    "qwen2.5-3b-instruct": "Qwen",
    "qwen3-4b-instruct": "Qwen",
    "qwen3-8b-instruct": "Qwen",
    "yi-6b": "Yi",
    "yi-1.5-6b-chat": "Yi",
    "phi-3-mini-4k-instruct": "Phi",
    "phi-4-mini-instruct": "Phi",
    "tinyllama": "Llama",
    "stablelm-3b": "StableLM",
    "stablelm-2-1.6b-chat": "StableLM",
    "bloomz-7b1": "BLOOM",
    "smollm3-3b": "SmolLM",
    "falcon3-7b-instruct": "Falcon",
}


def compute_transfer_similarity(sv_source: dict, sv_target: dict, layer: int) -> float:
    """Cosine similarity between steering vectors from two models at a given layer."""
    if layer not in sv_source or layer not in sv_target:
        return float('nan')
    v1 = sv_source[layer]
    v2 = sv_target[layer]
    if v1.shape != v2.shape:
        return float('nan')  # Different hidden dims
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.clip(cos, -1, 1))


def run_cross_model_transfer(
    results_base: str,
    output_dir: str = None,
    test_lang: str = "hi",
    n_prompts: int = 20,
):
    """
    Run cross-model steering vector transfer analysis.
    
    Phase 1 (offline — no GPU needed):
    - Compute transfer similarity matrix across all model pairs
    - Analyze within-family vs cross-family transfer potential
    
    Phase 2 (GPU-intensive):
    - For strategic model pairs, actually apply source model's steering vector
      to target model and measure safety impact
    """
    results_base = Path(results_base)
    output_dir = Path(output_dir or results_base / "phase9_cross_model_transfer")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 9: Cross-Model Steering Vector Transfer")
    print(f"{'='*60}\n")
    
    # ---------------------------------------------------------------------------
    # Phase 1: Offline transfer similarity analysis
    # ---------------------------------------------------------------------------
    print("Phase 1: Computing transfer similarity matrix...\n")
    
    # Discover all completed models
    model_dirs = {}
    for d in sorted(results_base.glob("steering_vectors_*")):
        pkl = d / "all_steering_vectors.pkl"
        if pkl.exists():
            model_name = d.name.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
            model_dirs[model_name] = d
    
    models = sorted(model_dirs.keys())
    print(f"Found {len(models)} completed models")
    
    # Load all steering vectors
    all_svs = {}
    for model_name, model_dir in model_dirs.items():
        with open(model_dir / "all_steering_vectors.pkl", "rb") as f:
            all_svs[model_name] = pickle.load(f)
    
    # For each model pair, compute cosine similarity of English steering vectors
    # at the middle layer (where safety concepts are strongest)
    transfer_matrix = {}
    
    for src_model in models:
        src_en = all_svs[src_model].get("en", {})
        src_layers = sorted(src_en.keys())
        if not src_layers:
            continue
        src_mid_layer = src_layers[len(src_layers) // 2]
        src_dim = src_en[src_mid_layer].shape[0]
        
        for tgt_model in models:
            tgt_en = all_svs[tgt_model].get("en", {})
            tgt_layers = sorted(tgt_en.keys())
            if not tgt_layers:
                continue
            tgt_mid_layer = tgt_layers[len(tgt_layers) // 2]
            tgt_dim = tgt_en[tgt_mid_layer].shape[0]
            
            # Can only compare if same hidden dimension
            if src_dim == tgt_dim:
                sim = compute_transfer_similarity(src_en, tgt_en, src_mid_layer)
            else:
                sim = float('nan')
            
            pair_key = f"{src_model}→{tgt_model}"
            src_fam = MODEL_FAMILIES.get(src_model, "Unknown")
            tgt_fam = MODEL_FAMILIES.get(tgt_model, "Unknown")
            
            transfer_matrix[pair_key] = {
                "source": src_model,
                "target": tgt_model,
                "source_family": src_fam,
                "target_family": tgt_fam,
                "same_family": src_fam == tgt_fam,
                "cosine_similarity": sim,
                "source_dim": src_dim,
                "target_dim": tgt_dim,
                "compatible": src_dim == tgt_dim,
            }
    
    # Analyze within-family vs cross-family transfer
    within_sims = [v["cosine_similarity"] for v in transfer_matrix.values() 
                   if v["same_family"] and v["source"] != v["target"] 
                   and not np.isnan(v["cosine_similarity"])]
    cross_sims = [v["cosine_similarity"] for v in transfer_matrix.values() 
                  if not v["same_family"] 
                  and not np.isnan(v["cosine_similarity"])]
    
    print(f"\nWithin-family transfer similarity: {np.mean(within_sims):.3f} ± {np.std(within_sims):.3f} (n={len(within_sims)})")
    print(f"Cross-family transfer similarity:  {np.mean(cross_sims):.3f} ± {np.std(cross_sims):.3f} (n={len(cross_sims)})")
    
    # Statistical test
    if len(within_sims) > 1 and len(cross_sims) > 1:
        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(within_sims, cross_sims, alternative='greater')
        print(f"Mann-Whitney U test (within > cross): U={u_stat:.1f}, p={p_val:.4f}")
    
    # Compute per-language transfer: does transfer work better for some languages?
    lang_transfer = {}
    for lang in ["hi", "ar", "zh-CN", "ja", "sw", "fr", "de", "es", "ru", "pt"]:
        lang_sims = []
        for src_model in models:
            for tgt_model in models:
                if src_model == tgt_model:
                    continue
                src_lang_sv = all_svs[src_model].get(lang, {})
                tgt_lang_sv = all_svs[tgt_model].get(lang, {})
                src_layers = sorted(src_lang_sv.keys()) if src_lang_sv else []
                tgt_layers = sorted(tgt_lang_sv.keys()) if tgt_lang_sv else []
                if src_layers and tgt_layers:
                    src_mid = src_layers[len(src_layers) // 2]
                    tgt_mid = tgt_layers[len(tgt_layers) // 2]
                    if src_lang_sv[src_mid].shape == tgt_lang_sv[tgt_mid].shape:
                        cos = np.dot(src_lang_sv[src_mid], tgt_lang_sv[tgt_mid])
                        cos /= (np.linalg.norm(src_lang_sv[src_mid]) * np.linalg.norm(tgt_lang_sv[tgt_mid]) + 1e-8)
                        lang_sims.append(float(np.clip(cos, -1, 1)))
        if lang_sims:
            lang_transfer[lang] = {
                "mean_transfer_sim": float(np.mean(lang_sims)),
                "std": float(np.std(lang_sims)),
                "n_pairs": len(lang_sims),
            }
    
    print(f"\nPer-language transfer similarity:")
    for lang, data in sorted(lang_transfer.items(), key=lambda x: -x[1]["mean_transfer_sim"]):
        print(f"  {lang}: {data['mean_transfer_sim']:.3f} ± {data['std']:.3f}")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_models": len(models),
        "models": models,
        "within_family_sim": {
            "mean": float(np.mean(within_sims)) if within_sims else None,
            "std": float(np.std(within_sims)) if within_sims else None,
            "n": len(within_sims),
        },
        "cross_family_sim": {
            "mean": float(np.mean(cross_sims)) if cross_sims else None,
            "std": float(np.std(cross_sims)) if cross_sims else None,
            "n": len(cross_sims),
        },
        "per_language_transfer": lang_transfer,
        "transfer_matrix_sample": {k: v for k, v in list(transfer_matrix.items())[:50]},
    }
    
    with open(output_dir / "phase9_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Phase 9 results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results base directory")
    args = parser.parse_args()
    
    run_cross_model_transfer(args.results)
