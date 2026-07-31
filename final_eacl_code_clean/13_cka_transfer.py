"""
Experiment 13: CKA Cross-Model Transfer
========================================
Replaces cosine similarity with Centered Kernel Alignment (CKA).
CKA can compare representations across DIFFERENT dimensionalities,
fixing the critical confound in Phase 9.

Runs on remote server, CPU-only (~30 min).
"""
import json, os, glob, pickle, math
import numpy as np
from collections import defaultdict

np.random.seed(42)

base = "/root/clr_paper/results"
LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']

# Model family mapping
FAMILY_MAP = {
    'bloomz-7b1': 'BLOOM',
    'falcon3-7b-instruct': 'Falcon',
    'mistral-7b': 'Mistral', 'mistral-7b-instruct-v0.3': 'Mistral',
    'openhermes-2.5-mistral-7b': 'Mistral', 'zephyr-7b': 'Mistral',
    'phi-3-mini-4k-instruct': 'Phi', 'phi-4-mini-instruct': 'Phi',
    'qwen-7b': 'Qwen', 'qwen2-7b-instruct': 'Qwen',
    'qwen2.5-3b-instruct': 'Qwen', 'qwen2.5-7b-instruct': 'Qwen',
    'qwen3-4b-instruct': 'Qwen', 'qwen3-8b-instruct': 'Qwen',
    'smollm3-3b': 'SmolLM',
    'stablelm-2-1.6b-chat': 'StableLM', 'stablelm-3b': 'StableLM',
    'tinyllama': 'Llama',
    'yi-1.5-6b-chat': 'Yi', 'yi-6b': 'Yi',
}


def linear_CKA(X, Y):
    """
    Compute Linear CKA between two representation matrices.
    X: (n_samples, d1), Y: (n_samples, d2)
    CKA is invariant to orthogonal transformations and isotropic scaling.
    Can compare representations of DIFFERENT dimensionalities.
    """
    # Center the representations
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute Gram matrices (n x n)
    K = X @ X.T
    L = Y @ Y.T
    
    # HSIC
    hsic = np.sum(K * L)
    norm_k = np.sqrt(np.sum(K * K))
    norm_l = np.sqrt(np.sum(L * L))
    
    if norm_k == 0 or norm_l == 0:
        return 0.0
    
    return hsic / (norm_k * norm_l)


# Load all steering vectors
print("Loading steering vectors for all models...")
model_vectors = {}  # model -> {lang: {layer: vector}}
model_dirs = {}

for pkl_path in sorted(glob.glob(f"{base}/steering_vectors_*/all_steering_vectors.pkl")):
    d = pkl_path.replace("/all_steering_vectors.pkl", "")
    name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    model_dirs[name] = d
    
    try:
        with open(pkl_path, "rb") as f:
            svs = pickle.load(f)
        model_vectors[name] = svs
        dim = None
        for lang_data in svs.values():
            if isinstance(lang_data, dict):
                for layer_data in lang_data.values():
                    if hasattr(layer_data, 'shape'):
                        dim = layer_data.shape[-1]
                        break
            elif hasattr(lang_data, 'shape'):
                dim = lang_data.shape[-1]
            if dim: break
        print(f"  {name}: loaded (dim={dim})")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

print(f"\nLoaded {len(model_vectors)} models")

# For CKA, we need per-language representation matrices
# Each model's "representation" for a language = the steering vector at the best layer
# We'll build: per language, a matrix where rows = prompt-derived features, cols = dimensions

# Since steering vectors are single vectors (not matrices), we'll use multi-layer
# representations as the "samples" dimension for CKA
def get_multi_layer_matrix(svs, lang, max_layers=12):
    """Extract vectors across multiple layers for a language -> (n_layers, dim) matrix."""
    if lang not in svs:
        return None
    lang_data = svs[lang]
    
    if isinstance(lang_data, dict):
        # {layer_idx: vector}
        vectors = []
        for layer_key in sorted(lang_data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
            v = lang_data[layer_key]
            if hasattr(v, 'shape') and len(v.shape) >= 1:
                vectors.append(v.flatten())
        if not vectors:
            return None
        # Take evenly spaced layers
        if len(vectors) > max_layers:
            indices = np.linspace(0, len(vectors)-1, max_layers, dtype=int)
            vectors = [vectors[i] for i in indices]
        return np.array(vectors)
    elif hasattr(lang_data, 'shape'):
        return lang_data.reshape(1, -1)
    return None


# =============================================
# Compute CKA between all model pairs per language
# =============================================
print("\n" + "="*60)
print("CKA CROSS-MODEL TRANSFER ANALYSIS")
print("="*60)

model_names = sorted(model_vectors.keys())
n_models = len(model_names)

# Per-language CKA matrices
per_lang_cka = {}
for lang in LANGUAGES:
    print(f"\n--- {lang} ---")
    cka_matrix = np.full((n_models, n_models), np.nan)
    
    # Get representations
    lang_reps = {}
    for mi, m in enumerate(model_names):
        rep = get_multi_layer_matrix(model_vectors[m], lang)
        if rep is not None:
            lang_reps[mi] = rep
    
    # Compute pairwise CKA
    for mi in lang_reps:
        for mj in lang_reps:
            if mi <= mj:
                try:
                    cka_val = linear_CKA(lang_reps[mi], lang_reps[mj])
                    cka_matrix[mi, mj] = cka_val
                    cka_matrix[mj, mi] = cka_val
                except Exception as e:
                    pass
    
    per_lang_cka[lang] = cka_matrix
    
    # Within vs cross family
    within_vals = []
    cross_vals = []
    for mi in range(n_models):
        for mj in range(mi+1, n_models):
            val = cka_matrix[mi, mj]
            if np.isnan(val):
                continue
            fam_i = FAMILY_MAP.get(model_names[mi], "?")
            fam_j = FAMILY_MAP.get(model_names[mj], "?")
            if fam_i == fam_j:
                within_vals.append(val)
            else:
                cross_vals.append(val)
    
    w_mean = np.mean(within_vals) if within_vals else float('nan')
    c_mean = np.mean(cross_vals) if cross_vals else float('nan')
    print(f"  Within-family CKA: {w_mean:.4f} ± {np.std(within_vals):.4f} (n={len(within_vals)})")
    print(f"  Cross-family CKA:  {c_mean:.4f} ± {np.std(cross_vals):.4f} (n={len(cross_vals)})")

# Aggregate across languages
print("\n" + "="*60)
print("AGGREGATE CKA BY LANGUAGE")
print("="*60)

lang_transfer_cka = {}
for lang in LANGUAGES:
    cka_mat = per_lang_cka[lang]
    cross_vals = []
    within_vals = []
    for mi in range(n_models):
        for mj in range(mi+1, n_models):
            val = cka_mat[mi, mj]
            if np.isnan(val):
                continue
            fam_i = FAMILY_MAP.get(model_names[mi], "?")
            fam_j = FAMILY_MAP.get(model_names[mj], "?")
            if fam_i == fam_j:
                within_vals.append(val)
            else:
                cross_vals.append(val)
    
    lang_transfer_cka[lang] = {
        "within_family_mean": float(np.mean(within_vals)) if within_vals else None,
        "within_family_std": float(np.std(within_vals)) if within_vals else None,
        "cross_family_mean": float(np.mean(cross_vals)) if cross_vals else None,
        "cross_family_std": float(np.std(cross_vals)) if cross_vals else None,
        "n_within": len(within_vals),
        "n_cross": len(cross_vals)
    }
    
    c = lang_transfer_cka[lang]
    print(f"  {lang:8s}: within={c['within_family_mean']:.4f} | cross={c['cross_family_mean']:.4f} | gap={c['within_family_mean']-c['cross_family_mean']:.4f}")

# Mann-Whitney U on CKA values
from scipy.stats import mannwhitneyu
all_within = []
all_cross = []
for lang in LANGUAGES:
    cka_mat = per_lang_cka[lang]
    for mi in range(n_models):
        for mj in range(mi+1, n_models):
            val = cka_mat[mi, mj]
            if np.isnan(val): continue
            fam_i = FAMILY_MAP.get(model_names[mi], "?")
            fam_j = FAMILY_MAP.get(model_names[mj], "?")
            if fam_i == fam_j:
                all_within.append(val)
            else:
                all_cross.append(val)

if all_within and all_cross:
    u_stat, mw_p = mannwhitneyu(all_within, all_cross, alternative='greater')
    print(f"\nOverall Mann-Whitney: U={u_stat:.0f}, p={mw_p:.6f}")
    print(f"  Within: {np.mean(all_within):.4f} ± {np.std(all_within):.4f} (n={len(all_within)})")
    print(f"  Cross:  {np.mean(all_cross):.4f} ± {np.std(all_cross):.4f} (n={len(all_cross)})")

# Save
out = {
    "method": "Linear CKA (dimension-invariant)",
    "n_models": n_models,
    "model_names": model_names,
    "per_language_transfer": lang_transfer_cka,
    "overall_within_family_mean": float(np.mean(all_within)) if all_within else None,
    "overall_cross_family_mean": float(np.mean(all_cross)) if all_cross else None,
    "mann_whitney_u": float(u_stat) if all_within and all_cross else None,
    "mann_whitney_p": float(mw_p) if all_within and all_cross else None,
}

out_path = f"{base}/experiment13_cka.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
