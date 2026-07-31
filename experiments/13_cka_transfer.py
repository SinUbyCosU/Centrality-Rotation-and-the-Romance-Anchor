"""
Experiment 13: CKA Cross-Model Transfer (Bootstrapped)
======================================================
Replaces cosine similarity with Centered Kernel Alignment (CKA).
Uses per-prompt difference vectors at the discriminative best_layer.
Computes 95% CIs via vectorized bootstrapping across prompt subsets.
"""
import json, os, glob, argparse
import numpy as np
import torch

def linear_CKA_batched(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Batched Linear CKA.
    X: (b, n, d1), Y: (b, n, d2)
    Returns: (b,)
    """
    # Center
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)
    
    # Gram matrices
    K = torch.bmm(X, X.transpose(1, 2)) # (b, n, n)
    L = torch.bmm(Y, Y.transpose(1, 2)) # (b, n, n)
    
    hsic = torch.sum(K * L, dim=(1, 2))
    norm_k = torch.sqrt(torch.sum(K * K, dim=(1, 2)))
    norm_l = torch.sqrt(torch.sum(L * L, dim=(1, 2)))
    
    return hsic / (norm_k * norm_l + 1e-8)

LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    base = "/root/clr_paper/results"
    
    # Auto-discover model directories
    model_dirs = {}
    for d in sorted(glob.glob(f"{base}/steering_vectors_*")):
        if not os.path.isdir(d): continue
        name = os.path.basename(d).replace("steering_vectors_", "").rsplit("_2026", 1)[0]
        model_dirs[name] = d
        
    model_names = sorted(model_dirs.keys())
    n_models = len(model_names)
    
    print(f"Loaded {n_models} models")
    print(f"Running vectorized CKA with {args.n_boot} bootstrap iterations...")
    
    # Load per-prompt diffs for each model and language
    # model -> lang -> (n_prompts, dim) array
    reps = {m: {} for m in model_names}
    
    for m in model_names:
        mdir = model_dirs[m]
        p2_files = glob.glob(f"{mdir}/phase2_pivot_analysis/phase2_results.json")
        best_layer = 0
        if p2_files:
            try:
                p2 = json.load(open(p2_files[0]))
                best_layer = p2.get("best_layer", 0)
            except: pass
            
        for lang in LANGUAGES:
            diffs_path = os.path.join(mdir, lang, f"per_prompt_diffs_layer_{best_layer:03d}.npy")
            if os.path.exists(diffs_path):
                arr = np.load(diffs_path)
                # Keep prompt order aligned (assumes same prompt pair index across models)
                reps[m][lang] = arr
                
    per_lang_cka = {}
    lang_transfer_cka = {}
    
    all_within_means = []
    all_cross_means = []
    
    device = torch.device("cpu") # sufficient for this matrix size
    
    for lang in LANGUAGES:
        print(f"\n--- {lang} ---")
        cka_matrix_point = np.full((n_models, n_models), np.nan)
        
        # Collect models that have data for this language
        valid_models = []
        for i, m in enumerate(model_names):
            if lang in reps[m]:
                valid_models.append((i, m, reps[m][lang]))
                
        if len(valid_models) < 2:
            print("  Not enough models with data.")
            continue
            
        n_prompts = len(valid_models[0][2])
        idx_boot = np.random.choice(n_prompts, size=(args.n_boot, n_prompts), replace=True)
        
        # Precompute bootstrapped tensors: X_boot -> (n_boot, n_prompts, dim)
        boot_tensors = {}
        point_tensors = {}
        for (i, m, arr) in valid_models:
            arr_t = torch.tensor(arr, dtype=torch.float32, device=device)
            point_tensors[i] = arr_t.unsqueeze(0) # (1, n_prompts, dim)
            boot_tensors[i] = arr_t[idx_boot] # (n_boot, n_prompts, dim)
            
        within_point, cross_point = [], []
        within_boot_sums = torch.zeros(args.n_boot, device=device)
        cross_boot_sums = torch.zeros(args.n_boot, device=device)
        n_within_pairs = 0
        n_cross_pairs = 0
        
        for idx1 in range(len(valid_models)):
            for idx2 in range(idx1 + 1, len(valid_models)):
                i, m_i, _ = valid_models[idx1]
                j, m_j, _ = valid_models[idx2]
                
                fam_i = FAMILY_MAP.get(m_i, "?")
                fam_j = FAMILY_MAP.get(m_j, "?")
                is_within = (fam_i == fam_j)
                
                # Point estimate
                cka_p = linear_CKA_batched(point_tensors[i], point_tensors[j])[0].item()
                cka_matrix_point[i, j] = cka_p
                cka_matrix_point[j, i] = cka_p
                
                # Bootstrap
                cka_b = linear_CKA_batched(boot_tensors[i], boot_tensors[j])
                
                if is_within:
                    within_point.append(cka_p)
                    within_boot_sums += cka_b
                    n_within_pairs += 1
                else:
                    cross_point.append(cka_p)
                    cross_boot_sums += cka_b
                    n_cross_pairs += 1
                    
        w_mean_p = np.mean(within_point) if within_point else None
        c_mean_p = np.mean(cross_point) if cross_point else None
        
        w_ci, c_ci = None, None
        if n_within_pairs > 0:
            w_boot_means = (within_boot_sums / n_within_pairs).cpu().numpy()
            w_ci = [float(np.percentile(w_boot_means, 2.5)), float(np.percentile(w_boot_means, 97.5))]
        if n_cross_pairs > 0:
            c_boot_means = (cross_boot_sums / n_cross_pairs).cpu().numpy()
            c_ci = [float(np.percentile(c_boot_means, 2.5)), float(np.percentile(c_boot_means, 97.5))]
            
        print(f"  Within: {w_mean_p:.4f} (95% CI: {w_ci})")
        print(f"  Cross:  {c_mean_p:.4f} (95% CI: {c_ci})")
        
        if w_mean_p is not None:
            all_within_means.append(w_mean_p)
        if c_mean_p is not None:
            all_cross_means.append(c_mean_p)
            
        lang_transfer_cka[lang] = {
            "within_family_mean": float(w_mean_p) if w_mean_p else None,
            "within_family_ci_95": w_ci,
            "cross_family_mean": float(c_mean_p) if c_mean_p else None,
            "cross_family_ci_95": c_ci,
            "n_within": n_within_pairs,
            "n_cross": n_cross_pairs,
        }
        
    out = {
        "method": "Linear CKA (dimension-invariant, bootstrapped per-prompt)",
        "n_models": n_models,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "per_language_transfer": lang_transfer_cka,
    }
    
    out_path = f"{base}/experiment13_cka.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
