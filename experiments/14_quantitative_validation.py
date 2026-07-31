"""
Experiment 14: Quantitative Validation of LAS (Bootstrapped)
=============================================================
Trains Language-Anchored Steering (LAS) matrices for all models and languages.
Computes quantitative metrics for ACL Main readiness:
- Pre-LAS vs Post-LAS Cosine Similarity
- Rotation Magnitude ||R - I||_F
- Vectorized Kabsch SVD over 1000 bootstrap iterations for 95% CIs.
"""
import os, sys, json, pickle, glob, argparse
import numpy as np
import torch
from pathlib import Path

LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']
NON_EN = [l for l in LANGUAGES if l != 'en']

def batched_kabsch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched Kabsch algorithm using SVD.
    A: (b, d, n), B: (b, d, n)
    Finds R (b, d, d) that minimizes ||R @ A - B||_F^2
    Returns R.
    """
    # Cross-covariance matrix C: (b, d, d)
    # A is (b, d, n), B is (b, d, n)
    # C = B @ A^T
    C = torch.bmm(B, A.transpose(1, 2))
    
    # SVD: C = U S V^T
    try:
        U, S, Vh = torch.linalg.svd(C)
    except Exception as e:
        # Fallback to pseudo-inverse or slower CPU SVD if it fails
        U, S, Vh = torch.linalg.svd(C.cpu())
        U = U.to(C.device)
        Vh = Vh.to(C.device)
        
    # R = U @ Vh
    R = torch.bmm(U, Vh)
    
    # Check for reflection and correct it
    det = torch.linalg.det(R)
    reflection_mask = det < 0
    if reflection_mask.any():
        # Flip the last column of U for the batches with reflection
        U_corrected = U.clone()
        U_corrected[reflection_mask, :, -1] *= -1
        R = torch.bmm(U_corrected, Vh)
        
    return R

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
        
    print(f"Loaded {len(model_dirs)} models")
    print(f"Running vectorized Kabsch LAS with {args.n_boot} bootstrap iterations...")
    
    device = torch.device("cpu")
    results = {}
    
    for model, mdir in model_dirs.items():
        print(f"\nProcessing {model}...")
        model_results = {}
        
        # Get best layer from phase2
        best_layer = 0
        p2_files = glob.glob(f"{mdir}/phase2_pivot_analysis/phase2_results.json")
        if p2_files:
            try:
                p2 = json.load(open(p2_files[0]))
                best_layer = p2.get("best_layer", 0)
            except: pass
            
        en_diffs_path = os.path.join(mdir, "en", f"per_prompt_diffs_layer_{best_layer:03d}.npy")
        if not os.path.exists(en_diffs_path):
            continue
            
        en_arr = np.load(en_diffs_path) # (n_prompts, dim)
        n_prompts = en_arr.shape[0]
        dim = en_arr.shape[1]
        
        idx_boot = np.random.choice(n_prompts, size=(args.n_boot, n_prompts), replace=True)
        
        en_t = torch.tensor(en_arr, dtype=torch.float32, device=device)
        
        # Point estimate (1, d, n)
        en_point = en_t.unsqueeze(0).transpose(1, 2)
        # Bootstrap (b, d, n)
        en_boot = en_t[idx_boot].transpose(1, 2)
        
        for lang in NON_EN:
            lang_diffs_path = os.path.join(mdir, lang, f"per_prompt_diffs_layer_{best_layer:03d}.npy")
            if not os.path.exists(lang_diffs_path):
                continue
                
            lang_arr = np.load(lang_diffs_path)
            lang_t = torch.tensor(lang_arr, dtype=torch.float32, device=device)
            
            lang_point = lang_t.unsqueeze(0).transpose(1, 2)
            lang_boot = lang_t[idx_boot].transpose(1, 2)
            
            # Point estimate (using exact means for consistency)
            en_mean = en_point.mean(dim=2) # (1, d)
            lang_mean = lang_point.mean(dim=2) # (1, d)
            
            en_vec = en_mean / (torch.norm(en_mean, dim=1, keepdim=True) + 1e-8)
            tgt_vec = lang_mean / (torch.norm(lang_mean, dim=1, keepdim=True) + 1e-8)
            
            pre_sim_val = torch.sum(en_vec * tgt_vec, dim=1)[0].item()
            
            R_point = batched_kabsch(en_point, lang_point) # (1, d, d)
            rotated_en_vec = torch.bmm(R_point, en_vec.unsqueeze(2)).squeeze(2)
            rotated_en_vec = rotated_en_vec / (torch.norm(rotated_en_vec, dim=1, keepdim=True) + 1e-8)
            post_sim_val = torch.sum(rotated_en_vec * tgt_vec, dim=1)[0].item()
            
            I = torch.eye(dim, device=device).unsqueeze(0)
            rot_mag_val = torch.norm(R_point - I, p="fro", dim=(1,2))[0].item()
            
            # Bootstraps
            en_b_mean = en_boot.mean(dim=2) # (b, d)
            lang_b_mean = lang_boot.mean(dim=2) # (b, d)
            
            en_b_vec = en_b_mean / (torch.norm(en_b_mean, dim=1, keepdim=True) + 1e-8)
            tgt_b_vec = lang_b_mean / (torch.norm(lang_b_mean, dim=1, keepdim=True) + 1e-8)
            
            pre_sim_b = torch.sum(en_b_vec * tgt_b_vec, dim=1)
            
            R_boot = batched_kabsch(en_boot, lang_boot) # (b, d, d)
            rotated_en_b_vec = torch.bmm(R_boot, en_b_vec.unsqueeze(2)).squeeze(2)
            rotated_en_b_vec = rotated_en_b_vec / (torch.norm(rotated_en_b_vec, dim=1, keepdim=True) + 1e-8)
            
            post_sim_b = torch.sum(rotated_en_b_vec * tgt_b_vec, dim=1)
            rot_mag_b = torch.norm(R_boot - I, p="fro", dim=(1,2))
            
            improv_b = post_sim_b - pre_sim_b
            
            post_sim_b = post_sim_b.cpu().numpy()
            improv_b = improv_b.cpu().numpy()
            
            res = {
                "pre_las_sim": float(pre_sim_val),
                "post_las_sim": float(post_sim_val),
                "post_las_sim_ci_95": [float(np.percentile(post_sim_b, 2.5)), float(np.percentile(post_sim_b, 97.5))],
                "improvement": float(post_sim_val - pre_sim_val),
                "improvement_ci_95": [float(np.percentile(improv_b, 2.5)), float(np.percentile(improv_b, 97.5))],
                "rotation_magnitude": float(rot_mag_val)
            }
            
            model_results[lang] = res
            print(f"  {lang}: Pre {res['pre_las_sim']:.3f} -> Post {res['post_las_sim']:.3f} "
                  f"(95% CI: [{res['post_las_sim_ci_95'][0]:.3f}, {res['post_las_sim_ci_95'][1]:.3f}]) "
                  f"(RotMag: {res['rotation_magnitude']:.3f})")
                  
        if model_results:
            results[model] = model_results
            
    out_path = f"{base}/experiment14_las_validation.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {"n_boot": args.n_boot, "seed": args.seed},
            "models": results
        }, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
