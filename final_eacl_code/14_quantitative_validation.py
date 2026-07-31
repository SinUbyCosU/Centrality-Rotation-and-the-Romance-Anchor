"""
Experiment 14: Quantitative Validation of LAS
=================================================
Trains Language-Anchored Steering (LAS) matrices for all models and languages.
Computes quantitative metrics for ACL Main readiness:
- Pre-LAS vs Post-LAS Cosine Similarity (Alignment Improvement)
- Rotation Magnitude ||R - I||_F
"""

import os, sys, json, pickle, glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']
NON_EN = [l for l in LANGUAGES if l != 'en']

class LanguageAnchoredSteering(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rotation = nn.Parameter(torch.eye(hidden_dim))

    def forward(self, english_sv: torch.Tensor) -> torch.Tensor:
        R = self.rotation
        Q, _ = torch.linalg.qr(R)
        return Q @ english_sv

def train_las_for_lang(english_sv: np.ndarray, target_sv: np.ndarray, n_epochs: int = 50, lr: float = 0.01) -> dict:
    hidden_dim = english_sv.shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    en_sv_t = torch.tensor(english_sv, dtype=torch.float32, device=device)
    target_sv_t = torch.tensor(target_sv, dtype=torch.float32, device=device)
    
    model = LanguageAnchoredSteering(hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    pre_cos_sim = torch.dot(en_sv_t, target_sv_t) / (torch.norm(en_sv_t) * torch.norm(target_sv_t) + 1e-8)
    pre_sim_val = pre_cos_sim.item()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        rotated = model(en_sv_t)
        cos_sim = torch.dot(rotated, target_sv_t) / (torch.norm(rotated) * torch.norm(target_sv_t) + 1e-8)
        loss = -cos_sim
        loss.backward()
        optimizer.step()
        
    rotated = model(en_sv_t)
    post_cos_sim = torch.dot(rotated, target_sv_t) / (torch.norm(rotated) * torch.norm(target_sv_t) + 1e-8)
    post_sim_val = post_cos_sim.item()
    
    # Rotation magnitude
    R = model.rotation.detach()
    I = torch.eye(hidden_dim, device=R.device)
    rot_mag = torch.norm(R - I, p="fro").item()
    
    return {
        "pre_las_sim": float(pre_sim_val),
        "post_las_sim": float(post_sim_val),
        "improvement": float(post_sim_val - pre_sim_val),
        "rotation_magnitude": float(rot_mag)
    }

def main():
    base = "/root/clr_paper/results"
    
    model_vectors = {}
    for pkl_path in sorted(glob.glob(f"{base}/steering_vectors_*/all_steering_vectors.pkl")):
        d = pkl_path.replace("/all_steering_vectors.pkl", "")
        name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
        try:
            with open(pkl_path, "rb") as f:
                svs = pickle.load(f)
            model_vectors[name] = svs
        except Exception:
            pass

    print(f"Loaded {len(model_vectors)} models")
    
    results = {}
    for model, svs in model_vectors.items():
        print(f"\nProcessing {model}...")
        model_results = {}
        
        # Get best layer from phase2
        best_layer = None
        p2f = f"{base}/steering_vectors_{model}_*/phase2_pivot_analysis/phase2_results.json"
        p2_files = glob.glob(p2f)
        if p2_files:
            try:
                p2 = json.load(open(p2_files[0]))
                best_layer = p2.get("best_layer")
            except: pass
            
        if 'en' not in svs:
            continue
            
        # extract english vector
        en_data = svs['en']
        en_vec = None
        if isinstance(en_data, dict):
            if best_layer and best_layer in en_data:
                en_vec = en_data[best_layer]
            else:
                en_vec = list(en_data.values())[-1] # fallback to last layer
        else:
            en_vec = en_data
            
        if en_vec is None or not hasattr(en_vec, 'shape'):
            continue
            
        en_vec = en_vec.flatten()
        
        for lang in NON_EN:
            if lang not in svs: continue
            lang_data = svs[lang]
            tgt_vec = None
            if isinstance(lang_data, dict):
                if best_layer and best_layer in lang_data:
                    tgt_vec = lang_data[best_layer]
                else:
                    tgt_vec = list(lang_data.values())[-1]
            else:
                tgt_vec = lang_data
                
            if tgt_vec is None or not hasattr(tgt_vec, 'shape'):
                continue
                
            tgt_vec = tgt_vec.flatten()
            
            try:
                res = train_las_for_lang(en_vec, tgt_vec, n_epochs=100)
                model_results[lang] = res
                print(f"  {lang}: Pre {res['pre_las_sim']:.3f} -> Post {res['post_las_sim']:.3f} (RotMag: {res['rotation_magnitude']:.3f})")
            except Exception as e:
                print(f"  {lang}: failed ({e})")
                
        results[model] = model_results
        
    out_path = f"{base}/experiment14_las_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
