"""
Experiment 15: Dual Process Safety Theory
==========================================
Tests Kahneman's Dual Process Theory in LLM safety:
- System 1 (automatic): Safety signal emerges at early layers
- System 2 (deliberate): Safety signal emerges at late layers

Measures Safety Emergence Layer (SEL) per language per model.
Hypothesis: English SEL < Hindi SEL < Swahili SEL

Runs on CPU only — uses existing steering vectors.
"""
import os, sys, json, glob, pickle, math
import numpy as np
from collections import defaultdict

np.random.seed(42)

base = "/root/clr_paper/results"
LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']
NON_EN = [l for l in LANGUAGES if l != 'en']

# Load all steering vectors
print("="*60)
print("EXPERIMENT 15: DUAL PROCESS SAFETY THEORY")
print("="*60)

model_dirs = {}
for pkl in sorted(glob.glob(f"{base}/steering_vectors_*/all_steering_vectors.pkl")):
    d = pkl.replace("/all_steering_vectors.pkl", "")
    name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    model_dirs[name] = d

print(f"Found {len(model_dirs)} models")

results = {"models": {}, "aggregate": {}}

for model_name, model_dir in sorted(model_dirs.items()):
    pkl_path = f"{model_dir}/all_steering_vectors.pkl"
    try:
        with open(pkl_path, "rb") as f:
            svs = pickle.load(f)
    except Exception as e:
        print(f"  {model_name}: FAILED to load ({e})")
        continue

    print(f"\n--- {model_name} ---")
    
    # Get English vectors per layer
    en_data = svs.get('en', {})
    if not isinstance(en_data, dict):
        print(f"  Skipping: English data is not layer-keyed")
        continue
    
    # Sort layers numerically
    layer_keys = sorted(en_data.keys(), key=lambda x: int(x) if str(x).isdigit() else 0)
    n_layers = len(layer_keys)
    
    if n_layers < 5:
        print(f"  Skipping: only {n_layers} layers")
        continue
    
    model_results = {"n_layers": n_layers, "languages": {}}
    
    for lang in LANGUAGES:
        lang_data = svs.get(lang, {})
        if not isinstance(lang_data, dict):
            continue
        
        # Compute per-layer cosine similarity with English
        layer_sims = []
        for lk in layer_keys:
            en_vec = en_data.get(lk)
            lang_vec = lang_data.get(lk)
            
            if en_vec is None or lang_vec is None:
                layer_sims.append(0.0)
                continue
            
            en_flat = en_vec.flatten().astype(np.float64)
            lang_flat = lang_vec.flatten().astype(np.float64)
            
            norm_en = np.linalg.norm(en_flat)
            norm_lang = np.linalg.norm(lang_flat)
            
            if norm_en < 1e-10 or norm_lang < 1e-10:
                layer_sims.append(0.0)
                continue
            
            cos_sim = np.dot(en_flat, lang_flat) / (norm_en * norm_lang)
            layer_sims.append(float(cos_sim))
        
        # Safety Emergence Layer (SEL): first layer where sim > threshold
        # We use multiple thresholds for robustness
        sel_results = {}
        for threshold in [0.1, 0.2, 0.3, 0.5]:
            sel = n_layers  # default: never emerges
            for i, sim in enumerate(layer_sims):
                if sim > threshold:
                    sel = i
                    break
            sel_results[f"sel_{threshold}"] = sel
        
        # Peak layer: where similarity is maximum
        peak_layer = int(np.argmax(layer_sims))
        peak_sim = float(max(layer_sims))
        
        # "Automaticity score": fraction of layers where sim > 0.1
        # High automaticity = safety signal present throughout = System 1
        automaticity = sum(1 for s in layer_sims if s > 0.1) / n_layers
        
        # Layer-wise safety profile (normalized layer positions)
        # Early peak = System 1, Late peak = System 2
        if peak_sim > 0.05:
            normalized_peak = peak_layer / n_layers  # 0=early, 1=late
        else:
            normalized_peak = 1.0  # no peak = System 2
        
        lang_result = {
            "sel": sel_results,
            "peak_layer": peak_layer,
            "peak_layer_normalized": float(normalized_peak),
            "peak_sim": peak_sim,
            "automaticity": float(automaticity),
            "layer_profile": [round(s, 4) for s in layer_sims],
        }
        
        model_results["languages"][lang] = lang_result
        
        if lang == 'en':
            print(f"  {lang}: peak=L{peak_layer}/{n_layers} (sim={peak_sim:.3f}), auto={automaticity:.2f}")
        elif lang in ['hi', 'ar', 'sw', 'ja']:
            print(f"  {lang}: peak=L{peak_layer}/{n_layers} (sim={peak_sim:.3f}), auto={automaticity:.2f}, SEL@0.2=L{sel_results['sel_0.2']}")
    
    results["models"][model_name] = model_results

# =============================================
# Aggregate analysis
# =============================================
print("\n" + "="*60)
print("AGGREGATE DUAL PROCESS ANALYSIS")
print("="*60)

# Per-language aggregates
lang_peaks = defaultdict(list)
lang_auto = defaultdict(list)
lang_sel = defaultdict(list)

for model_name, mdata in results["models"].items():
    n_layers = mdata["n_layers"]
    for lang, ldata in mdata["languages"].items():
        lang_peaks[lang].append(ldata["peak_layer_normalized"])
        lang_auto[lang].append(ldata["automaticity"])
        lang_sel[lang].append(ldata["sel"]["sel_0.2"] / n_layers)  # normalize by model depth

print(f"\n{'Language':10s} {'Peak (norm)':>12s} {'Automaticity':>14s} {'SEL@0.2 (norm)':>16s} {'Process':>10s}")
for lang in LANGUAGES:
    if lang in lang_peaks:
        peak = np.mean(lang_peaks[lang])
        auto = np.mean(lang_auto[lang])
        sel = np.mean(lang_sel[lang])
        # Classify as System 1 or System 2
        process = "System 1" if auto > 0.5 else "System 2"
        print(f"{lang:10s} {peak:>12.3f} {auto:>14.3f} {sel:>16.3f} {process:>10s}")
        
        results["aggregate"][lang] = {
            "mean_peak_normalized": float(peak),
            "mean_automaticity": float(auto),
            "mean_sel_normalized": float(sel),
            "process_type": process,
            "n_models": len(lang_peaks[lang])
        }

# Correlation: SCD vs SEL
print("\n--- SCD vs SEL Correlation ---")
scd_vals = []
sel_vals = []
for model_name, model_dir in model_dirs.items():
    p2f = f"{model_dir}/phase2_pivot_analysis/phase2_results.json"
    if not os.path.exists(p2f):
        continue
    p2 = json.load(open(p2f))
    scds = p2.get("scd_scores_degrees", {})
    
    mdata = results["models"].get(model_name, {})
    n_layers = mdata.get("n_layers", 32)
    
    for lang in NON_EN:
        scd = scds.get(lang)
        ldata = mdata.get("languages", {}).get(lang, {})
        sel = ldata.get("sel", {}).get("sel_0.2")
        
        if scd is not None and sel is not None and not math.isnan(scd):
            scd_vals.append(scd)
            sel_vals.append(sel / n_layers)

if len(scd_vals) > 10:
    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(scd_vals, sel_vals)
    rho, rho_p = spearmanr(scd_vals, sel_vals)
    print(f"  SCD vs SEL: Pearson r={r:.3f} (p={p:.4f}), Spearman rho={rho:.3f} (p={rho_p:.4f})")
    print(f"  n={len(scd_vals)} observations")
    results["scd_vs_sel"] = {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "n": len(scd_vals)
    }

# Correlation: Automaticity vs SCD
auto_vals = []
scd_vals2 = []
for model_name, model_dir in model_dirs.items():
    p2f = f"{model_dir}/phase2_pivot_analysis/phase2_results.json"
    if not os.path.exists(p2f):
        continue
    p2 = json.load(open(p2f))
    scds = p2.get("scd_scores_degrees", {})
    
    mdata = results["models"].get(model_name, {})
    for lang in NON_EN:
        scd = scds.get(lang)
        ldata = mdata.get("languages", {}).get(lang, {})
        auto = ldata.get("automaticity")
        if scd is not None and auto is not None and not math.isnan(scd):
            scd_vals2.append(scd)
            auto_vals.append(auto)

if len(auto_vals) > 10:
    r, p = pearsonr(scd_vals2, auto_vals)
    rho, rho_p = spearmanr(scd_vals2, auto_vals)
    print(f"\n  SCD vs Automaticity: Pearson r={r:.3f} (p={p:.4f}), Spearman rho={rho:.3f} (p={rho_p:.4f})")
    results["scd_vs_automaticity"] = {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "n": len(auto_vals)
    }

out_path = f"{base}/experiment15_dual_process.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
