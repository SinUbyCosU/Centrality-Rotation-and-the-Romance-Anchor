"""
Experiment 16: Linguistic Security Index
==========================================
Based on Labov's Linguistic Insecurity (1966):
Speakers of non-prestige dialects show measurable uncertainty.

We measure analogous "insecurity" in LLM safety representations:
1. Steering Vector Variance: How consistent is the safety direction across prompts?
2. Directional Stability: Does the safety vector "wobble" across layers?
3. Norm Confidence: How strong is the safety signal magnitude?

High security = consistent, stable, strong signal (English)
Low security = variable, unstable, weak signal (Swahili)

Runs on CPU only — uses existing steering vectors.
"""
import os, sys, json, glob, pickle, math
import numpy as np
from collections import defaultdict

np.random.seed(42)

base = "/root/clr_paper/results"
LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']
NON_EN = [l for l in LANGUAGES if l != 'en']

print("="*60)
print("EXPERIMENT 16: LINGUISTIC SECURITY INDEX")
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
        print(f"  {model_name}: FAILED ({e})")
        continue

    print(f"\n--- {model_name} ---")
    
    # Check if we have per-prompt vectors (some models store aggregated only)
    # We'll use multi-layer analysis for security metrics
    en_data = svs.get('en', {})
    if not isinstance(en_data, dict):
        print(f"  Skipping: not layer-keyed")
        continue
    
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
        
        # Collect vectors across all layers
        layer_vectors = []
        layer_norms = []
        for lk in layer_keys:
            vec = lang_data.get(lk)
            if vec is not None and hasattr(vec, 'shape'):
                v = vec.flatten().astype(np.float64)
                layer_vectors.append(v)
                layer_norms.append(float(np.linalg.norm(v)))
        
        if len(layer_vectors) < 5:
            continue
        
        # ========================================
        # METRIC 1: Directional Stability
        # How much does the safety direction change across layers?
        # Low stability = model keeps changing its mind = insecure
        # ========================================
        consecutive_sims = []
        for i in range(len(layer_vectors) - 1):
            v1, v2 = layer_vectors[i], layer_vectors[i+1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                sim = np.dot(v1, v2) / (n1 * n2)
                consecutive_sims.append(float(sim))
        
        directional_stability = float(np.mean(consecutive_sims)) if consecutive_sims else 0.0
        
        # ========================================
        # METRIC 2: Norm Confidence
        # How strong is the safety signal? Weak norms = insecure
        # Normalized by English norm for comparability
        # ========================================
        mean_norm = float(np.mean(layer_norms))
        norm_cv = float(np.std(layer_norms) / (mean_norm + 1e-10))  # coefficient of variation
        
        # ========================================
        # METRIC 3: Cross-Layer Coherence
        # Pairwise cosine similarity across ALL layers (not just consecutive)
        # High coherence = consistent safety concept across depths
        # ========================================
        pairwise_sims = []
        # Sample to keep computation reasonable
        sample_indices = np.linspace(0, len(layer_vectors)-1, min(10, len(layer_vectors)), dtype=int)
        for i in range(len(sample_indices)):
            for j in range(i+1, len(sample_indices)):
                v1 = layer_vectors[sample_indices[i]]
                v2 = layer_vectors[sample_indices[j]]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    pairwise_sims.append(float(np.dot(v1, v2) / (n1 * n2)))
        
        cross_layer_coherence = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0
        
        # ========================================
        # METRIC 4: Alignment Confidence (vs English)
        # How consistently does this language align with English across layers?
        # ========================================
        en_layer_data = svs.get('en', {})
        en_alignment_sims = []
        for lk in layer_keys:
            en_vec = en_layer_data.get(lk)
            lang_vec = lang_data.get(lk)
            if en_vec is not None and lang_vec is not None:
                ev = en_vec.flatten().astype(np.float64)
                lv = lang_vec.flatten().astype(np.float64)
                ne, nl = np.linalg.norm(ev), np.linalg.norm(lv)
                if ne > 1e-10 and nl > 1e-10:
                    en_alignment_sims.append(float(np.dot(ev, lv) / (ne * nl)))
        
        alignment_stability = float(np.std(en_alignment_sims)) if en_alignment_sims else 1.0
        alignment_mean = float(np.mean(en_alignment_sims)) if en_alignment_sims else 0.0
        
        # ========================================
        # COMPOSITE: Linguistic Security Index (LSI)
        # Combines all metrics into a single 0-1 score
        # ========================================
        # Normalize each component to [0, 1] range
        stability_score = max(0, min(1, (directional_stability + 1) / 2))  # [-1,1] -> [0,1]
        coherence_score = max(0, min(1, (cross_layer_coherence + 1) / 2))
        norm_score = max(0, min(1, 1 - norm_cv))  # low CV = high confidence
        align_score = max(0, min(1, 1 - alignment_stability))  # low std = high stability
        
        lsi = float(np.mean([stability_score, coherence_score, norm_score, align_score]))
        
        lang_result = {
            "directional_stability": directional_stability,
            "cross_layer_coherence": cross_layer_coherence,
            "mean_norm": mean_norm,
            "norm_cv": norm_cv,
            "alignment_mean": alignment_mean,
            "alignment_stability": alignment_stability,
            "linguistic_security_index": lsi,
        }
        
        model_results["languages"][lang] = lang_result
        
        if lang in ['en', 'es', 'hi', 'ar', 'sw', 'ja']:
            print(f"  {lang}: LSI={lsi:.3f} | stability={directional_stability:.3f} | coherence={cross_layer_coherence:.3f} | align_std={alignment_stability:.3f}")
    
    results["models"][model_name] = model_results

# =============================================
# Aggregate
# =============================================
print("\n" + "="*60)
print("AGGREGATE LINGUISTIC SECURITY INDEX")
print("="*60)

lang_lsi = defaultdict(list)
lang_stability = defaultdict(list)
lang_coherence = defaultdict(list)

for model_name, mdata in results["models"].items():
    for lang, ldata in mdata.get("languages", {}).items():
        lang_lsi[lang].append(ldata["linguistic_security_index"])
        lang_stability[lang].append(ldata["directional_stability"])
        lang_coherence[lang].append(ldata["cross_layer_coherence"])

print(f"\n{'Language':10s} {'LSI':>8s} {'Stability':>12s} {'Coherence':>12s} {'n':>4s} {'Interpretation':>20s}")
for lang in LANGUAGES:
    if lang in lang_lsi:
        lsi = np.mean(lang_lsi[lang])
        stab = np.mean(lang_stability[lang])
        coh = np.mean(lang_coherence[lang])
        n = len(lang_lsi[lang])
        interp = "Secure" if lsi > 0.65 else "Moderate" if lsi > 0.55 else "Insecure"
        print(f"{lang:10s} {lsi:>8.3f} {stab:>12.3f} {coh:>12.3f} {n:>4d} {interp:>20s}")
        
        results["aggregate"][lang] = {
            "mean_lsi": float(lsi),
            "std_lsi": float(np.std(lang_lsi[lang])),
            "mean_stability": float(stab),
            "mean_coherence": float(coh),
            "n_models": n,
            "interpretation": interp
        }

# Correlation: LSI vs SCD
print("\n--- LSI vs SCD Correlation ---")
lsi_vals = []
scd_vals = []
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
        lsi = ldata.get("linguistic_security_index")
        if scd is not None and lsi is not None and not math.isnan(scd):
            scd_vals.append(scd)
            lsi_vals.append(lsi)

if len(lsi_vals) > 10:
    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(scd_vals, lsi_vals)
    rho, rho_p = spearmanr(scd_vals, lsi_vals)
    print(f"  LSI vs SCD: Pearson r={r:.3f} (p={p:.6f}), Spearman rho={rho:.3f} (p={rho_p:.6f})")
    print(f"  n={len(lsi_vals)}")
    results["lsi_vs_scd"] = {
        "pearson_r": float(r), "pearson_p": float(p),
        "spearman_rho": float(rho), "spearman_p": float(rho_p),
        "n": len(lsi_vals)
    }

out_path = f"{base}/experiment16_linguistic_security.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
