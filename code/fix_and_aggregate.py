"""
fix_and_aggregate.py
Patches all known issues and produces a unified results dataset.
Reads directly from the remote server's JSON files.
"""
import paramiko, json, sys, math, os
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"  
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

def ssh_exec(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=120)
    return stdout.read().decode("utf-8", errors="replace").strip()

# Upload and run the aggregation on the server (avoids quoting issues)
agg_script = r'''
import json, os, glob, math
from collections import defaultdict
import numpy as np

base = "/root/clr_paper/results"
LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']
NON_EN = [l for l in LANGUAGES if l != 'en']

# Find all model dirs
model_dirs = {}
for pkl in sorted(glob.glob(f"{base}/steering_vectors_*/all_steering_vectors.pkl")):
    d = pkl.replace("/all_steering_vectors.pkl", "")
    name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    model_dirs[name] = d

print(f"Processing {len(model_dirs)} models...")

unified = {
    "models": {},
    "aggregate": {},
    "phase9": {},
    "phase10_summary": {},
    "phase8_summary": {}
}

# Per-language SCD accumulators
scd_by_lang = defaultdict(list)
# Per-language ASR accumulators
asr_by_lang = defaultdict(list)
# Per-language transfer accumulators  
transfer_by_lang = defaultdict(list)
# Phase 8 lift accumulators
lift_by_lang = defaultdict(list)

for model, d in sorted(model_dirs.items()):
    mdata = {"model": model}
    
    # =============================================
    # PHASE 2: Pivot + SCD (FIX: read pivot_centrality)
    # =============================================
    p2f = f"{d}/phase2_pivot_analysis/phase2_results.json"
    if os.path.exists(p2f):
        p2 = json.load(open(p2f))
        # FIX Bug 1: use pivot_centrality
        mdata["pivot_language"] = p2.get("pivot_centrality", "unknown")
        mdata["pivot_projection"] = p2.get("pivot_projection", "unknown")
        mdata["best_layer"] = p2.get("best_layer")
        
        scd = p2.get("scd_scores_degrees", {})
        mdata["scd_degrees"] = scd
        for lang, val in scd.items():
            if lang != "en" and not math.isnan(val):
                scd_by_lang[lang].append(val)
        
        # Centrality scores
        mdata["centrality_scores"] = p2.get("centrality_scores", {})
        
        # Statistical tests
        mdata["statistical_tests"] = p2.get("statistical_tests", {})
    
    # =============================================
    # PHASE 4: Psycholinguistic (FIX: read nested structure)
    # =============================================
    p4f = f"{d}/phase4_psycholinguistic/phase4_results.json"
    if os.path.exists(p4f):
        p4 = json.load(open(p4f))
        # FIX Bug 2: extract from regression_models
        reg = p4.get("regression_models", {})
        mdata["psycholinguistic_improvement"] = p4.get("psycholinguistic_improvement")
        
        # Try to extract individual regression metrics
        if isinstance(reg, dict):
            for reg_name, reg_data in reg.items():
                if isinstance(reg_data, dict):
                    for k, v in reg_data.items():
                        mdata[f"phase4_{reg_name}_{k}"] = v
    
    # =============================================
    # PHASE 5: LAS
    # =============================================
    p5f = f"{d}/phase5_las/phase5_results.json"
    if os.path.exists(p5f):
        p5 = json.load(open(p5f))
        mdata["phase5_las"] = p5
    
    # =============================================
    # PHASE 7: Code-mix
    # =============================================
    p7f = f"{d}/phase7_codemix/phase7_results.json"
    if os.path.exists(p7f):
        p7 = json.load(open(p7f))
        mdata["phase7_instability_note"] = p7.get("instability_note", "")
        mdata["phase7_eacl_connection"] = p7.get("eacl_connection", "")
    
    # =============================================
    # PHASE 8: Causal Intervention (FIX: compute best_steered)
    # =============================================
    p8f = f"{d}/phase8_causal_intervention/phase8_results.json"
    if os.path.exists(p8f):
        p8 = json.load(open(p8f))
        p8_langs = p8.get("languages", {})
        mdata["phase8"] = {}
        
        for lang, ldata in p8_langs.items():
            alpha_results = ldata.get("alpha_results", {})
            
            # FIX Bug 4: compute best_steered_refusal_rate
            rates = []
            for akey, adata in alpha_results.items():
                r = adata.get("refusal_rate")
                if r is not None and not math.isnan(r):
                    rates.append(r)
            
            baseline = ldata.get("baseline_refusal_rate", 0)
            if baseline is None: baseline = 0
            best_steered = max(rates) if rates else baseline
            safety_lift = ldata.get("safety_lift", best_steered - baseline)
            scd_val = ldata.get("scd_degrees", None)
            
            mdata["phase8"][lang] = {
                "baseline_refusal_rate": baseline,
                "best_steered_refusal_rate": best_steered,
                "safety_lift": safety_lift,
                "scd_degrees": scd_val,
                "alpha_results": {k: v.get("refusal_rate") for k, v in alpha_results.items()}
            }
            
            if lang != "en" and safety_lift is not None and not math.isnan(safety_lift):
                lift_by_lang[lang].append(safety_lift)
        
        mdata["phase8_correlation"] = p8.get("scd_vs_lift_correlation", {})
    
    # =============================================
    # PHASE 10: Adversarial Probing
    # =============================================
    p10f = f"{d}/phase10_adversarial_probing/phase10_results.json"
    if os.path.exists(p10f):
        p10 = json.load(open(p10f))
        p10_langs = p10.get("languages", {})
        mdata["phase10"] = {}
        
        for lang, ldata in p10_langs.items():
            overall_asr = ldata.get("overall_asr", 0)
            attacks = {}
            for atk, adata in ldata.get("attacks", {}).items():
                attacks[atk] = adata.get("asr", 0)
            mdata["phase10"][lang] = {
                "overall_asr": overall_asr,
                "attacks": attacks
            }
            if overall_asr is not None and not math.isnan(overall_asr):
                asr_by_lang[lang].append(overall_asr)
        
        mdata["phase10_correlation"] = p10.get("scd_vs_asr", {})
    
    unified["models"][model] = mdata

# =============================================
# PHASE 9: Cross-Model Transfer (FIX: recompute p-value)
# =============================================
p9f = f"{base}/phase9_cross_model_transfer/phase9_results.json"
if os.path.exists(p9f):
    p9 = json.load(open(p9f))
    within = p9.get("within_family_sim", {})
    cross = p9.get("cross_family_sim", {})
    
    # FIX Bug 7: recompute Mann-Whitney
    try:
        from scipy.stats import mannwhitneyu
        # We need the raw values - reconstruct from transfer_matrix_sample
        tm = p9.get("transfer_matrix_sample", {})
        within_vals = []
        cross_vals = []
        for pair_key, pair_data in tm.items():
            sim = pair_data.get("cosine_similarity")
            if sim is not None and not math.isnan(sim):
                if pair_data.get("same_family", False):
                    within_vals.append(sim)
                else:
                    cross_vals.append(sim)
        
        if within_vals and cross_vals:
            u_stat, p_val = mannwhitneyu(within_vals, cross_vals, alternative='greater')
            p9["mann_whitney_p_fixed"] = p_val
            p9["mann_whitney_u"] = float(u_stat)
        else:
            p9["mann_whitney_p_fixed"] = None
    except ImportError:
        p9["mann_whitney_p_fixed"] = "scipy not available"
    
    per_lang = p9.get("per_language_transfer", {})
    for lang, stats in per_lang.items():
        mean_val = stats.get("mean_transfer_sim")
        if mean_val is not None and not math.isnan(mean_val):
            transfer_by_lang[lang].append(mean_val)
    
    unified["phase9"] = {
        "n_models": p9.get("n_models"),
        "within_family_mean": within.get("mean"),
        "within_family_std": within.get("std"),
        "within_family_n": within.get("n"),
        "cross_family_mean": cross.get("mean"),
        "cross_family_std": cross.get("std"),
        "cross_family_n": cross.get("n"),
        "mann_whitney_p": p9.get("mann_whitney_p_fixed"),
        "per_language": per_lang
    }

# =============================================
# AGGREGATE STATISTICS
# =============================================

# SCD by language
scd_agg = {}
for lang in NON_EN:
    vals = scd_by_lang.get(lang, [])
    if vals:
        scd_agg[lang] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": len(vals)
        }
unified["aggregate"]["scd_by_language"] = scd_agg

# ASR by language (Phase 10)
asr_agg = {}
for lang in LANGUAGES:
    vals = asr_by_lang.get(lang, [])
    if vals:
        asr_agg[lang] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals)
        }
unified["aggregate"]["asr_by_language"] = asr_agg

# Safety lift by language (Phase 8)
lift_agg = {}
for lang in NON_EN:
    vals = lift_by_lang.get(lang, [])
    if vals:
        lift_agg[lang] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals)
        }
unified["aggregate"]["lift_by_language"] = lift_agg

# Pivot language distribution
pivot_counts = defaultdict(list)
for model, mdata in unified["models"].items():
    pivot = mdata.get("pivot_language", "unknown")
    pivot_counts[pivot].append(model)
unified["aggregate"]["pivot_distribution"] = {k: {"count": len(v), "models": v} for k, v in pivot_counts.items()}

# Save
out_path = f"{base}/unified_results.json"
with open(out_path, "w") as f:
    json.dump(unified, f, indent=2, default=str)
print(f"Saved unified results to {out_path}")
print(f"Models: {len(unified['models'])}")
print()

# Print summary tables for the README
print("=== PIVOT LANGUAGE DISTRIBUTION ===")
for pivot, data in sorted(unified["aggregate"]["pivot_distribution"].items(), key=lambda x: -x[1]["count"]):
    print(f"  {pivot}: {data['count']} models — {', '.join(data['models'])}")

print("\n=== SCD BY LANGUAGE (sorted by mean) ===")
for lang, stats in sorted(scd_agg.items(), key=lambda x: -x[1]["mean"]):
    print(f"  {lang}: mean={stats['mean']:.2f}° ± {stats['std']:.2f}° (median={stats['median']:.2f}°, range={stats['min']:.1f}-{stats['max']:.1f}°, n={stats['n']})")

print("\n=== ASR BY LANGUAGE (Phase 10, sorted by mean) ===")
for lang, stats in sorted(asr_agg.items(), key=lambda x: -x[1]["mean"]):
    print(f"  {lang}: mean ASR={stats['mean']*100:.1f}% ± {stats['std']*100:.1f}% (n={stats['n']})")

print("\n=== SAFETY LIFT BY LANGUAGE (Phase 8, sorted by mean) ===")
for lang, stats in sorted(lift_agg.items(), key=lambda x: -x[1]["mean"]):
    print(f"  {lang}: mean lift={stats['mean']*100:+.1f}% ± {stats['std']*100:.1f}% (n={stats['n']})")

print("\n=== PHASE 9: CROSS-MODEL TRANSFER ===")
p9s = unified["phase9"]
print(f"  Within-family similarity: {p9s.get('within_family_mean',0):.4f} ± {p9s.get('within_family_std',0):.4f} (n={p9s.get('within_family_n')})")
print(f"  Cross-family similarity:  {p9s.get('cross_family_mean',0):.4f} ± {p9s.get('cross_family_std',0):.4f} (n={p9s.get('cross_family_n')})")
print(f"  Mann-Whitney p-value:     {p9s.get('mann_whitney_p')}")

print("\n=== PER-MODEL PHASE 8 DETAIL ===")
for model, mdata in sorted(unified["models"].items()):
    p8 = mdata.get("phase8", {})
    if not p8: continue
    en_base = p8.get("en", {}).get("baseline_refusal_rate", 0)
    hi_base = p8.get("hi", {}).get("baseline_refusal_rate", 0)
    hi_best = p8.get("hi", {}).get("best_steered_refusal_rate", 0)
    hi_lift = p8.get("hi", {}).get("safety_lift", 0)
    ar_base = p8.get("ar", {}).get("baseline_refusal_rate", 0)
    ar_best = p8.get("ar", {}).get("best_steered_refusal_rate", 0)
    ar_lift = p8.get("ar", {}).get("safety_lift", 0)
    print(f"  {model:40s} | en_base={en_base*100:.0f}% | hi: {hi_base*100:.0f}%→{hi_best*100:.0f}% (lift={hi_lift*100:+.0f}%) | ar: {ar_base*100:.0f}%→{ar_best*100:.0f}% (lift={ar_lift*100:+.0f}%)")

print("\n=== PER-MODEL PHASE 10 DETAIL ===")
for model, mdata in sorted(unified["models"].items()):
    p10 = mdata.get("phase10", {})
    if not p10: continue
    en_asr = p10.get("en", {}).get("overall_asr", 0)
    hi_asr = p10.get("hi", {}).get("overall_asr", 0)
    ar_asr = p10.get("ar", {}).get("overall_asr", 0)
    sw_asr = p10.get("sw", {}).get("overall_asr", 0)
    corr = mdata.get("phase10_correlation", {})
    r = corr.get("pearson_r", float('nan'))
    p = corr.get("pearson_p", float('nan'))
    r_str = f"r={r:.3f}" if r is not None and not math.isnan(r) else "r=NaN"
    print(f"  {model:40s} | en={en_asr*100:.0f}% | hi={hi_asr*100:.0f}% | ar={ar_asr*100:.0f}% | sw={sw_asr*100:.0f}% | SCD-ASR {r_str}")
'''

# Upload and run
sftp = client.open_sftp()
with sftp.open("/root/clr_paper/fix_and_aggregate.py", "w") as f:
    f.write(agg_script)
sftp.close()

print("Running aggregation on remote server...")
out = ssh_exec("cd /root/clr_paper && python3 fix_and_aggregate.py 2>&1")
print(out)

client.close()
