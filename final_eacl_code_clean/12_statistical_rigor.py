"""
Experiment 12: Statistical Rigor Pass
======================================
Runs on the remote server (CPU-only, no GPU needed).
Re-analyzes existing steering vectors with proper statistical methods:

1. Mixed-effects decomposition of SCD variance (language vs model)
2. Bootstrap CIs for Phase 8 safety lifts  
3. Mantel test for Phase 4 (SCD matrix vs typological distance matrix)
4. Permutation test for pivot language significance
"""
import json, os, glob, pickle, math
import numpy as np
from collections import defaultdict
from pathlib import Path
from itertools import combinations

np.random.seed(42)

base = "/root/clr_paper/results"
LANGUAGES = ['en','es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']
NON_EN = [l for l in LANGUAGES if l != 'en']

# Typological distance matrix (WALS-inspired, normalized 0-1)
# Based on phylogenetic + typological features
TYPOLOGICAL_DISTANCES = {
    ('en','es'): 0.3, ('en','fr'): 0.3, ('en','pt'): 0.35, ('en','de'): 0.2,
    ('en','ru'): 0.5, ('en','ar'): 0.8, ('en','hi'): 0.6, ('en','ja'): 0.9,
    ('en','zh-CN'): 0.85, ('en','sw'): 0.75,
    ('es','fr'): 0.1, ('es','pt'): 0.05, ('es','de'): 0.4, ('es','ru'): 0.5,
    ('es','ar'): 0.8, ('es','hi'): 0.6, ('es','ja'): 0.9, ('es','zh-CN'): 0.85,
    ('es','sw'): 0.7,
    ('fr','pt'): 0.1, ('fr','de'): 0.4, ('fr','ru'): 0.5, ('fr','ar'): 0.75,
    ('fr','hi'): 0.6, ('fr','ja'): 0.9, ('fr','zh-CN'): 0.85, ('fr','sw'): 0.7,
    ('pt','de'): 0.45, ('pt','ru'): 0.5, ('pt','ar'): 0.8, ('pt','hi'): 0.65,
    ('pt','ja'): 0.9, ('pt','zh-CN'): 0.85, ('pt','sw'): 0.7,
    ('de','ru'): 0.4, ('de','ar'): 0.75, ('de','hi'): 0.5, ('de','ja'): 0.85,
    ('de','zh-CN'): 0.8, ('de','sw'): 0.7,
    ('ru','ar'): 0.7, ('ru','hi'): 0.5, ('ru','ja'): 0.8, ('ru','zh-CN'): 0.75,
    ('ru','sw'): 0.65,
    ('ar','hi'): 0.6, ('ar','ja'): 0.85, ('ar','zh-CN'): 0.8, ('ar','sw'): 0.5,
    ('hi','ja'): 0.8, ('hi','zh-CN'): 0.75, ('hi','sw'): 0.6,
    ('ja','zh-CN'): 0.4, ('ja','sw'): 0.9,
    ('zh-CN','sw'): 0.85,
}

def get_typo_dist(l1, l2):
    if l1 == l2: return 0.0
    key = (min(l1,l2), max(l1,l2)) if (min(l1,l2), max(l1,l2)) in TYPOLOGICAL_DISTANCES else (l1, l2)
    return TYPOLOGICAL_DISTANCES.get(key, TYPOLOGICAL_DISTANCES.get((l2,l1), 0.5))

# =============================================
# Load all data
# =============================================
model_dirs = {}
for pkl in sorted(glob.glob(f"{base}/steering_vectors_*/all_steering_vectors.pkl")):
    d = pkl.replace("/all_steering_vectors.pkl", "")
    name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    model_dirs[name] = d

print(f"Loaded {len(model_dirs)} models")

# Collect SCD matrices
scd_matrix = {}  # model -> {lang: scd_degrees}
for model, d in model_dirs.items():
    p2f = f"{d}/phase2_pivot_analysis/phase2_results.json"
    if os.path.exists(p2f):
        p2 = json.load(open(p2f))
        scd_matrix[model] = p2.get("scd_scores_degrees", {})

# Collect Phase 8 data
phase8_data = {}
for model, d in model_dirs.items():
    p8f = f"{d}/phase8_causal_intervention/phase8_results.json"
    if os.path.exists(p8f):
        phase8_data[model] = json.load(open(p8f))

results = {}

# =============================================
# 1. VARIANCE DECOMPOSITION (Language vs Model effects on SCD)
# =============================================
print("\n" + "="*60)
print("1. SCD VARIANCE DECOMPOSITION")
print("="*60)

# Build data table: (model, language) -> SCD
all_scd_values = []
model_means = {}
lang_means = defaultdict(list)
for model, scds in scd_matrix.items():
    model_vals = []
    for lang in NON_EN:
        val = scds.get(lang)
        if val is not None and not math.isnan(val):
            all_scd_values.append({"model": model, "lang": lang, "scd": val})
            model_vals.append(val)
            lang_means[lang].append(val)
    if model_vals:
        model_means[model] = np.mean(model_vals)

grand_mean = np.mean([x["scd"] for x in all_scd_values])
lang_mean_dict = {l: np.mean(v) for l, v in lang_means.items()}

# Compute SS decomposition
SS_total = sum((x["scd"] - grand_mean)**2 for x in all_scd_values)
SS_model = len(NON_EN) * sum((model_means[m] - grand_mean)**2 for m in model_means)
SS_lang = len(model_means) * sum((lang_mean_dict.get(l, grand_mean) - grand_mean)**2 for l in NON_EN if l in lang_mean_dict)
SS_residual = SS_total - SS_model - SS_lang

eta2_model = SS_model / SS_total if SS_total > 0 else 0
eta2_lang = SS_lang / SS_total if SS_total > 0 else 0
eta2_residual = SS_residual / SS_total if SS_total > 0 else 0

print(f"Grand mean SCD: {grand_mean:.2f}°")
print(f"SS_total:    {SS_total:.1f}")
print(f"SS_model:    {SS_model:.1f} (η² = {eta2_model:.3f}, {eta2_model*100:.1f}% of variance)")
print(f"SS_language: {SS_lang:.1f} (η² = {eta2_lang:.3f}, {eta2_lang*100:.1f}% of variance)")
print(f"SS_residual: {SS_residual:.1f} (η² = {eta2_residual:.3f}, {eta2_residual*100:.1f}% of variance)")

# F-tests
n_models = len(model_means)
n_langs = len([l for l in NON_EN if l in lang_mean_dict])
n_total = len(all_scd_values)
df_model = n_models - 1
df_lang = n_langs - 1
df_resid = n_total - n_models - n_langs + 1

MS_model = SS_model / df_model if df_model > 0 else 0
MS_lang = SS_lang / df_lang if df_lang > 0 else 0
MS_resid = SS_residual / df_resid if df_resid > 0 else 1

F_model = MS_model / MS_resid if MS_resid > 0 else 0
F_lang = MS_lang / MS_resid if MS_resid > 0 else 0

print(f"\nF(model):    {F_model:.2f} (df={df_model}, {df_resid})")
print(f"F(language): {F_lang:.2f} (df={df_lang}, {df_resid})")

# Permutation test for language effect
n_perms = 10000
F_lang_null = []
scd_list = [x["scd"] for x in all_scd_values]
for _ in range(n_perms):
    perm_langs = np.random.permutation([x["lang"] for x in all_scd_values])
    perm_lang_means = defaultdict(list)
    for i, val in enumerate(scd_list):
        perm_lang_means[perm_langs[i]].append(val)
    perm_SS_lang = len(model_means) * sum(
        (np.mean(perm_lang_means.get(l, [grand_mean])) - grand_mean)**2
        for l in NON_EN if l in perm_lang_means
    )
    F_lang_null.append(perm_SS_lang / df_lang / MS_resid if MS_resid > 0 else 0)

p_lang_perm = np.mean([f >= F_lang for f in F_lang_null])
print(f"Permutation p(language effect): {p_lang_perm:.4f} (n_perms={n_perms})")

results["variance_decomposition"] = {
    "grand_mean_scd": grand_mean,
    "eta2_model": eta2_model,
    "eta2_language": eta2_lang,
    "eta2_residual": eta2_residual,
    "F_model": F_model,
    "F_language": F_lang,
    "p_language_permutation": p_lang_perm,
    "n_observations": n_total,
    "n_models": n_models,
    "n_languages": n_langs
}

# =============================================
# 2. BOOTSTRAP CIs FOR PHASE 8 SAFETY LIFTS
# =============================================
print("\n" + "="*60)
print("2. BOOTSTRAP CIs FOR PHASE 8 SAFETY LIFTS")
print("="*60)

n_boot = 10000
lift_results = {}

for lang in NON_EN:
    lifts = []
    for model, p8 in phase8_data.items():
        langs_data = p8.get("languages", {})
        if lang in langs_data:
            lift = langs_data[lang].get("safety_lift")
            if lift is not None and not math.isnan(lift):
                lifts.append(lift)
    
    if len(lifts) >= 5:
        observed_mean = np.mean(lifts)
        boot_means = []
        for _ in range(n_boot):
            sample = np.random.choice(lifts, size=len(lifts), replace=True)
            boot_means.append(np.mean(sample))
        
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        # p-value: proportion of bootstrap samples with opposite sign
        if observed_mean > 0:
            p_val = np.mean([m <= 0 for m in boot_means])
        else:
            p_val = np.mean([m >= 0 for m in boot_means])
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {lang:8s}: lift={observed_mean*100:+.1f}% [{ci_low*100:+.1f}%, {ci_high*100:+.1f}%] p={p_val:.4f} {sig} (n={len(lifts)})")
        
        lift_results[lang] = {
            "mean_lift": observed_mean,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "bootstrap_p": p_val,
            "significant": p_val < 0.05,
            "n": len(lifts)
        }

results["phase8_bootstrap"] = lift_results

# FDR correction (Benjamini-Hochberg)
p_vals_sorted = sorted([(lang, d["bootstrap_p"]) for lang, d in lift_results.items()], key=lambda x: x[1])
n_tests = len(p_vals_sorted)
fdr_results = {}
for rank, (lang, p) in enumerate(p_vals_sorted, 1):
    fdr_threshold = 0.05 * rank / n_tests
    fdr_sig = p <= fdr_threshold
    fdr_results[lang] = {"p": p, "fdr_threshold": fdr_threshold, "fdr_significant": fdr_sig}
    print(f"    FDR rank {rank}: {lang} p={p:.4f} threshold={fdr_threshold:.4f} {'SIG' if fdr_sig else 'ns'}")

results["phase8_fdr"] = fdr_results

# =============================================
# 3. MANTEL TEST (SCD matrix vs Typological distance matrix)
# =============================================
print("\n" + "="*60)
print("3. MANTEL TEST: SCD vs Typological Distance")
print("="*60)

mantel_results = {}
for model, scds in scd_matrix.items():
    # Build distance matrices
    langs = [l for l in LANGUAGES if l in scds]
    n = len(langs)
    if n < 5:
        continue
    
    scd_dist = np.zeros((n, n))
    typo_dist = np.zeros((n, n))
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i != j:
                s1 = scds.get(l1, 0)
                s2 = scds.get(l2, 0)
                scd_dist[i, j] = abs(s1 - s2)
                typo_dist[i, j] = get_typo_dist(l1, l2)
    
    # Flatten upper triangle
    idx = np.triu_indices(n, k=1)
    scd_flat = scd_dist[idx]
    typo_flat = typo_dist[idx]
    
    # Pearson correlation
    if np.std(scd_flat) > 0 and np.std(typo_flat) > 0:
        observed_r = np.corrcoef(scd_flat, typo_flat)[0, 1]
    else:
        observed_r = 0.0
    
    # Permutation test
    n_perms_mantel = 9999
    count_ge = 0
    for _ in range(n_perms_mantel):
        perm = np.random.permutation(n)
        perm_scd = scd_dist[np.ix_(perm, perm)]
        perm_flat = perm_scd[idx]
        if np.std(perm_flat) > 0:
            perm_r = np.corrcoef(perm_flat, typo_flat)[0, 1]
        else:
            perm_r = 0
        if perm_r >= observed_r:
            count_ge += 1
    
    p_mantel = (count_ge + 1) / (n_perms_mantel + 1)
    sig = "***" if p_mantel < 0.001 else "**" if p_mantel < 0.01 else "*" if p_mantel < 0.05 else "ns"
    print(f"  {model:40s}: r={observed_r:.4f}, p={p_mantel:.4f} {sig}")
    
    mantel_results[model] = {
        "mantel_r": observed_r,
        "mantel_p": p_mantel,
        "significant": p_mantel < 0.05,
        "n_languages": n
    }

results["mantel_tests"] = mantel_results

# Aggregate Mantel
sig_count = sum(1 for v in mantel_results.values() if v["significant"])
print(f"\n  Significant: {sig_count}/{len(mantel_results)} models (p<0.05)")
mean_r = np.mean([v["mantel_r"] for v in mantel_results.values()])
print(f"  Mean Mantel r: {mean_r:.4f}")

# =============================================
# 4. PERMUTATION TEST FOR PIVOT LANGUAGE
# =============================================
print("\n" + "="*60)
print("4. PERMUTATION TEST: Pivot Language Western Bias")
print("="*60)

western_langs = {'en', 'es', 'fr', 'pt', 'de'}
pivots = []
for model, scds in scd_matrix.items():
    p2f = f"{model_dirs[model]}/phase2_pivot_analysis/phase2_results.json"
    if os.path.exists(p2f):
        p2 = json.load(open(p2f))
        pivot = p2.get("pivot_centrality", "unknown")
        pivots.append(pivot)

observed_western = sum(1 for p in pivots if p in western_langs)
print(f"Observed: {observed_western}/{len(pivots)} Western pivots")

# Under null: pivot is randomly assigned from all 11 languages
n_perms_pivot = 100000
count_ge = 0
for _ in range(n_perms_pivot):
    random_pivots = np.random.choice(LANGUAGES, size=len(pivots))
    random_western = sum(1 for p in random_pivots if p in western_langs)
    if random_western >= observed_western:
        count_ge += 1

p_pivot = count_ge / n_perms_pivot
print(f"Permutation p-value (all Western): {p_pivot:.6f}")
print(f"Expected under null: {len(pivots) * len(western_langs) / len(LANGUAGES):.1f} Western pivots")

results["pivot_permutation"] = {
    "observed_western": observed_western,
    "total_models": len(pivots),
    "expected_under_null": len(pivots) * len(western_langs) / len(LANGUAGES),
    "p_value": p_pivot,
    "n_permutations": n_perms_pivot
}

# =============================================
# 5. PER-MODEL SCD TABLE (for transparency)
# =============================================
print("\n" + "="*60)
print("5. PER-MODEL SCD TABLE")
print("="*60)

header = f"{'Model':40s} | " + " | ".join(f"{l:6s}" for l in NON_EN)
print(header)
print("-" * len(header))
for model in sorted(scd_matrix.keys()):
    scds = scd_matrix[model]
    vals = []
    for l in NON_EN:
        v = scds.get(l)
        if v is not None and not math.isnan(v):
            vals.append(f"{v:6.1f}")
        else:
            vals.append(f"{'NaN':>6s}")
    print(f"{model:40s} | " + " | ".join(vals))

# Save results
out_path = f"{base}/experiment12_stats.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {out_path}")
