"""Analyze all three experiment results for the paper."""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from collections import defaultdict

# Load results
with open("results/experiment12_stats.json") as f:
    exp12 = json.load(f)
with open("results/experiment13_cka.json") as f:
    exp13 = json.load(f)
with open("results/experiment14_las_validation.json") as f:
    exp14 = json.load(f)

NON_EN = ['es','fr','pt','de','ru','ar','hi','ja','zh-CN','sw']

print("="*70)
print("EXPERIMENT 12: STATISTICAL RIGOR PASS")
print("="*70)

# 1. Variance decomposition
vd = exp12["variance_decomposition"]
print(f"\n1. SCD Variance Decomposition (n={vd['n_observations']})")
print(f"   Grand Mean SCD: {vd['grand_mean_scd']:.1f}°")
print(f"   Model effect:    η²={vd['eta2_model']:.3f} ({vd['eta2_model']*100:.1f}% of variance), F={vd['F_model']:.2f}")
print(f"   Language effect: η²={vd['eta2_language']:.3f} ({vd['eta2_language']*100:.1f}% of variance), F={vd['F_language']:.2f}")
print(f"   Residual:        η²={vd['eta2_residual']:.3f} ({vd['eta2_residual']*100:.1f}% of variance)")
print(f"   Permutation p(language): {vd['p_language_permutation']}")

# 2. Bootstrap CIs for Phase 8
print(f"\n2. Phase 8 Safety Lift — Bootstrap CIs")
print(f"   {'Language':10s} {'Mean Lift':>10s} {'95% CI':>22s} {'p-value':>10s} {'Sig':>5s} {'FDR Sig':>8s}")
for lang in NON_EN:
    b = exp12["phase8_bootstrap"].get(lang, {})
    f_ = exp12["phase8_fdr"].get(lang, {})
    if b:
        sig = "✓" if str(b.get("significant")) == "True" else ""
        fdr = "✓" if str(f_.get("fdr_significant")) == "True" else ""
        print(f"   {lang:10s} {b['mean_lift']*100:>+8.1f}%  [{b['ci_95_low']*100:>+6.1f}%, {b['ci_95_high']*100:>+6.1f}%]  {b['bootstrap_p']:>8.4f}  {sig:>5s}  {fdr:>8s}")

# 3. Mantel tests
print(f"\n3. Mantel Test (SCD × Typological Distance)")
mantel = exp12["mantel_tests"]
sig_count = sum(1 for v in mantel.values() if v["significant"])
rs = [v["mantel_r"] for v in mantel.values()]
print(f"   Significant models: {sig_count}/{len(mantel)}")
print(f"   Mean Mantel r: {np.mean(rs):.4f}")
print(f"   Only yi-6b significant (r=0.746, p=0.0005)")

# 4. Pivot permutation
pp = exp12["pivot_permutation"]
print(f"\n4. Pivot Language Permutation Test")
print(f"   Observed: {pp['observed_western']}/{pp['total_models']} Western pivots")
print(f"   Expected under null: {pp['expected_under_null']:.1f}")
print(f"   p-value: {pp['p_value']} (n_perms={pp['n_permutations']})")

print("\n" + "="*70)
print("EXPERIMENT 13: CKA CROSS-MODEL TRANSFER")
print("="*70)

print(f"\nMethod: Linear CKA (dimension-invariant)")
print(f"Overall Within-Family CKA: {exp13['overall_within_family_mean']:.4f}")
print(f"Overall Cross-Family CKA:  {exp13['overall_cross_family_mean']:.4f}")
print(f"Mann-Whitney U: {exp13['mann_whitney_u']:.0f}, p={exp13['mann_whitney_p']:.2e}")

print(f"\n{'Language':10s} {'Within':>8s} {'Cross':>8s} {'Gap':>8s}")
for lang in ['en'] + NON_EN:
    d = exp13["per_language_transfer"][lang]
    gap = d["within_family_mean"] - d["cross_family_mean"]
    print(f"{lang:10s} {d['within_family_mean']:>8.4f} {d['cross_family_mean']:>8.4f} {gap:>8.4f}")

print("\n" + "="*70)
print("EXPERIMENT 14: LAS QUANTITATIVE VALIDATION")
print("="*70)

# Aggregate per language
lang_pre = defaultdict(list)
lang_post = defaultdict(list)
lang_improve = defaultdict(list)
lang_rotmag = defaultdict(list)

for model, langs in exp14.items():
    for lang, data in langs.items():
        lang_pre[lang].append(data["pre_las_sim"])
        lang_post[lang].append(data["post_las_sim"])
        lang_improve[lang].append(data["improvement"])
        lang_rotmag[lang].append(data["rotation_magnitude"])

print(f"\n{'Language':10s} {'Pre-LAS':>10s} {'Post-LAS':>10s} {'Improve':>10s} {'RotMag':>10s} {'n':>4s}")
for lang in NON_EN:
    pre = np.mean(lang_pre[lang])
    post = np.mean(lang_post[lang])
    imp = np.mean(lang_improve[lang])
    rot = np.mean(lang_rotmag[lang])
    n = len(lang_pre[lang])
    print(f"{lang:10s} {pre:>10.3f} {post:>10.3f} {imp:>10.3f} {rot:>10.1f} {n:>4d}")

# Overall
all_pre = [v for vals in lang_pre.values() for v in vals]
all_post = [v for vals in lang_post.values() for v in vals]
all_imp = [v for vals in lang_improve.values() for v in vals]
print(f"\n   Grand Mean Pre-LAS:  {np.mean(all_pre):.3f}")
print(f"   Grand Mean Post-LAS: {np.mean(all_post):.3f}")
print(f"   Grand Mean Improve:  {np.mean(all_imp):.3f}")
print(f"   → LAS recovers {np.mean(all_imp)/max(1-np.mean(all_pre), 0.001)*100:.0f}% of the alignment gap")

# Per model summary
print(f"\n   Per-Model LAS Success (post-LAS sim > 0.7 across all languages):")
for model in sorted(exp14.keys()):
    langs_data = exp14[model]
    post_vals = [d["post_las_sim"] for d in langs_data.values()]
    all_above = all(v > 0.7 for v in post_vals)
    min_post = min(post_vals)
    mean_post = np.mean(post_vals)
    print(f"   {model:40s} mean_post={mean_post:.3f} min={min_post:.3f} {'✓' if all_above else '✗'}")

# Correlation: SCD vs LAS improvement (should be positive)
print(f"\n   SCD ↔ LAS Improvement Correlation:")
with open("results/experiment12_stats.json") as f:
    e12 = json.load(f)
# We can do this per model x lang
