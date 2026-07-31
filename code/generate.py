import json
import numpy as np

d = json.load(open('unified_results.json'))['models']
models = list(d.keys())

high_risk = ['zh-CN', 'ar', 'hi', 'sw', 'ru']
low_risk = ['fr', 'de', 'es', 'pt', 'ja']

print("## Table B — SCD x Robustness (Cohen's d, corrected, 14/18 models)")
print("| Model | Cohen's d | High-Risk Mean | Low-Risk Mean |")
print("|---|---|---|---|")

for m in sorted(models):
    if m in ['bloomz-7b1', 'stablelm-3b', 'yi-6b']: continue # Exclude base models
    scd_dict = d[m].get('scd_degrees', {})
    if not scd_dict: continue
    
    high_scd = [scd_dict[l] for l in high_risk if l in scd_dict]
    low_scd = [scd_dict[l] for l in low_risk if l in scd_dict]
    
    if len(high_scd) > 1 and len(low_scd) > 1:
        mean_h, mean_l = np.mean(high_scd), np.mean(low_scd)
        var_h, var_l = np.var(high_scd, ddof=1), np.var(low_scd, ddof=1)
        pooled_std = np.sqrt(((len(high_scd)-1)*var_h + (len(low_scd)-1)*var_l) / (len(high_scd)+len(low_scd)-2))
        cohen_d = (mean_h - mean_l) / (pooled_std + 1e-10)
        print(f"| {m} | {cohen_d:.3f} | {mean_h:.1f}° | {mean_l:.1f}° |")

print("\n## Table E — Causal Intervention (Phase 8)")
print("| Model | r(SCD, Lift) |")
print("|---|---|")
for m in sorted(models):
    if m in ['bloomz-7b1', 'stablelm-3b', 'yi-6b']: continue
    p8 = d[m].get('phase8_correlation', {})
    if not p8:
        print(f"| {m} | N/A |")
    else:
        r = p8.get('pearson_r', 'N/A')
        p = p8.get('pearson_p', 'N/A')
        if type(r) == float:
            print(f"| {m} | {r:.3f} (p={p:.4f}) |")
        else:
            print(f"| {m} | {r} |")

