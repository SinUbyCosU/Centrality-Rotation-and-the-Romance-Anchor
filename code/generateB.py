import json
import numpy as np

d = json.load(open('unified_results.json'))['models']
models = list(d.keys())

high_risk = ['zh-CN', 'ar', 'hi', 'sw', 'ru']
low_risk = ['fr', 'de', 'es', 'pt', 'ja']

with open('tableB_fixed.md', 'w', encoding='utf-8') as f:
    f.write("| Model | Cohen's d | High-Risk Mean | Low-Risk Mean |\n")
    f.write("|---|---|---|---|\n")

    for m in sorted(models):
        if m in ['bloomz-7b1', 'stablelm-3b', 'yi-6b', 'mistral-7b']: continue
        scd_dict = d[m].get('scd_degrees', {})
        if not scd_dict: continue
        
        high_scd = [scd_dict[l] for l in high_risk if l in scd_dict]
        low_scd = [scd_dict[l] for l in low_risk if l in scd_dict]
        
        if len(high_scd) > 1 and len(low_scd) > 1:
            mean_h, mean_l = np.mean(high_scd), np.mean(low_scd)
            var_h, var_l = np.var(high_scd, ddof=1), np.var(low_scd, ddof=1)
            pooled_std = np.sqrt(((len(high_scd)-1)*var_h + (len(low_scd)-1)*var_l) / (len(high_scd)+len(low_scd)-2))
            cohen_d = (mean_h - mean_l) / (pooled_std + 1e-10)
            f.write(f"| {m} | {cohen_d:.3f} | {mean_h:.1f}° | {mean_l:.1f}° |\n")
