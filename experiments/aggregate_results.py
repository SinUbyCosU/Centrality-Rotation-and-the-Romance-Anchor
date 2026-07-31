"""Cross-model aggregation of CLR experiment results."""
import json, glob, os
import sys
import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass
from pathlib import Path

results_dir = Path('./results')

# Auto-discover all model result directories
all_model_dirs = {}
for d in sorted(results_dir.glob('steering_vectors_*')):
    if not d.is_dir():
        continue
    # Check it has actual results (all_steering_vectors.pkl)
    if not (d / 'all_steering_vectors.pkl').exists():
        continue
    # Extract model name from directory name
    parts = d.name.replace('steering_vectors_', '').rsplit('_', 2)
    model_name = parts[0] if len(parts) >= 3 else d.name.replace('steering_vectors_', '')
    # Keep only the latest run per model
    all_model_dirs[model_name] = d

models = sorted(all_model_dirs.keys())
print(f"Found {len(models)} completed models\n")

# Summary table
print('| Model | Pivot Language | Romance Sim | Germanic Sim | \u0394R\u00b2 (Psycho) | LOO R\u00b2 (SCD) | Bootstrap R\u00b2 | Cohen\'s d |')
print('|---|---|---|---|---|---|---|---|')

# Cross-model SCD aggregation
cross_model_scd = {}  # {lang: [scd_across_models]}
pivot_counts = {}

for model in models:
    latest_dir = all_model_dirs[model]
    
    # Phase 2
    p2_file = latest_dir / 'phase2_pivot_analysis/phase2_results.json'
    pivot = 'N/A'
    rom_sim = 'N/A'
    ger_sim = 'N/A'
    if p2_file.exists():
        d = json.load(open(p2_file))
        pivot = d.get('pivot_centrality', 'N/A')
        pivot_counts[pivot] = pivot_counts.get(pivot, 0) + 1
        fams = d.get('family_clustering_scores', {})
        rom_sim = f"{fams.get('Romance', 0):.3f}" if 'Romance' in fams else 'N/A'
        ger_sim = f"{fams.get('Germanic', 0):.3f}" if 'Germanic' in fams else 'N/A'
        # Collect per-language SCD for cross-model aggregation
        scd = d.get('scd_scores_degrees', {})
        for lang, score in scd.items():
            if lang != 'en':
                cross_model_scd.setdefault(lang, []).append(score)

    # Phase 4
    p4_file = latest_dir / 'phase4_psycholinguistic/phase4_results.json'
    delta_r2 = 'N/A'
    if p4_file.exists():
        d = json.load(open(p4_file))
        delta_r2 = f"{d.get('psycholinguistic_improvement', 0):.3f}"

    # Phase 6
    p6_file = latest_dir / 'phase6_safety_prediction/phase6_results.json'
    safe_r2 = 'N/A'
    boot_r2 = 'N/A'
    cohen_d = 'N/A'
    if p6_file.exists():
        d = json.load(open(p6_file))
        reg = d.get('regression', {})
        safe_r2 = f"{reg.get('loo_r2_scd', 0):.3f}"
        boot_r2 = f"{reg.get('bootstrap_r2_scd_mean', 0):.3f}"
        cohen_d = f"{reg.get('cohens_d', float('nan')):.3f}"

    print(f'| {model} | {pivot} | {rom_sim} | {ger_sim} | {delta_r2} | {safe_r2} | {boot_r2} | {cohen_d} |')

# Cross-model aggregation summary
print(f"\n\n## Cross-Model Aggregation (N={len(models)} models)")
print("\n### Pivot Language Consistency")
for lang, count in sorted(pivot_counts.items(), key=lambda x: -x[1]):
    print(f"  {lang}: detected as pivot in {count}/{len(models)} models ({100*count/len(models):.0f}%)")

print("\n### Mean SCD \u00b1 Std Across Models (degrees)")
print("| Language | Mean SCD\u00b0 | Std\u00b0 | N models |")
print("|---|---|---|---|")
for lang in sorted(cross_model_scd.keys()):
    scds = cross_model_scd[lang]
    print(f"| {lang} | {np.mean(scds):.1f} | {np.std(scds):.1f} | {len(scds)} |")
