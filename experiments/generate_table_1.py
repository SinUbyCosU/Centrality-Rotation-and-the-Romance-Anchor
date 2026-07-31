"""
Generate Table 1: Steering Concept Drift (SCD) at Max-Variance vs 2/3rd-Depth Layers
"""
import json, glob
import numpy as np
from pathlib import Path

def generate_table1():
    results_dir = Path('./results')
    
    # Auto-discover all model result directories
    all_model_dirs = {}
    for d in sorted(results_dir.glob('steering_vectors_*')):
        if not d.is_dir():
            continue
        parts = d.name.replace('steering_vectors_', '').rsplit('_', 2)
        model_name = parts[0] if len(parts) >= 3 else d.name.replace('steering_vectors_', '')
        all_model_dirs[model_name] = d

    models = sorted(all_model_dirs.keys())
    
    print("| Model | Best Layer (Max-Variance) | Max-Var SCD (deg) | Best Layer (2/3rd Depth) | 2/3rd-Depth SCD (deg) |")
    print("|---|---|---|---|---|")
    
    for model in models:
        latest_dir = all_model_dirs[model]
        p2_file = latest_dir / 'phase2_pivot_analysis' / 'phase2_results.json'
        
        l_max = "N/A"
        scd_max_mean = "N/A"
        l_2_3 = "N/A"
        scd_2_3_mean = "N/A"
        
        if p2_file.exists():
            d = json.load(open(p2_file))
            l_max = d.get('best_layer_max_variance', 'N/A')
            l_2_3 = d.get('best_layer_2_3_depth', 'N/A')
            
            scd_max = d.get('scd_scores_degrees_max_var', {})
            scd_2_3 = d.get('scd_scores_degrees_2_3_depth', {})
            
            # Use mean SCD across languages as the summary metric, or just average
            if scd_max:
                scd_max_mean = f"{np.mean([v for k,v in scd_max.items() if k != 'en']):.1f}\u00b0"
            if scd_2_3:
                scd_2_3_mean = f"{np.mean([v for k,v in scd_2_3.items() if k != 'en']):.1f}\u00b0"
                
        print(f"| {model} | Layer {l_max} | {scd_max_mean} | Layer {l_2_3} | {scd_2_3_mean} |")

if __name__ == "__main__":
    generate_table1()
