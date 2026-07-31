import json, math, glob
import numpy as np
from pathlib import Path

def run():
    results_dir = Path("results")
    for p2_file in results_dir.glob("steering_vectors_*/phase2_pivot_analysis/phase2_results.json"):
        print(f"Fixing {p2_file}...")
        try:
            with open(p2_file, "r") as f:
                data = json.load(f)
            
            if "scd_per_layer" not in data:
                print("  No scd_per_layer found, skipping.")
                continue
                
            scd_per_layer = data["scd_per_layer"]
            
            # Find best_layer_max_variance
            layer_variances = {}
            for layer_str, scores in scd_per_layer.items():
                valid_scores = [v for k, v in scores.items() if k != "en" and not math.isnan(v)]
                if valid_scores:
                    layer_variances[int(layer_str)] = np.var(valid_scores)
            
            if not layer_variances:
                continue
                
            best_layer_max_variance = max(layer_variances, key=layer_variances.__getitem__)
            
            # Find 2/3rd depth layer
            layers_list = sorted([int(k) for k in scd_per_layer.keys()])
            if layers_list:
                best_layer_2_3_depth = layers_list[len(layers_list) * 2 // 3]
            else:
                best_layer_2_3_depth = best_layer_max_variance
                
            # Extract SCD scores for both
            scd_scores_rad_max_var = scd_per_layer[str(best_layer_max_variance)]
            scd_scores_rad_2_3 = scd_per_layer[str(best_layer_2_3_depth)]
            
            scd_deg_max_var = {k: math.degrees(v) for k, v in scd_scores_rad_max_var.items() if not math.isnan(v)}
            scd_deg_2_3 = {k: math.degrees(v) for k, v in scd_scores_rad_2_3.items() if not math.isnan(v)}
            
            data["best_layer_max_variance"] = best_layer_max_variance
            data["best_layer_2_3_depth"] = best_layer_2_3_depth
            data["scd_scores_degrees_max_var"] = scd_deg_max_var
            data["scd_scores_degrees_2_3_depth"] = scd_deg_2_3
            
            with open(p2_file, "w") as f:
                json.dump(data, f, indent=2)
                
            print(f"  Fixed! Max-Var: {best_layer_max_variance}, 2/3rd: {best_layer_2_3_depth}")
            
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    run()
