import json
import csv
from pathlib import Path

def generate_csvs():
    input_file = Path("results/unified_results.json")
    output_dir = Path("results/csv_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(input_file, "r") as f:
        data = json.load(f)
        
    models_data = data.get("models", {})
    if not models_data:
        print("No models found in the JSON file.")
        return
        
    models = list(models_data.keys())
    
    def safe_get(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return ""
        return d

    # scd_degrees.csv
    langs_scd = set()
    for m in models:
        langs_scd.update(models_data[m].get("scd_degrees", {}).keys())
    langs_scd = sorted(list(langs_scd))
    
    with open(output_dir / "scd_degrees.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + langs_scd)
        for m in models:
            row = [m] + [safe_get(models_data[m], "scd_degrees", lang) for lang in langs_scd]
            writer.writerow(row)
            
    # centrality_scores.csv
    langs_cent = set()
    for m in models:
        langs_cent.update(models_data[m].get("centrality_scores", {}).keys())
    langs_cent = sorted(list(langs_cent))
    
    with open(output_dir / "centrality_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + langs_cent)
        for m in models:
            row = [m] + [safe_get(models_data[m], "centrality_scores", lang) for lang in langs_cent]
            writer.writerow(row)
            
    # Phase 5: safety_lift and baseline_refusal_rate
    langs_p5 = set()
    for m in models:
        p5 = models_data[m].get("phase5_las", {})
        if isinstance(p5, dict):
            langs_p5.update(p5.keys())
    langs_p5 = sorted(list(langs_p5))
    
    if langs_p5:
        with open(output_dir / "safety_lift.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + langs_p5)
            for m in models:
                row = [m] + [safe_get(models_data[m], "phase5_las", lang, "safety_lift") for lang in langs_p5]
                writer.writerow(row)
                
        with open(output_dir / "baseline_refusal_rate.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + langs_p5)
            for m in models:
                row = [m] + [safe_get(models_data[m], "phase5_las", lang, "baseline_refusal_rate") for lang in langs_p5]
                writer.writerow(row)

    # Phase 8: overall_asr
    langs_p8 = set()
    for m in models:
        p8 = models_data[m].get("phase8", {})
        if isinstance(p8, dict):
            langs_p8.update(p8.keys())
    langs_p8 = sorted(list(langs_p8))
    
    if langs_p8:
        with open(output_dir / "overall_asr.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + langs_p8)
            for m in models:
                row = [m] + [safe_get(models_data[m], "phase8", lang, "overall_asr") for lang in langs_p8]
                writer.writerow(row)
                
    # overall_metrics.csv
    overall_fields = ["model", "pivot_language", "pivot_projection", "best_layer", "psycholinguistic_improvement", 
                      "phase4_typological_r_squared", "phase4_psycholinguistic_r_squared", "phase4_combined_r_squared"]
    with open(output_dir / "overall_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(overall_fields)
        for m in models:
            row = [safe_get(models_data[m], field) for field in overall_fields]
            writer.writerow(row)
            
    print(f"Successfully generated CSV files in {output_dir}")

if __name__ == "__main__":
    generate_csvs()
