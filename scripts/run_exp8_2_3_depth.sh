#!/bin/bash
# run_exp8_2_3_depth.sh
# This script executes Experiment 8 (Causal Intervention) across all models at the 2/3rd depth layer.
# It computes bootstrapped CIs for the refusal rate lift.

# Navigate to the correct directory if needed
cd "$(dirname "$0")/.."

echo "Starting Experiment 8 (2/3rd Depth Causal Intervention) across all models..."

# Find all steering vector directories
for sv_dir in results/steering_vectors_*; do
    if [ -d "$sv_dir" ]; then
        model_name=$(basename "$sv_dir" | sed 's/steering_vectors_\(.*\)_2026.*/\1/')
        echo "=========================================================="
        echo "Processing $model_name"
        echo "=========================================================="
        
        # Run Exp 8. Output will be saved to phase8_causal_intervention inside the sv_dir.
        python experiments/08_causal_intervention.py --input "$sv_dir"
    fi
done

echo "Experiment 8 completed."
