#!/bin/bash
# run_exp18_2_3_depth.sh
# This script executes Experiment 18 (LAS Behavioral Intervention) across all models at the 2/3rd depth layer.
# It computes bootstrapped CIs for the LAS refusal rate lift.

# Navigate to the correct directory if needed
cd "$(dirname "$0")/.."

echo "Starting Experiment 18 (2/3rd Depth LAS Behavioral Validation) across all models..."

# Find all steering vector directories
for sv_dir in results/steering_vectors_*; do
    if [ -d "$sv_dir" ]; then
        model_name=$(basename "$sv_dir" | sed 's/steering_vectors_\(.*\)_2026.*/\1/')
        echo "=========================================================="
        echo "Processing $model_name"
        echo "=========================================================="
        
        # Run Exp 18. Output will be saved to phase18_las_behavioral inside the sv_dir.
        python experiments/18_las_behavioral.py --input "$sv_dir"
    fi
done

echo "Experiment 18 completed."
