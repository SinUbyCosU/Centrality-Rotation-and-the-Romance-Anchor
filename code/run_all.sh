#!/bin/bash
# run_all.sh - Parallel CLR Paper Pipeline

set -e

MODELS=("mistral-7b" "zephyr-7b" "qwen-7b" "yi-6b" "falcon-7b" "phi-3" "stablelm-3b" "tinyllama")
OUTPUT="./results"
PHASE=${1:-"all"}

mkdir -p $OUTPUT

run_model_pipeline() {
    local MODEL=$1
    local GPU_ID=$2
    
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    
    echo "=================================================="
    echo "Starting pipeline for $MODEL on GPU $GPU_ID"
    echo "=================================================="
    
    # Phase 1
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "1" ] || [ "$PHASE" = "resume" ]; then
        PHASE1_DIR=$(ls -td $OUTPUT/steering_vectors_${MODEL}_* 2>/dev/null | head -1)
        if [ -n "$PHASE1_DIR" ]; then
            echo "Found existing Phase 1 results for $MODEL in $PHASE1_DIR. Skipping extraction."
            echo $PHASE1_DIR > $OUTPUT/phase1_dir_${MODEL}.txt
        else
            echo "Running Phase 1 extraction for $MODEL..."
            python experiments/01_extract_steering_vectors.py --model $MODEL --output $OUTPUT
            PHASE1_DIR=$(ls -td $OUTPUT/steering_vectors_${MODEL}_* 2>/dev/null | head -1)
            echo $PHASE1_DIR > $OUTPUT/phase1_dir_${MODEL}.txt
        fi
    fi
    
    PHASE1_DIR=$(cat $OUTPUT/phase1_dir_${MODEL}.txt 2>/dev/null || echo "")
    if [ -z "$PHASE1_DIR" ]; then
        echo "ERROR: Phase 1 results not found for $MODEL"
        return
    fi
    
    # Phase 2
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "2" ] || [ "$PHASE" = "resume" ]; then
        python experiments/02_pivot_language_analysis.py --input $PHASE1_DIR
    fi
    
    # Phase 4+5
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "4" ] || [ "$PHASE" = "5" ] || [ "$PHASE" = "resume" ]; then
        python experiments/04_05_psycholing_las.py --phase both --input $PHASE1_DIR --phase2 $PHASE1_DIR
    fi
    
    # Phase 6+7
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "6" ] || [ "$PHASE" = "7" ] || [ "$PHASE" = "resume" ]; then
        python experiments/06_07_safety_codemix.py --phase both --input $PHASE1_DIR --phase2 $PHASE1_DIR
    fi
    
    # Figures
    if [ "$PHASE" = "all" ] || [ "$PHASE" = "figures" ] || [ "$PHASE" = "resume" ]; then
        python -c "from utils.visualization import generate_all_figures; generate_all_figures('$PHASE1_DIR', '$OUTPUT/figures_${MODEL}')"
    fi
    
    echo "Pipeline complete for $MODEL on GPU $GPU_ID"
}

# Run 2 models concurrently
for ((i=0; i<${#MODELS[@]}; i+=2)); do
    MODEL_1=${MODELS[$i]}
    MODEL_2=${MODELS[$i+1]}
    
    if [ -n "$MODEL_1" ]; then
        run_model_pipeline $MODEL_1 0 &
    fi
    
    if [ -n "$MODEL_2" ]; then
        run_model_pipeline $MODEL_2 1 &
    fi
    
    wait
done

echo "All models completed successfully!"
