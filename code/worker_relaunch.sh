#!/bin/bash
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
cd /root/clr_paper
for model in qwen2-7b-instruct smollm3-3b stablelm-2-1.6b-chat stablelm-3b tinyllama; do
    echo "Worker 1 Relaunch: Starting $model"
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_1_$model
    export TRITON_CACHE_DIR=/tmp/triton_1_$model
    timeout 3600 /root/clr_paper/venv/bin/python experiments/11_scaled_adversarial.py --model-key "$model" --results-dir results
done
