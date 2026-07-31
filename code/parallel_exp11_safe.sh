#!/bin/bash
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
cd /root/clr_paper

MODELS=(
phi-3-mini-4k-instruct
phi-4-mini-instruct
qwen2.5-3b-instruct
qwen2.5-7b-instruct
qwen2-7b-instruct
qwen3-4b-instruct
qwen3-8b-instruct
qwen-7b
smollm3-3b
stablelm-2-1.6b-chat
stablelm-3b
tinyllama
yi-1.5-6b-chat
yi-6b
zephyr-7b
)

for worker_idx in 0 1; do
cat << INNER_EOF > worker_${worker_idx}.sh
#!/bin/bash
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
cd /root/clr_paper
for model in "\$@"; do
    echo "Worker ${worker_idx}: Starting \$model"
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torch_${worker_idx}_\$model
    export TRITON_CACHE_DIR=/tmp/triton_${worker_idx}_\$model
    /root/clr_paper/venv/bin/python experiments/11_scaled_adversarial.py --model-key "\$model" --results-dir results
done
INNER_EOF
chmod +x worker_${worker_idx}.sh
done

CUDA_VISIBLE_DEVICES=0 nohup ./worker_0.sh "${MODELS[@]:0:8}" > worker_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup ./worker_1.sh "${MODELS[@]:8:7}" > worker_1.log 2>&1 &

echo "2 sequential workers launched! 1 per GPU!"
