import os
import subprocess
import sys
import shutil

models = [
    'openhermes-2.5-mistral-7b', 
    'zephyr-7b', 
    'smollm3-3b', 
    'mistral-7b-instruct-v0.3', 
    'stablelm-2-1.6b-chat', 
    'qwen3-4b-instruct', 
    'phi-3-mini-4k-instruct', 
    'qwen2.5-3b-instruct', 
    'qwen2.5-7b-instruct', 
    'qwen2-7b-instruct', 
    'falcon3-7b-instruct'
]

print(f"Starting batch Phase 11 rerun for {len(models)} models...")

for i, model in enumerate(models):
    print(f"\n[{i+1}/{len(models)}] Processing model: {model}")
    
    # Identify and delete old results folder so it is not skipped
    results_dir = "/root/clr_paper/results"
    target_dir = None
    for d in os.listdir(results_dir):
        if d.startswith(f"steering_vectors_{model}_"):
            target_dir = os.path.join(results_dir, d)
            break
    if target_dir:
        phase11_dir = os.path.join(target_dir, "phase11_scaled_adversarial")
        if os.path.exists(phase11_dir):
            print(f"Removing old phase11 dir: {phase11_dir}")
            shutil.rmtree(phase11_dir)
            
    cmd = [
        "bash",
        "-c",
        f"export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 && python3 experiments/11_scaled_adversarial_v5.py --model-key {model}"
    ]
    
    # Run the command and capture output so we can stream it
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        
    process.stdout.close()
    return_code = process.wait()
    
    if return_code != 0:
        print(f"\n[!] BATCH SKIPPED: Model '{model}' failed the sanity check or crashed. Exit code: {return_code}")
        print("Continuing to next model.")
        continue
        
    print(f"[OK] Model '{model}' processed successfully and passed all sanity checks.")
    
print("\n[SUCCESS] Batch Phase 11 rerun completed for all models!")
