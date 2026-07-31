import os
import subprocess
import sys

models = [
    'openhermes-2.5-mistral-7b',
    'smollm3-3b',
    'zephyr-7b',
    'mistral-7b-instruct-v0.3',
    'stablelm-2-1.6b-chat',
    'qwen3-4b-instruct',
    'phi-3-mini-4k-instruct',
    'qwen2.5-3b-instruct',
    'qwen2.5-7b-instruct',
    'qwen2-7b-instruct',
    'falcon3-7b-instruct'
]

RESULTS_DIR = "/root/clr_paper/results"
EXP_SCRIPT = "/root/clr_paper/experiments/11_scaled_adversarial_v5.py"

print(f"Starting batch Phase 11 on GPU 0 for {len(models)} models...", flush=True)

for i, model in enumerate(models):
    print(f"\n[{i+1}/{len(models)}] Processing model: {model}", flush=True)

    # Check if already completed successfully - look for the results json
    completed = False
    for d in os.listdir(RESULTS_DIR):
        if d.startswith(f"steering_vectors_{model}_"):
            phase11_dir = os.path.join(RESULTS_DIR, d, "phase11_scaled_adversarial")
            out_file = os.path.join(phase11_dir, "phase11_results.json")
            if os.path.exists(out_file):
                print(f"  [SKIP] Already completed: {out_file}", flush=True)
                completed = True
                break
            # If directory exists but no result json, it's a partial run - remove it to restart cleanly
            if os.path.exists(phase11_dir):
                import shutil
                print(f"  Removing partial/failed phase11 dir: {phase11_dir}", flush=True)
                shutil.rmtree(phase11_dir)
            break

    if completed:
        continue

    # Run with CUDA_VISIBLE_DEVICES=0 to pin to GPU 0, NEVER touching GPU 1
    # PYTHONIOENCODING=utf-8 prevents crashes on unicode output (tqdm bars etc.)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    env["PYTHONIOENCODING"] = "utf-8"

    cmd = [
        "python3", EXP_SCRIPT,
        "--model-key", model,
        "--results-dir", RESULTS_DIR
    ]

    print(f"  Running: {' '.join(cmd)}", flush=True)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd="/root/clr_paper"
        )

        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            print(f"\n[WARNING] Model '{model}' exited with code {return_code}. Checking if results were saved...", flush=True)
        else:
            print(f"[OK] Model '{model}' finished (exit 0).", flush=True)

    except Exception as e:
        print(f"\n[ERROR] Exception running model '{model}': {e}", flush=True)

    print(f"  Moving to next model.\n", flush=True)

print("\n[SUCCESS] Batch Phase 11 GPU 0 completed!")
