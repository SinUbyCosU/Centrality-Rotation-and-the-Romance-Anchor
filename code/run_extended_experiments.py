"""
run_extended_experiments.py

Runs Experiments 8, 9, 10 across all 20 completed models.
Designed to keep 2 GPUs busy for ~5-7 days.

Experiment 8: Causal Intervention    (~2 days)
Experiment 9: Cross-Model Transfer   (~0.5 day, mostly offline)
Experiment 10: Adversarial Probing   (~3-4 days)
"""
import os
import sys
import subprocess
import glob
import threading
import queue
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

OUTPUT = "./results"

def get_completed_models():
    """Find all models with completed Phase 1 results."""
    models = {}
    for d in sorted(Path(OUTPUT).glob("steering_vectors_*")):
        if (d / "all_steering_vectors.pkl").exists():
            name = d.name.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
            models[name] = str(d)
    return models


def run_experiment(model, results_dir, gpu_id, experiment):
    """Run a single experiment for a single model."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    if experiment == "8":
        print(f"[GPU {gpu_id}] Exp 8 (Causal Intervention): {model}")
        subprocess.run([
            "python", "-u", "experiments/08_causal_intervention.py",
            "--model", model, "--input", results_dir, "--n-prompts", "30"
        ], env=env, check=True)
        
    elif experiment == "9":
        print(f"[GPU {gpu_id}] Exp 9 (Cross-Model Transfer): {model}")
        subprocess.run([
            "python", "-u", "experiments/09_cross_model_transfer.py",
            "--results", OUTPUT
        ], env=env, check=True)
        
    elif experiment == "10":
        print(f"[GPU {gpu_id}] Exp 10 (Adversarial Probing): {model}")
        subprocess.run([
            "python", "-u", "experiments/10_adversarial_probing.py",
            "--model", model, "--input", results_dir, "--n-prompts", "5"
        ], env=env, check=True)


def worker(gpu_id, task_queue):
    """Worker thread for a single GPU."""
    while True:
        try:
            model, results_dir, experiment = task_queue.get_nowait()
        except queue.Empty:
            break
        try:
            run_experiment(model, results_dir, gpu_id, experiment)
        except Exception as e:
            print(f"[GPU {gpu_id}] ERROR {experiment} on {model}: {e}")
        finally:
            task_queue.task_done()


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    models = get_completed_models()
    print(f"Found {len(models)} completed models")
    print(f"Running experiment phase: {phase}\n")
    
    task_queue = queue.Queue()
    
    # Experiment 9 only needs to run once (not per-model)
    if phase in ["all", "9"]:
        task_queue.put(("all", OUTPUT, "9"))
    
    # Experiments 8 and 10 run per-model
    for model_name, results_dir in sorted(models.items()):
        if phase in ["all", "8"]:
            # Check if already done
            if not (Path(results_dir) / "phase8_causal_intervention" / "phase8_results.json").exists():
                task_queue.put((model_name, results_dir, "8"))
        
        if phase in ["all", "10"]:
            if not (Path(results_dir) / "phase10_adversarial_probing" / "phase10_results.json").exists():
                task_queue.put((model_name, results_dir, "10"))
    
    print(f"Total tasks queued: {task_queue.qsize()}")
    
    # Run with 2 GPU workers
    threads = []
    for gpu_id in [0, 1]:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    print("\n" + "="*60)
    print("All extended experiments completed!")
    print("="*60)
