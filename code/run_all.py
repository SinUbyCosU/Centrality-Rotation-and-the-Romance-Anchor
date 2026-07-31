import os
import sys
import subprocess
import glob
import threading
import queue

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

MODELS = [
    # Original 8 base models
    "mistral-7b", "zephyr-7b", "qwen-7b", "yi-6b", "falcon-7b", "stablelm-3b", "tinyllama",
    # NOTE: "phi-3" removed — it mapped to the same HF checkpoint as phi-3-mini-4k-instruct (duplicate)
    # Extended instruct models
    "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct", "qwen2.5-3b-instruct",
    "qwen3-4b-instruct", "qwen3-8b-instruct", "phi-3-mini-4k-instruct", "phi-4-mini-instruct",
    "falcon-7b-instruct", "bloomz-7b1", "mt0-xl", "olmo-2-7b-instruct", "smollm3-3b", "aya-23-8b",
    "deepseek-r1-distill-llama-8b", "openhermes-2.5-mistral-7b", "yi-1.5-6b-chat", "qwen2-7b-instruct",
    "stablelm-2-1.6b-chat", "internlm2.5-7b-chat", "falcon3-7b-instruct"
]
OUTPUT = "./results"

def run_model_pipeline(model, gpu_id, phase):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print("=" * 50)
    print(f"Starting pipeline for {model} on GPU {gpu_id}")
    print("=" * 50)
    
    # Phase 1
    dirs = sorted(glob.glob(f"{OUTPUT}/steering_vectors_{model}_*"), reverse=True)
    valid_dirs = [d for d in dirs if os.path.exists(os.path.join(d, "all_steering_vectors.pkl"))]
    phase1_dir = valid_dirs[0] if valid_dirs else ""
    
    if phase in ["all", "1", "resume"]:
        if phase1_dir:
            print(f"Found existing Phase 1 results for {model} in {phase1_dir}. Skipping extraction.")
            with open(f"{OUTPUT}/phase1_dir_{model}.txt", "w") as f:
                f.write(phase1_dir + "\n")
        else:
            print(f"Running Phase 1 extraction for {model}...")
            subprocess.run(["python", "-u", "experiments/01_extract_steering_vectors.py", "--model", model, "--output", OUTPUT], env=env, check=True)
            dirs = sorted(glob.glob(f"{OUTPUT}/steering_vectors_{model}_*"), reverse=True)
            valid_dirs = [d for d in dirs if os.path.exists(os.path.join(d, "all_steering_vectors.pkl"))]
            phase1_dir = valid_dirs[0] if valid_dirs else ""
            if phase1_dir:
                with open(f"{OUTPUT}/phase1_dir_{model}.txt", "w") as f:
                    f.write(phase1_dir + "\n")

    try:
        with open(f"{OUTPUT}/phase1_dir_{model}.txt", "r") as f:
            phase1_dir = f.read().strip()
    except FileNotFoundError:
        phase1_dir = ""
        
    if not phase1_dir:
        print(f"ERROR: Phase 1 results not found for {model}")
        return False
        
    # Phase 2
    if phase in ["all", "2", "resume"]:
        subprocess.run(["python", "-u", "experiments/02_pivot_language_analysis.py", "--input", phase1_dir], env=env, check=True)
        
    # Phase 4+5
    if phase in ["all", "4", "5", "resume"]:
        subprocess.run(["python", "-u", "experiments/04_05_psycholing_las.py", "--phase", "both", "--input", phase1_dir, "--phase2", phase1_dir], env=env, check=True)
        
    # Phase 6+7
    if phase in ["all", "6", "7", "resume"]:
        subprocess.run(["python", "-u", "experiments/06_07_safety_codemix.py", "--phase", "both", "--input", phase1_dir, "--phase2", phase1_dir], env=env, check=True)
        
    # Figures
    if phase in ["all", "figures", "resume"]:
        cmd = f"from utils.visualization import generate_all_figures; generate_all_figures(r'{phase1_dir}', r'{OUTPUT}/figures_{model}')"
        subprocess.run(["python", "-u", "-c", cmd], env=env, check=True)
        
    print(f"Pipeline complete for {model} on GPU {gpu_id}")
    return True

def worker(gpu_id, q, phase):
    while True:
        try:
            model = q.get_nowait()
        except queue.Empty:
            break
        try:
            run_model_pipeline(model, gpu_id, phase)
        except Exception as e:
            print(f"Error processing {model} on GPU {gpu_id}: {e}")
        finally:
            q.task_done()

if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    q = queue.Queue()
    for m in MODELS:
        q.put(m)
        
    # Create one worker per GPU
    threads = []
    for gpu_id in [0, 1]:
        t = threading.Thread(target=worker, args=(gpu_id, q, phase))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    print("All models completed successfully!")
