"""
Redeploy ONLY the fixes needed for the 12 failed models.
- Upload fixed model_loader.py
- Install sentencepiece (for OpenHermes)
- Clean up failed result dirs
- Rerun ONLY the 12 failed models
"""
import paramiko
import os
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"
REMOTE_DIR = "/root/clr_paper"
LOCAL_DIR = r"C:\Users\Tanushree\Downloads\work"
HF_TOKEN = "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"

FAILED_MODELS = [
    "falcon-7b", "falcon-7b-instruct",
    "phi-3-mini-4k-instruct", "phi-4-mini-instruct",
    "qwen3-4b-instruct", "qwen3-8b-instruct",
    "mt0-xl", "olmo-2-7b-instruct",
    "aya-23-8b", "deepseek-r1-distill-llama-8b",
    "openhermes-2.5-mistral-7b", "internlm2.5-7b-chat",
]

def get_ssh():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    return client

def ssh_exec(client, cmd, timeout=300):
    print(f"  CMD: {cmd[:140]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(f"  OUT: {out.strip()[:500]}")
    if code != 0 and err.strip():
        print(f"  ERR: {err.strip()[:500]}")
    return out, err, code

def main():
    client = get_ssh()
    sftp = client.open_sftp()
    
    # 1. Upload fixed model_loader.py and steering.py
    print("\n=== STEP 1: Upload fixed files ===")
    for f, remote in [
        ("models/model_loader.py", f"{REMOTE_DIR}/models/model_loader.py"),
        ("utils/steering.py", f"{REMOTE_DIR}/utils/steering.py"),
        ("experiments/01_extract_steering_vectors.py", f"{REMOTE_DIR}/experiments/01_extract_steering_vectors.py"),
    ]:
        local = os.path.join(LOCAL_DIR, f)
        print(f"  UPLOAD: {f}")
        sftp.put(local, remote)
    sftp.close()
    
    # 2. Install sentencepiece (needed for OpenHermes)
    print("\n=== STEP 2: Install sentencepiece ===")
    ssh_exec(client, f"source {REMOTE_DIR}/venv/bin/activate && pip install -q sentencepiece protobuf")
    
    # 3. Clean up failed result directories
    print("\n=== STEP 3: Clean up failed dirs ===")
    for model in FAILED_MODELS:
        ssh_exec(client, f"rm -rf {REMOTE_DIR}/results/steering_vectors_{model}_* {REMOTE_DIR}/results/phase1_dir_{model}.txt")
    
    # 4. Create a retry script that only runs the 12 failed models
    print("\n=== STEP 4: Create retry script ===")
    retry_script = f'''
import os, sys, subprocess, glob, threading, queue
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

MODELS = {FAILED_MODELS}
OUTPUT = "./results"

def run_model_pipeline(model, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print("=" * 50)
    print(f"RETRY: {{model}} on GPU {{gpu_id}}")
    print("=" * 50)
    
    # Phase 1
    dirs = sorted(glob.glob(f"{{OUTPUT}}/steering_vectors_{{model}}_*"), reverse=True)
    valid = [d for d in dirs if os.path.exists(os.path.join(d, "all_steering_vectors.pkl"))]
    phase1_dir = valid[0] if valid else ""
    
    if not phase1_dir:
        print(f"Running Phase 1 for {{model}}...")
        subprocess.run(["python", "-u", "experiments/01_extract_steering_vectors.py", "--model", model, "--output", OUTPUT], env=env, check=True)
        dirs = sorted(glob.glob(f"{{OUTPUT}}/steering_vectors_{{model}}_*"), reverse=True)
        valid = [d for d in dirs if os.path.exists(os.path.join(d, "all_steering_vectors.pkl"))]
        phase1_dir = valid[0] if valid else ""
    
    if not phase1_dir:
        print(f"ERROR: Phase 1 still failed for {{model}}")
        return False
    
    with open(f"{{OUTPUT}}/phase1_dir_{{model}}.txt", "w") as f:
        f.write(phase1_dir + "\\n")
    
    # Phase 2
    subprocess.run(["python", "-u", "experiments/02_pivot_language_analysis.py", "--input", phase1_dir], env=env, check=True)
    # Phase 4+5
    subprocess.run(["python", "-u", "experiments/04_05_psycholing_las.py", "--phase", "both", "--input", phase1_dir, "--phase2", phase1_dir], env=env, check=True)
    # Phase 6+7
    subprocess.run(["python", "-u", "experiments/06_07_safety_codemix.py", "--phase", "both", "--input", phase1_dir, "--phase2", phase1_dir], env=env, check=True)
    # Figures
    cmd = f"from utils.visualization import generate_all_figures; generate_all_figures(r\\'{{phase1_dir}}\\', r\\'{{OUTPUT}}/figures_{{model}}\\')"
    subprocess.run(["python", "-u", "-c", cmd], env=env, check=True)
    
    print(f"RETRY complete for {{model}}")
    return True

def worker(gpu_id, q):
    while True:
        try:
            model = q.get_nowait()
        except queue.Empty:
            break
        try:
            run_model_pipeline(model, gpu_id)
        except Exception as e:
            print(f"RETRY ERROR {{model}} on GPU {{gpu_id}}: {{e}}")
        finally:
            q.task_done()

if __name__ == "__main__":
    os.makedirs(OUTPUT, exist_ok=True)
    q = queue.Queue()
    for m in MODELS:
        q.put(m)
    threads = []
    for gpu_id in [0, 1]:
        t = threading.Thread(target=worker, args=(gpu_id, q))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print("\\nAll retries completed!")
'''
    
    # Write retry script to remote
    stdin, stdout, stderr = client.exec_command(f"cat > {REMOTE_DIR}/retry_failed.py << 'ENDOFSCRIPT'\n{retry_script}\nENDOFSCRIPT")
    stdout.channel.recv_exit_status()
    
    # 5. Verify the fixed model_loader
    print("\n=== STEP 5: Verify fixes ===")
    ssh_exec(client, f"source {REMOTE_DIR}/venv/bin/activate && cd {REMOTE_DIR} && python3 -c \"from models.model_loader import ModelLoader; ml = ModelLoader(); print('Qwen3-4B:', ml.MODEL_MAP['qwen3-4b-instruct']); print('OLMo:', ml.MODEL_MAP['olmo-2-7b-instruct']); print('Seq2Seq:', ml.SEQ2SEQ_MODELS); print('Eager:', ml.EAGER_ATTN_MODELS)\"")
    
    # 6. Launch the retry in tmux
    print("\n=== STEP 6: Launch retry pipeline ===")
    ssh_exec(client, "tmux kill-session -t clr 2>/dev/null; echo done")
    launch_cmd = (
        f"tmux new-session -d -s clr "
        f"'export HF_TOKEN={HF_TOKEN} && "
        f"source {REMOTE_DIR}/venv/bin/activate && "
        f"cd {REMOTE_DIR} && "
        f"python3 -u retry_failed.py 2>&1 | tee retry_failed.log'"
    )
    ssh_exec(client, launch_cmd)
    ssh_exec(client, "tmux list-sessions 2>&1")
    
    import time
    time.sleep(8)
    ssh_exec(client, f"tail -30 {REMOTE_DIR}/retry_failed.log 2>/dev/null")
    
    client.close()
    print("\n=== RETRY DEPLOYMENT COMPLETE ===")
    print(f"12 failed models being retried across 2 GPUs.")

if __name__ == "__main__":
    main()
