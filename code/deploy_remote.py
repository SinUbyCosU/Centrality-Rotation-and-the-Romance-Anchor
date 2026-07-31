"""
Deploy the FIXED CLR pipeline to the remote GPU server.
Uploads all fixed files (bugs resolved), sets HF_TOKEN, and relaunches in tmux.
"""
import paramiko
import os

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"
REMOTE_DIR = "/root/clr_paper"
LOCAL_DIR = r"C:\Users\Tanushree\Downloads\work"
HF_TOKEN = "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"

def get_ssh():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    return client

def ssh_exec(client, cmd, timeout=300):
    print(f"  CMD: {cmd[:120]}...")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(f"  OUT: {out.strip()[:500]}")
    if err.strip() and code != 0:
        print(f"  ERR: {err.strip()[:500]}")
    return out, err, code

def upload_file(sftp, local_path, remote_path):
    print(f"  UPLOAD: {os.path.basename(local_path)} -> {remote_path}")
    sftp.put(local_path, remote_path)

def main():
    client = get_ssh()
    sftp = client.open_sftp()
    
    # ---- 1. Kill existing pipeline ----
    print("\n=== STEP 1: Stopping existing pipeline ===")
    ssh_exec(client, "tmux kill-session -t clr 2>/dev/null; echo done")
    
    # ---- 2. Upload ALL fixed files ----
    print("\n=== STEP 2: Uploading fixed source files ===")
    
    files_to_upload = [
        # Core scripts
        ("run_all.py", f"{REMOTE_DIR}/run_all.py"),
        
        # Fixed utils
        ("utils/steering.py", f"{REMOTE_DIR}/utils/steering.py"),
        ("utils/languages.py", f"{REMOTE_DIR}/utils/languages.py"),
        ("utils/psycholing.py", f"{REMOTE_DIR}/utils/psycholing.py"),
        ("utils/visualization.py", f"{REMOTE_DIR}/utils/visualization.py"),
        ("utils/geometry.py", f"{REMOTE_DIR}/utils/geometry.py"),
        
        # Fixed experiments
        ("experiments/01_extract_steering_vectors.py", f"{REMOTE_DIR}/experiments/01_extract_steering_vectors.py"),
        ("experiments/02_pivot_language_analysis.py", f"{REMOTE_DIR}/experiments/02_pivot_language_analysis.py"),
        ("experiments/04_05_psycholing_las.py", f"{REMOTE_DIR}/experiments/04_05_psycholing_las.py"),
        ("experiments/06_07_safety_codemix.py", f"{REMOTE_DIR}/experiments/06_07_safety_codemix.py"),
        ("experiments/load_results.py", f"{REMOTE_DIR}/experiments/load_results.py"),
        ("experiments/aggregate_results.py", f"{REMOTE_DIR}/experiments/aggregate_results.py"),
        
        # Model loader (unchanged but re-upload for consistency)
        ("models/model_loader.py", f"{REMOTE_DIR}/models/model_loader.py"),
        
        # All 11 language data files
        ("data/safety_prompts_en.json", f"{REMOTE_DIR}/data/safety_prompts_en.json"),
        ("data/safety_prompts_hi.json", f"{REMOTE_DIR}/data/safety_prompts_hi.json"),
        ("data/safety_prompts_ar.json", f"{REMOTE_DIR}/data/safety_prompts_ar.json"),
        ("data/safety_prompts_fr.json", f"{REMOTE_DIR}/data/safety_prompts_fr.json"),
        ("data/safety_prompts_de.json", f"{REMOTE_DIR}/data/safety_prompts_de.json"),
        ("data/safety_prompts_es.json", f"{REMOTE_DIR}/data/safety_prompts_es.json"),
        ("data/safety_prompts_ru.json", f"{REMOTE_DIR}/data/safety_prompts_ru.json"),
        ("data/safety_prompts_zh-CN.json", f"{REMOTE_DIR}/data/safety_prompts_zh-CN.json"),
        ("data/safety_prompts_ja.json", f"{REMOTE_DIR}/data/safety_prompts_ja.json"),
        ("data/safety_prompts_sw.json", f"{REMOTE_DIR}/data/safety_prompts_sw.json"),
        ("data/safety_prompts_pt.json", f"{REMOTE_DIR}/data/safety_prompts_pt.json"),
    ]
    
    for local_rel, remote_path in files_to_upload:
        local_path = os.path.join(LOCAL_DIR, local_rel)
        if os.path.exists(local_path):
            upload_file(sftp, local_path, remote_path)
        else:
            print(f"  SKIP (not found): {local_rel}")

    sftp.close()
    
    # ---- 3. Wipe ALL old results (must rerun with fixed code) ----
    print("\n=== STEP 3: Wiping all old results ===")
    ssh_exec(client, f"rm -rf {REMOTE_DIR}/results/steering_vectors_* {REMOTE_DIR}/results/phase1_dir_* {REMOTE_DIR}/results/figures_*")
    
    # ---- 4. Verify deployment ----
    print("\n=== STEP 4: Verification ===")
    ssh_exec(client, f"source {REMOTE_DIR}/venv/bin/activate && cd {REMOTE_DIR} && python3 -c \"from run_all import MODELS; print('Models:', len(MODELS), MODELS[:5], '...')\"")
    ssh_exec(client, f"ls {REMOTE_DIR}/data/safety_prompts_*.json | wc -l")
    ssh_exec(client, f"source {REMOTE_DIR}/venv/bin/activate && cd {REMOTE_DIR} && python3 -c \"from utils.languages import LANGUAGE_FAMILIES; print('zh-CN family:', LANGUAGE_FAMILIES.get('zh-CN')); print('pt family:', LANGUAGE_FAMILIES.get('pt'))\"")
    ssh_exec(client, f"source {REMOTE_DIR}/venv/bin/activate && cd {REMOTE_DIR} && python3 -c \"from utils.psycholing import PSYCHOLINGUISTIC_DISTANCES; print('zh-CN dist:', PSYCHOLINGUISTIC_DISTANCES.get('zh-CN')); print('pt dist:', PSYCHOLINGUISTIC_DISTANCES.get('pt'))\"")
    
    # ---- 5. Launch pipeline in tmux WITH HF_TOKEN ----
    print("\n=== STEP 5: Launching fixed pipeline ===")
    ssh_exec(client, "tmux kill-session -t clr 2>/dev/null; echo done")
    launch_cmd = (
        f"tmux new-session -d -s clr "
        f"'export HF_TOKEN={HF_TOKEN} && "
        f"source {REMOTE_DIR}/venv/bin/activate && "
        f"cd {REMOTE_DIR} && "
        f"python3 -u run_all.py all 2>&1 | tee run_fixed.log'"
    )
    ssh_exec(client, launch_cmd)
    ssh_exec(client, "tmux list-sessions 2>&1")
    ssh_exec(client, "sleep 5 && tail -20 /root/clr_paper/run_fixed.log 2>/dev/null")
    
    client.close()
    print("\n=== DEPLOYMENT COMPLETE ===")
    print("Fixed pipeline running in tmux session 'clr' with HF_TOKEN set.")
    print(f"27 models x 11 languages. ETA: ~24-36 hours.")

if __name__ == "__main__":
    main()
