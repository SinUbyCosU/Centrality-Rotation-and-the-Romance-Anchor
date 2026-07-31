"""Hot-patch: fix Phi/Falcon models and requeue them."""
import paramiko, os, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"
REMOTE_DIR = "/root/clr_paper"
LOCAL_DIR = r"C:\Users\Tanushree\Downloads\work"
HF_TOKEN = "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"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

def ssh_exec(cmd):
    print(f"  CMD: {cmd[:140]}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip(): print(f"  OUT: {out.strip()[:500]}")
    if code != 0 and err.strip(): print(f"  ERR: {err.strip()[:300]}")
    return out, err, code

# 1. Upload fixed model_loader.py
print("=== Uploading fixed model_loader.py ===")
sftp = client.open_sftp()
sftp.put(os.path.join(LOCAL_DIR, "models/model_loader.py"), f"{REMOTE_DIR}/models/model_loader.py")
sftp.close()

# 2. Clear cached Phi remote code that's causing the crash
print("\n=== Clearing cached Phi remote code ===")
ssh_exec("rm -rf /root/.cache/huggingface/modules/transformers_modules/microsoft/Phi*")

# 3. Check what's currently running (don't kill it — qwen3 models are proceeding)
print("\n=== Current tmux status ===")
ssh_exec("tmux list-sessions 2>&1")

# 4. Wait for current retry to finish, then start a second retry for just phi+falcon
# Create a small script that waits and then retries
retry2_script = '''#!/bin/bash
set -e
export HF_TOKEN=''' + HF_TOKEN + '''
source /root/clr_paper/venv/bin/activate
cd /root/clr_paper

# Wait for the main retry to finish
echo "Waiting for main retry to complete..."
while tmux has-session -t clr 2>/dev/null; do
    sleep 30
done
echo "Main retry finished. Starting Phi/Falcon retry..."

MODELS="falcon-7b falcon-7b-instruct phi-3-mini-4k-instruct phi-4-mini-instruct"
GPU=0
for model in $MODELS; do
    echo "===================================================="
    echo "RETRY2: $model on GPU $GPU"
    echo "===================================================="
    rm -rf results/steering_vectors_${model}_* results/phase1_dir_${model}.txt
    
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/01_extract_steering_vectors.py --model $model --output ./results 2>&1 || {
        echo "Phase 1 FAILED for $model, skipping"
        continue
    }
    
    DIR=$(ls -td results/steering_vectors_${model}_* 2>/dev/null | head -1)
    if [ -z "$DIR" ] || [ ! -f "$DIR/all_steering_vectors.pkl" ]; then
        echo "No valid results for $model, skipping"
        continue
    fi
    
    echo "$DIR" > results/phase1_dir_${model}.txt
    
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/02_pivot_language_analysis.py --input $DIR 2>&1 || true
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/04_05_psycholing_las.py --phase both --input $DIR --phase2 $DIR 2>&1 || true
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/06_07_safety_codemix.py --phase both --input $DIR --phase2 $DIR 2>&1 || true
    python -u -c "from utils.visualization import generate_all_figures; generate_all_figures('$DIR', 'results/figures_${model}')" 2>&1 || true
    
    echo "RETRY2 complete for $model"
    
    # Alternate GPUs
    if [ $GPU -eq 0 ]; then GPU=1; else GPU=0; fi
done

echo "All Phi/Falcon retries done!"
'''

print("\n=== Writing retry2 script ===")
stdin, stdout, stderr = client.exec_command(f"cat > {REMOTE_DIR}/retry2_phi_falcon.sh << 'ENDOFSCRIPT'\n{retry2_script}\nENDOFSCRIPT")
stdout.channel.recv_exit_status()
ssh_exec(f"chmod +x {REMOTE_DIR}/retry2_phi_falcon.sh")

# Launch retry2 in a separate tmux session that waits for the first to finish
print("\n=== Launching retry2 (waits for retry1 to finish) ===")
ssh_exec(f"tmux new-session -d -s retry2 'bash {REMOTE_DIR}/retry2_phi_falcon.sh 2>&1 | tee {REMOTE_DIR}/retry2.log'")
ssh_exec("tmux list-sessions 2>&1")

# Check current progress
time.sleep(3)
print("\n=== Current retry progress ===")
ssh_exec("tail -10 /root/clr_paper/retry_failed.log 2>/dev/null")

client.close()
print("\n=== DONE ===")
print("retry2 session will auto-start Phi/Falcon once the current qwen3 retry finishes.")
