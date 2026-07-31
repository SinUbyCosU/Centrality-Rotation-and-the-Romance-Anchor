"""Deploy and launch extended experiments (8, 9, 10) on remote GPU server."""
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

# 1. Upload new experiment files
print("=== Uploading new experiment files ===")
sftp = client.open_sftp()
for f in [
    "experiments/08_causal_intervention.py",
    "experiments/09_cross_model_transfer.py",
    "experiments/10_adversarial_probing.py",
    "run_extended_experiments.py",
    "models/model_loader.py",
]:
    local = os.path.join(LOCAL_DIR, f)
    remote = f"{REMOTE_DIR}/{f}"
    if os.path.exists(local):
        print(f"  UPLOAD: {f}")
        sftp.put(local, remote)
sftp.close()

# 2. Kill any existing sessions
print("\n=== Clearing old sessions ===")
ssh_exec("tmux kill-server 2>/dev/null; echo done")

# 3. Launch extended experiments in tmux
print("\n=== Launching extended experiments ===")
launch_cmd = (
    f"tmux new-session -d -s experiments "
    f"'export HF_TOKEN={HF_TOKEN} && "
    f"source {REMOTE_DIR}/venv/bin/activate && "
    f"cd {REMOTE_DIR} && "
    f"python3 -u run_extended_experiments.py all 2>&1 | tee extended_experiments.log'"
)
ssh_exec(launch_cmd)

time.sleep(5)
ssh_exec("tmux list-sessions 2>&1")
ssh_exec(f"tail -20 {REMOTE_DIR}/extended_experiments.log 2>/dev/null")

client.close()
print("\n=== DEPLOYMENT COMPLETE ===")
print("Extended experiments running in tmux session 'experiments'.")
print("ETA: ~5-7 days for all 3 experiments across 20 models.")
