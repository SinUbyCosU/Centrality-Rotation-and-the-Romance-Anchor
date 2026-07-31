"""Deploy and run Experiments 11-14 on the remote server via tmux."""
import paramiko, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

sftp = client.open_sftp()
for local, remote in [
    ("experiments/11_scaled_adversarial.py", "/root/clr_paper/experiments/11_scaled_adversarial.py"),
    ("experiments/12_statistical_rigor.py", "/root/clr_paper/experiments/12_statistical_rigor.py"),
    ("experiments/13_cka_transfer.py", "/root/clr_paper/experiments/13_cka_transfer.py"),
    ("experiments/14_quantitative_validation.py", "/root/clr_paper/experiments/14_quantitative_validation.py"),
]:
    print(f"Uploading {local}...")
    sftp.put(local, remote)

# Create a shell script on the remote server to run everything sequentially
run_script_content = """#!/bin/bash
cd /root/clr_paper
echo "Starting ACL Experiments (11-14)..." > acl_run.log

echo "--- RUNNING EXPERIMENT 12 (Statistical Rigor) ---" >> acl_run.log
python3 experiments/12_statistical_rigor.py >> acl_run.log 2>&1

echo "--- RUNNING EXPERIMENT 13 (CKA Transfer) ---" >> acl_run.log
python3 experiments/13_cka_transfer.py >> acl_run.log 2>&1

echo "--- RUNNING EXPERIMENT 14 (LAS Validation) ---" >> acl_run.log
python3 experiments/14_quantitative_validation.py >> acl_run.log 2>&1

echo "--- RUNNING EXPERIMENT 11 (Scaled Adversarial Probing) ---" >> acl_run.log
# Note: This will take a long time (GPUs needed)
python3 experiments/11_scaled_adversarial.py --model-key all >> acl_run.log 2>&1

echo "--- ALL EXPERIMENTS COMPLETED ---" >> acl_run.log
"""

with sftp.file("/root/clr_paper/run_all.sh", "w") as f:
    f.write(run_script_content)

sftp.close()

# Start tmux session (detached)
print("\nStarting tmux session 'acl_experiments' on the remote server...")
stdin, stdout, stderr = client.exec_command(
    "chmod +x /root/clr_paper/run_all.sh && "
    "tmux kill-session -t acl_experiments 2>/dev/null; "
    "tmux new-session -d -s acl_experiments 'bash /root/clr_paper/run_all.sh'"
)
print("Tmux session started successfully.")
print("To check progress, SSH in and run: tmux attach -t acl_experiments")
print("Or view the log file: tail -f /root/clr_paper/acl_run.log")

client.close()
