"""Install missing deps and restart Exp 11."""
import paramiko, sys, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

def run(cmd, timeout=300):
    print(f">>> {cmd}")
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    if out: print(out[-2000:])
    if err: print(f"STDERR: {err[-500:]}")
    return out

# Install accelerate + bitsandbytes (needed for 4-bit quantization)
run("pip3 install accelerate bitsandbytes --break-system-packages 2>&1 | tail -5")

# Verify
run('python3 -c "import accelerate; print(\'accelerate:\', accelerate.__version__); import bitsandbytes; print(\'bitsandbytes:\', bitsandbytes.__version__)"')

# Create run script for just Exp 11
run_script = """#!/bin/bash
cd /root/clr_paper
echo "=== Experiment 11 (Restart) ===" > acl_run_exp11.log
date >> acl_run_exp11.log

echo "--- RUNNING EXPERIMENT 11 (Scaled Adversarial - GPU) ---" >> acl_run_exp11.log
python3 experiments/11_scaled_adversarial.py --model-key all >> acl_run_exp11.log 2>&1

echo "--- EXPERIMENT 11 COMPLETED ---" >> acl_run_exp11.log
date >> acl_run_exp11.log
"""

sftp = client.open_sftp()
with sftp.file("/root/clr_paper/run_exp11.sh", "w") as f:
    f.write(run_script)
sftp.close()

# Launch in tmux
run("tmux kill-session -t acl_experiments 2>/dev/null; echo ok")
run("chmod +x /root/clr_paper/run_exp11.sh && tmux new-session -d -s acl_experiments 'bash /root/clr_paper/run_exp11.sh'")
time.sleep(5)
print("\n" + run("tmux has-session -t acl_experiments 2>/dev/null && echo 'TMUX RUNNING' || echo 'FAILED'"))
print("\n" + run("tail -10 /root/clr_paper/acl_run_exp11.log"))

client.close()
print("\nDone!")
