import paramiko, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

sftp = client.open_sftp()
for script in ["15_dual_process.py", "16_linguistic_security.py", "17_moral_foundations.py"]:
    print(f"Uploading {script}...")
    sftp.put(f"experiments/{script}", f"/root/clr_paper/experiments/{script}")

print("\nRunning Experiment 15 (Dual Process, CPU only)...")
stdin, stdout, stderr = client.exec_command("python3 /root/clr_paper/experiments/15_dual_process.py")
print(stdout.read().decode())
print(stderr.read().decode())

print("\nRunning Experiment 16 (Linguistic Security, CPU only)...")
stdin, stdout, stderr = client.exec_command("python3 /root/clr_paper/experiments/16_linguistic_security.py")
print(stdout.read().decode())
print(stderr.read().decode())

# Queue Experiment 17 to run after Experiment 11
run_script = """#!/bin/bash
echo "Waiting for Experiment 11 to finish before running 17..." > /root/clr_paper/acl_run_exp17.log
while pgrep -f "11_scaled_adversarial.py" > /dev/null; do
    sleep 60
done
echo "Experiment 11 finished. Starting Experiment 17..." >> /root/clr_paper/acl_run_exp17.log
date >> /root/clr_paper/acl_run_exp17.log
python3 /root/clr_paper/experiments/17_moral_foundations.py --model-key all >> /root/clr_paper/acl_run_exp17.log 2>&1
echo "Experiment 17 completed." >> /root/clr_paper/acl_run_exp17.log
"""

with sftp.file("/root/clr_paper/run_exp17_queued.sh", "w") as f:
    f.write(run_script)
sftp.close()

print("\nQueueing Experiment 17 to run after Exp 11 in a new tmux session...")
client.exec_command("chmod +x /root/clr_paper/run_exp17_queued.sh")
client.exec_command("tmux new-session -d -s acl_exp17 'bash /root/clr_paper/run_exp17_queued.sh'")
print("Done! Exp 17 will wait for GPUs to free up and then start automatically.")
client.close()
