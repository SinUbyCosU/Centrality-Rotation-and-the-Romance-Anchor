import paramiko
import time

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

models_to_run = ["phi-4-mini-instruct", "qwen2.5-7b-instruct", "zephyr-7b"]

for model in models_to_run:
    print(f"Running Exp 17 for {model} in isolated process...")
    # Run synchronously to ensure fully isolated processes
    stdin, stdout, stderr = client.exec_command(f"python3 /root/clr_paper/experiments/17_moral_foundations.py --model-key {model}")
    exit_status = stdout.channel.recv_exit_status()
    print(f"Finished {model} with status {exit_status}")
    print(stdout.read().decode('utf-8', errors='replace'))
    if exit_status != 0:
        print("Error:")
        print(stderr.read().decode('utf-8', errors='replace'))

client.close()
print("All 3 isolated runs complete.")
