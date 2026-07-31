"""Check remote experiment status."""
import paramiko, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

def run(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    return stdout.read().decode("utf-8", errors="replace").strip()

print("Tmux:", run("tmux has-session -t acl_experiments 2>/dev/null && echo 'RUNNING' || echo 'STOPPED'"))
print("\nGPU Usage:")
print(run("nvidia-smi | head -20"))
print("\nResult files:")
print(run("ls -lh /root/clr_paper/results/experiment*.json 2>&1"))
print("\nLast 40 lines of log:")
print(run("tail -40 /root/clr_paper/acl_run.log 2>&1"))

client.close()
