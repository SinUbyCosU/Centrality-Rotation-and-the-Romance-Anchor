"""Check errors for failed models and diagnose issues."""
import paramiko
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

def ssh_exec(client, cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out.strip(), err.strip()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

# Get list of all dirs vs successful dirs
all_dirs, _ = ssh_exec(client, "ls -d /root/clr_paper/results/steering_vectors_*/")
success, _ = ssh_exec(client, "ls /root/clr_paper/results/steering_vectors_*/all_steering_vectors.pkl")

all_set = set()
for line in all_dirs.strip().split("\n"):
    line = line.strip().rstrip("/")
    all_set.add(line)

success_set = set()
for line in success.strip().split("\n"):
    line = line.strip().replace("/all_steering_vectors.pkl", "")
    success_set.add(line)

failed = all_set - success_set
print(f"Total: {len(all_set)}, Success: {len(success_set)}, Failed: {len(failed)}\n")

for d in sorted(failed):
    model = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    print(f"{'='*60}")
    print(f"FAILED: {model}")
    print(f"Dir: {d}")
    
    # Check what files exist
    files, _ = ssh_exec(client, f"ls {d}/")
    print(f"Files: {files}")
    
    # Check the run log for this model's error
    out, _ = ssh_exec(client, f"grep -A 20 'Error processing {model}' /root/clr_paper/run_fixed.log | head -25")
    if out:
        print(f"Error from log:\n{out}")
    else:
        # Try to find the error differently
        out2, _ = ssh_exec(client, f"grep -B 2 -A 10 '{model}' /root/clr_paper/run_fixed.log | grep -i -A 5 'error\\|traceback\\|exception\\|fail\\|oom\\|killed\\|cannot\\|denied'")
        if out2:
            print(f"Error context:\n{out2[:500]}")
        else:
            print("No error found in log - checking if model dir is just empty")
    print()

client.close()
