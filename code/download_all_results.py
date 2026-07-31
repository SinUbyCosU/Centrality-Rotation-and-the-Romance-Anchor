import paramiko, json

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Get Exp 11 partial results
stdin, stdout, stderr = client.exec_command("cat /root/clr_paper/acl_run_exp11.log")
with open("exp11_partial_log.txt", "w", encoding="utf-8") as f:
    f.write(stdout.read().decode("utf-8", errors="replace"))

# Get all result JSON files
for exp in [12, 13, 14, 15, 16]:
    fname = f"experiment{exp}_*.json"
    stdin, stdout, stderr = client.exec_command(f"cat /root/clr_paper/results/{fname}")
    data = stdout.read().decode("utf-8", errors="replace")
    if data.strip():
        with open(f"exp{exp}_results.json", "w", encoding="utf-8") as f:
            f.write(data)
        print(f"Exp {exp}: downloaded ({len(data)} bytes)")
    else:
        print(f"Exp {exp}: no results yet")

# Check for any phase11 results saved so far
stdin, stdout, stderr = client.exec_command("find /root/clr_paper/results -name 'phase11*' -o -name '*experiment11*' | head -20")
print("\nExp 11 result files:")
print(stdout.read().decode("utf-8", errors="replace"))

# Check for per-model phase11 results
stdin, stdout, stderr = client.exec_command("find /root/clr_paper/results/steering_vectors_* -name 'phase11*' | head -20")
print("Per-model phase11 files:")
print(stdout.read().decode("utf-8", errors="replace"))

client.close()
print("\nDone downloading.")
