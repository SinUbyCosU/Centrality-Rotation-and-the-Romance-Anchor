"""Download and display Exp 12, 13, 14 results."""
import paramiko, sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

sftp = client.open_sftp()
for remote, local in [
    ("/root/clr_paper/results/experiment12_stats.json", "results/experiment12_stats.json"),
    ("/root/clr_paper/results/experiment13_cka.json", "results/experiment13_cka.json"),
    ("/root/clr_paper/results/experiment14_las_validation.json", "results/experiment14_las_validation.json"),
]:
    print(f"Downloading {remote}...")
    sftp.get(remote, local)
sftp.close()

# Also check Exp 11 progress
stdin, stdout, stderr = client.exec_command("tail -15 /root/clr_paper/acl_run_exp11.log 2>&1")
print("\n--- Exp 11 Progress ---")
print(stdout.read().decode("utf-8", errors="replace"))

client.close()
print("Done!")
