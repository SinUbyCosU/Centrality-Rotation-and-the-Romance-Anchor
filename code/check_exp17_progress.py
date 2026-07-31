import paramiko
import time

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

stdin, stdout, stderr = client.exec_command("grep 'Experiment 17:' /root/clr_paper/acl_run_exp17.log")
lines = stdout.read().decode('utf-8', errors='replace').strip().split('\n')
print(f"Total models started for Exp 17: {len(lines)}")
for line in lines[-3:]:
    print("  " + line.strip())

client.close()
