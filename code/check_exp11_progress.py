import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Get the count of models started
stdin, stdout, stderr = client.exec_command('grep "Experiment 11:" /root/clr_paper/acl_run_exp11.log')
lines = stdout.read().decode('utf-8', errors='replace').strip().split('\n')
print(f"Total models started so far: {len(lines)}")
for line in lines[-5:]:  # show the last 5 models started
    print("  " + line.strip())

print("\n--- Checking Exp 17 Status ---")
stdin, stdout, stderr = client.exec_command('cat /root/clr_paper/acl_run_exp17.log | head -n 15')
result = stdout.read().decode('utf-8', errors='replace')
print(result.encode('cp1252', errors='replace').decode('cp1252'))

client.close()
