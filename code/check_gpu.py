import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

print("\n--- Running Exp 11 Processes ---")
stdin, stdout, stderr = client.exec_command('ps aux | grep 11_scaled | grep -v grep')
print(stdout.read().decode('utf-8'))

print("\n--- NVIDIA SMI ---")
stdin, stdout, stderr = client.exec_command('nvidia-smi')
print(stdout.read().decode('utf-8'))

client.close()
