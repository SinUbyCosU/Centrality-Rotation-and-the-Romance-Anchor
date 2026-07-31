import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Check what experiment results exist
_, out, _ = client.exec_command('find /root/clr_paper/results -name "*.json" -type f | head -40')
print("JSON results files:")
print(out.read().decode('utf-8', errors='ignore'))

# Check what data files are available
_, out, _ = client.exec_command('ls /root/clr_paper/data/ 2>/dev/null || echo "No data dir"')
print("\nData files:")
print(out.read().decode('utf-8', errors='ignore'))

# Check GPU availability
_, out, _ = client.exec_command('nvidia-smi --query-gpu=name,memory.free --format=csv,noheader')
print("GPU status:")
print(out.read().decode('utf-8', errors='ignore'))

# Check if any processes are running
_, out, _ = client.exec_command('ps aux | grep python | grep -v grep')
print("Running Python processes:")
print(out.read().decode('utf-8', errors='ignore'))

# Check what safety prompts exist
_, out, _ = client.exec_command('ls /root/clr_paper/data/safety_prompts_* 2>/dev/null | head -20')
print("Safety prompt files:")
print(out.read().decode('utf-8', errors='ignore'))

client.close()
