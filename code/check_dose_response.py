import paramiko, re, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# How many models done?
_, out, _ = client.exec_command('grep "^MODEL:" /root/clr_paper/dose_response.log | wc -l')
models_started = out.read().decode().strip()

# How many lang x alpha lines done?
_, out, _ = client.exec_command('grep "alpha=" /root/clr_paper/dose_response.log | wc -l')
alpha_lines = out.read().decode().strip()

# Check for errors
_, out, _ = client.exec_command('grep -i "error\|traceback\|failed" /root/clr_paper/dose_response.log | tail -3')
errors = out.read().decode('utf-8', errors='ignore').strip()

# Check if finished
_, out, _ = client.exec_command('grep "GRAND CORRELATION" /root/clr_paper/dose_response.log')
done = out.read().decode('utf-8', errors='ignore').strip()

# Last few lines
_, out, _ = client.exec_command('tail -8 /root/clr_paper/dose_response.log')
tail = out.read().decode('utf-8', errors='ignore')

# GPU check
_, out, _ = client.exec_command('nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader')
gpu = out.read().decode().strip()

print(f"Models started: {models_started}/5")
print(f"Alpha lines completed: {alpha_lines} / ~550 expected")
if errors:
    print(f"Errors: {errors}")
if done:
    print(f"\n*** EXPERIMENT COMPLETE ***")
    # Get the final results
    _, out, _ = client.exec_command('grep -A 30 "AGGREGATE DOSE" /root/clr_paper/dose_response.log')
    print(out.read().decode('utf-8', errors='ignore'))
else:
    print(f"Status: Running...")
print(f"GPU mem: {gpu}")
print(f"\nLast output:\n{tail}")

client.close()
