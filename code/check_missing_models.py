import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Get the exact 20 models from Phase 18
stdin, stdout, stderr = client.exec_command('ls -d /root/clr_paper/results/steering_vectors_*/phase18_las_behavioral 2>/dev/null')
exp18_dirs = stdout.read().decode('utf-8').strip().split('\n')

exp18_models = set()
for d in exp18_dirs:
    if not d: continue
    name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    exp18_models.add(name)

# Get models that finished Phase 11
stdin, stdout, stderr = client.exec_command('find /root/clr_paper/results/steering_vectors_* -name "phase11_results.json" 2>/dev/null')
exp11_dirs = stdout.read().decode('utf-8').strip().split('\n')

exp11_models = set()
for d in exp11_dirs:
    if not d: continue
    name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    exp11_models.add(name)

print(f"Total Target Models (from Exp 18): {len(exp18_models)}")
print(f"Target Models that finished Phase 11: {len(exp11_models.intersection(exp18_models))}")

missing = exp18_models - exp11_models
running_models = {"zephyr-7b", "qwen3-4b-instruct"}
really_missing = missing - running_models

print(f"Target Models still needing to run Phase 11: {len(really_missing)}")
for m in sorted(really_missing):
    print(f"  - {m}")

client.close()
