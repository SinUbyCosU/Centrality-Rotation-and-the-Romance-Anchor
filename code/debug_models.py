import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

_, out, _ = client.exec_command('python3 -c "import glob; print(len(glob.glob(\'/root/clr_paper/results/steering_vectors_*/all_steering_vectors.pkl\')))"')
print("Model count:", out.read().decode().strip())

_, out, _ = client.exec_command('python3 -c "import glob; pkls = sorted(glob.glob(\'/root/clr_paper/results/steering_vectors_*/all_steering_vectors.pkl\')); models = [pkl.split(\'steering_vectors_\')[1].rsplit(\'_2026\', 1)[0] for pkl in pkls]; target_models = [\'llama-3-8b-instruct\', \'mistral-7b-instruct-v0.2\', \'qwen1.5-7b-chat\', \'gemma-7b-it\', \'vicuna-7b-v1.5\']; models2 = [m for m in models if any(t in m.lower() for t in target_models)][:5]; print(models2); print(models[:5] if not models2 else models2)"')
print("Models script sees:", out.read().decode().strip())
client.close()
