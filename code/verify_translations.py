import paramiko
import json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, _ = c.exec_command(
    'grep translation_attack /root/clr_paper/results/steering_vectors_mistral-7b-instruct-v0.3_20260622_081644/phase11_scaled_adversarial/phase11_raw_responses.jsonl | head -n 3'
)
lines = out.read().decode('utf-8').strip().split('\n')
for line in lines:
    if line.strip():
        try:
            data = json.loads(line)
            print(f"[{data['language']}] Prompt: {data['prompt'][:100]}...")
        except Exception as e:
            print(f"Error parsing: {e}")

c.close()
