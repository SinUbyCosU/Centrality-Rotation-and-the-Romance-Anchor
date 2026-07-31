import paramiko
import json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, _ = c.exec_command(
    'find /root/.cache/huggingface/hub -name "config.json" | grep -i phi-3-mini-4k-instruct'
)
config_paths = out.read().decode().strip().split('\n')

for path in config_paths:
    if not path.strip(): continue
    print(f"\n--- {path} ---")
    _, out2, _ = c.exec_command(f'cat {path}')
    try:
        config = json.loads(out2.read().decode())
        print("rope_scaling:", config.get("rope_scaling", "NOT FOUND"))
    except Exception as e:
        print("Error parsing JSON:", e)

c.close()
