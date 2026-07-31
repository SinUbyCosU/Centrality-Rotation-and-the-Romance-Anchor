import paramiko
import json

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

stdin, stdout, stderr = client.exec_command('cat /root/clr_paper/results/steering_vectors_bloomz-7b1_20260622_092805/phase11_scaled_adversarial/phase11_results.json')
try:
    data = json.loads(stdout.read().decode('utf-8'))
    print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)

client.close()
