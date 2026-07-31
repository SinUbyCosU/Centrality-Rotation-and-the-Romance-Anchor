import paramiko
import time
import subprocess
import os

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

print("Waiting for remote run_tasks.py to finish...")
while True:
    _, stdout, _ = client.exec_command('ps aux | grep "[r]un_tasks.py"')
    out = stdout.read().decode().strip()
    if not out:
        break
    time.sleep(30)

print("Remote tasks finished! Now running fix_and_aggregate.py...")
client.close()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
subprocess.run("python fix_and_aggregate.py > fix_agg_final.txt", shell=True)
print("Done!")
