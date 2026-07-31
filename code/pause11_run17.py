import paramiko
import time

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

print("Killing Experiment 11...")
client.exec_command("pkill -9 -f '11_scaled_adversarial.py'")

time.sleep(3)
stdin, stdout, stderr = client.exec_command("pgrep -f '11_scaled_adversarial.py'")
running_11 = stdout.read().decode().strip()
print(f"Exp 11 PID after kill: {running_11}")

print("Checking if Exp 17 started automatically...")
time.sleep(5)
stdin, stdout, stderr = client.exec_command("pgrep -f '17_moral_foundations.py'")
running_17 = stdout.read().decode().strip()
print(f"Exp 17 PID: {running_17}")

client.close()
