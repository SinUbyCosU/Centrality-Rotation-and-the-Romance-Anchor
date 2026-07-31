import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

stdin, stdout, stderr = client.exec_command("ls -la /root/clr_paper/results/experiment15*.json /root/clr_paper/results/experiment16*.json")
print(stdout.read().decode())
client.close()
