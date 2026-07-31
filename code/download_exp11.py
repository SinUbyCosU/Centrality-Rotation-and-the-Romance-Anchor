import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

sftp = client.open_sftp()
sftp.get('/root/clr_paper/experiments/11_scaled_adversarial.py', '11_scaled_adversarial_remote.py')
sftp.close()
client.close()
