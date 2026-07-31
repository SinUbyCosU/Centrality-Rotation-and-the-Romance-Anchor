import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Get the last 15 lines of the log safely
stdin, stdout, stderr = client.exec_command("tail -n 15 /root/clr_paper/acl_run_exp17.log | sed 's/\x1b\[[0-9;]*m//g'")
print(stdout.read().decode('utf-8', errors='replace'))

client.close()
