import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

stdin, stdout, stderr = client.exec_command("grep -c 'results saved' /root/clr_paper/acl_run_exp17.log")
saved_count = stdout.read().decode().strip()

stdin, stdout, stderr = client.exec_command("grep -c 'FAILED' /root/clr_paper/acl_run_exp17.log")
failed_count = stdout.read().decode().strip()

print(f"Models successfully completed (results saved): {saved_count}")
print(f"Models failed: {failed_count}")

client.close()
