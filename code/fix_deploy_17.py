import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

sftp = client.open_sftp()
sftp.put('experiments/17_moral_foundations.py', '/root/clr_paper/experiments/17_moral_foundations.py')
sftp.close()

# Re-queue Exp 17
client.exec_command("tmux kill-session -t acl_exp17")
client.exec_command("tmux new-session -d -s acl_exp17 'bash /root/clr_paper/run_exp17_queued.sh'")

client.close()
print("Fixed and re-queued.")
