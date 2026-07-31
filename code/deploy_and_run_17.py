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

client.exec_command("tmux kill-session -t acl_exp17")
client.exec_command("tmux new-session -d -s acl_exp17 'python3 /root/clr_paper/experiments/17_moral_foundations.py --model-key all > /root/clr_paper/acl_run_exp17.log 2>&1'")

client.close()
print("Fix deployed and Exp 17 restarted.")
