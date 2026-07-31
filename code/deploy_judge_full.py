import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

print("Uploading llm_judge_v3.py...")
sftp.put('llm_judge_v3.py', '/root/clr_paper/llm_judge_v3.py')

print("Starting full judge evaluation...")
c.exec_command("rm -f /root/clr_paper/judge_full.log")
_, out, _ = c.exec_command('nohup bash -c "export CUDA_VISIBLE_DEVICES=0 && python3 /root/clr_paper/llm_judge_v3.py" > /root/clr_paper/judge_full.log 2>&1 & echo $!')

print("Judge PID:", out.read().decode().strip())
c.close()
