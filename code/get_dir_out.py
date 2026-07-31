import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
_, o, e = c.exec_command('python3 -c "from experiments.utils import get_results_dir; print(get_results_dir(\'qwen2-7b-instruct\'))"')
print("OUT:", o.read().decode())
print("ERR:", e.read().decode())
