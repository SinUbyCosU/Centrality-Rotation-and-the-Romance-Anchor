import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, o, _ = c.exec_command('python3 -c "from experiments.utils import get_results_dir; print(get_results_dir(\'qwen2-7b-instruct\'))"')
with open('p17_dir3.txt', 'w', encoding='utf-8') as f:
    f.write(o.read().decode())
