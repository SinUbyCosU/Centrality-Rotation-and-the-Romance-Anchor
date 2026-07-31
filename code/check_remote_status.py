import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    
    print('--- WHAT IS RUNNING ---')
    _, out, _ = c.exec_command('ps aux | grep python')
    print(out.read().decode('utf-8').strip())
    
    print('\n--- NVIDIA SMI ---')
    _, out, _ = c.exec_command('nvidia-smi')
    print(out.read().decode('utf-8').strip())
    
    c.close()
except Exception as e:
    print('SSH Error:', e)
