import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()
    
    print('--- RUNNING PROCESSES ---')
    _, out, _ = c.exec_command('ps aux | grep python3 | grep -v grep')
    print(out.read().decode('utf-8', errors='replace').strip())
    
    print('\n--- GPU STATUS ---')
    _, out, _ = c.exec_command('nvidia-smi')
    print(out.read().decode('utf-8', errors='replace').strip())
    
    # Wait 10s then check the log
    time.sleep(10)
    print('\n--- BATCH LOG (first 30 lines) ---')
    _, out, _ = c.exec_command('head -n 30 /root/clr_paper/batch_gpu0.log')
    print(out.read().decode('utf-8', errors='replace').strip())
    
    c.close()
except Exception as e:
    print(f'Error: {e}')
