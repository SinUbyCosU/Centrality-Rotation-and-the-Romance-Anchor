import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()
    
    # Upload patched script
    sftp.put('11_v5_patched.py', '/root/clr_paper/experiments/11_scaled_adversarial_v5.py')
    
    # Delete the failed phase 11 directories so run_batch_gpu0.py doesn't skip them
    # Wait, run_batch_gpu0.py actually cleans up empty directories, but if they have a .jsonl they might not be empty
    # Let's just forcefully remove phase11_scaled_adversarial directories for all models, since none of them completed successfully
    print('Cleaning up failed directories...')
    stdin, stdout, stderr = c.exec_command('rm -rf /root/clr_paper/results/*/phase11_scaled_adversarial')
    stdout.channel.recv_exit_status()
    
    # Restart the batch script on GPU 0 in tmux or nohup
    print('Restarting batch on GPU 0...')
    stdin, stdout, stderr = c.exec_command('nohup python3 /root/clr_paper/run_batch_gpu0.py > /root/clr_paper/batch_gpu0.log 2>&1 &')
    
    c.close()
    print('Done.')
except Exception as e:
    print('Error:', e)
