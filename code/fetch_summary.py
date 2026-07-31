import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()
    
    # Run simple grep command on the remote
    stdin, stdout, stderr = c.exec_command('grep -E "Exit|SUCCESS|OK|FAILED" /root/clr_paper/batch_gpu0.log > /root/clr_paper/batch_summary.txt')
    stdout.channel.recv_exit_status()
    
    # Download the text file
    sftp.get('/root/clr_paper/batch_summary.txt', 'batch_summary.txt')
    c.close()
    
    # Read it locally with utf-8 replace
    with open('batch_summary.txt', 'r', encoding='utf-8', errors='replace') as f:
        print(f.read())
except Exception as e:
    print('Error:', e)
