import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()
    sftp.get('/root/clr_paper/run_batch_gpu0.py', 'run_batch_gpu0.py')
    c.close()
    print('Downloaded.')
except Exception as e:
    print('Error:', e)
