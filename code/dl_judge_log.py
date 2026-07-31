import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()
    
    # Download the judge log
    sftp.get('/root/clr_paper/judge.log', 'judge_remote.log')
    
    c.close()
    print('Log downloaded.')
except Exception as e:
    print('Error:', e)
