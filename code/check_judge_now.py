import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    print('Checking latest judge.log...')
    _, out, _ = c.exec_command('tail -n 20 /root/clr_paper/judge.log')
    print(out.read().decode('utf-8', errors='replace').strip())
    c.close()
except Exception as e:
    print('Error:', e)
