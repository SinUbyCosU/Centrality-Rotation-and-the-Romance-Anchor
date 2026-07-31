import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    _, out, _ = c.exec_command('grep -A 2 -B 20 "Exit code: 1" /root/clr_paper/batch_gpu0.log')
    print(out.read().decode('utf-8'))
    c.close()
except Exception as e:
    print('SSH Error:', e)
