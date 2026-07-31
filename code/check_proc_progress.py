import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
_, o, _ = c.exec_command('tail -n 20 /root/clr_paper/phase17_nohup.log; echo "---"; tail -n 20 /root/clr_paper/phase11_nohup.log')
print(o.read().decode())
