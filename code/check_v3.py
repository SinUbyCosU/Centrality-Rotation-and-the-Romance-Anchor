import paramiko
c=paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
_,out,_ = c.exec_command('grep "Saved to" /root/clr_paper/exp18_las_v3.log | wc -l')
print('Models saved in v3:', out.read().decode().strip())
c.close()
