import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, _ = c.exec_command('grep -C 5 "Phase 17 results saved" /root/clr_paper/exp17_all.log')
print(out.read().decode('utf-8', errors='replace').encode('cp1252', errors='replace').decode('cp1252'))
