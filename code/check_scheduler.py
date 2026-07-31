import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

print("=== SCHEDULER LOG ===")
_, out, _ = c.exec_command('tail -n 10 /root/clr_paper/scheduler.log')
print(out.read().decode())

print("\n=== RUNNING PYTHON PROCESSES ===")
_, out, _ = c.exec_command('pgrep -a python')
print(out.read().decode())

c.close()
