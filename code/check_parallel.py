import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

print("=== GPU USAGE ===")
_, out, _ = c.exec_command('nvidia-smi')
print(out.read().decode('utf-8'))

print("\n=== GPU 0 LOG (Last 5 lines) ===")
_, out, _ = c.exec_command('tail -n 5 /root/clr_paper/exp18_gpu0.log')
print(out.read().decode('utf-8', errors='replace').encode('cp1252', errors='replace').decode('cp1252'))

print("\n=== GPU 1 LOG (Last 5 lines) ===")
_, out, _ = c.exec_command('tail -n 5 /root/clr_paper/exp18_gpu1.log')
print(out.read().decode('utf-8', errors='replace').encode('cp1252', errors='replace').decode('cp1252'))

c.close()
