import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Upload the fixed script
s = c.open_sftp()
s.put('C:\\Users\\Tanushree\\Downloads\\work\\experiments\\18_las_behavioral.py', '/root/clr_paper/experiments/18_las_behavioral.py')
s.close()

# Rerun setup_parallel to launch with the fixed script
_, out, err = c.exec_command('python3 /root/run_parallel.py')
print(out.read().decode())
if err.read().decode():
    print("ERR:", err.read().decode())

c.close()
