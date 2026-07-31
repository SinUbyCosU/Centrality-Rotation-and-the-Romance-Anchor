import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Upload the script
s = c.open_sftp()
s.put('C:\\Users\\Tanushree\\Downloads\\work\\experiments\\18_las_behavioral.py',
      '/root/clr_paper/experiments/18_las_behavioral.py')
s.close()
print("Uploaded 18_las_behavioral.py")

# Launch in background with nohup
cmd = 'nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key all --n-prompts 30 > /root/clr_paper/exp18_las.log 2>&1 &'
_, out, err = c.exec_command(cmd)
print("Launched:", cmd)
print("STDOUT:", out.read().decode())
print("STDERR:", err.read().decode())

# Verify it's running
import time
time.sleep(2)
_, out, _ = c.exec_command('ps aux | grep 18_las')
print("\nProcess check:")
print(out.read().decode())

c.close()
