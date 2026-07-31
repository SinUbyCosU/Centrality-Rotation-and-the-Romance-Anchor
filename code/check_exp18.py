import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Check how many Exp 18 models completed
_, out, _ = c.exec_command('ls -d /root/clr_paper/results/steering_vectors_*/phase18_las_behavioral 2>/dev/null | wc -l')
print("Exp 18 completed models:", out.read().decode().strip())

# Check if process is still running
_, out, _ = c.exec_command('ps aux | grep 18_las | grep -v grep')
proc = out.read().decode().strip()
print("Process running:", proc if proc else "NO (finished or killed)")

# Check queue status
_, out, _ = c.exec_command('ps aux | grep queue_exp | grep -v grep')
q = out.read().decode().strip()
print("Queue running:", q if q else "NO")

# Check if Exp 11 ran
_, out, _ = c.exec_command('ls -d /root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial 2>/dev/null | wc -l')
print("Exp 11 models:", out.read().decode().strip())

# Tail of Exp 18 log
_, out, _ = c.exec_command('tail -30 /root/clr_paper/exp18_las.log')
print("\n=== EXP 18 LOG TAIL ===")
print(out.read().decode('utf-8', errors='replace').encode('cp1252', errors='replace').decode('cp1252'))

c.close()
