import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Check Exp 15
_, out, _ = c.exec_command('ls -la /root/clr_paper/results/experiment15_dual_process.json 2>/dev/null; tail -5 /root/clr_paper/exp15.log 2>/dev/null')
print("=== EXP 15 ===")
print(out.read().decode('utf-8', errors='replace'))

# Check Exp 16
_, out, _ = c.exec_command('ls -la /root/clr_paper/results/experiment16_linguistic_security.json 2>/dev/null; tail -5 /root/clr_paper/exp16.log 2>/dev/null')
print("=== EXP 16 ===")
print(out.read().decode('utf-8', errors='replace'))

# Check Exp 18 progress
_, out, _ = c.exec_command('tail -5 /root/clr_paper/exp18_las.log 2>/dev/null')
print("=== EXP 18 (tail) ===")
print(out.read().decode('utf-8', errors='replace').encode('cp1252', errors='replace').decode('cp1252'))

# Check queue
_, out, _ = c.exec_command('cat /root/clr_paper/queue.log 2>/dev/null')
print("=== QUEUE LOG ===")
print(out.read().decode('utf-8', errors='replace'))

c.close()
