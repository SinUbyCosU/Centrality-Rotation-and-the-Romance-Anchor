import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Check v2 log
_, out, _ = c.exec_command('tail -40 /root/clr_paper/exp18_las_v2.log')
print("=== EXP 18 V2 LOG ===")
print(out.read().decode('utf-8', errors='replace').encode('cp1252', errors='replace').decode('cp1252'))

# Count completed
_, out, _ = c.exec_command('grep "Saved to" /root/clr_paper/exp18_las_v2.log | wc -l')
print("Models saved in v2:", out.read().decode().strip())

# Check for errors
_, out, _ = c.exec_command('grep -c "device" /root/clr_paper/exp18_las_v2.log')
print("Device errors in v2:", out.read().decode().strip())

# List phase18 dirs
_, out, _ = c.exec_command('ls -d /root/clr_paper/results/steering_vectors_*/phase18_las_behavioral 2>/dev/null | wc -l')
print("Phase 18 result dirs:", out.read().decode().strip())

c.close()
