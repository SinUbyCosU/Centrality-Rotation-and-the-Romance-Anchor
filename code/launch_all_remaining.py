import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Upload Phase 15 and 16
s = c.open_sftp()
s.put('C:\\Users\\Tanushree\\Downloads\\work\\experiments\\15_dual_process.py',
      '/root/clr_paper/experiments/15_dual_process.py')
s.put('C:\\Users\\Tanushree\\Downloads\\work\\experiments\\16_linguistic_security.py',
      '/root/clr_paper/experiments/16_linguistic_security.py')
s.put('C:\\Users\\Tanushree\\Downloads\\work\\experiments\\11_scaled_adversarial.py',
      '/root/clr_paper/experiments/11_scaled_adversarial.py')
s.close()
print("Uploaded experiments 15, 16, 11")

# Launch Phase 15 (CPU-only) immediately
cmd15 = 'nohup python3 /root/clr_paper/experiments/15_dual_process.py > /root/clr_paper/exp15.log 2>&1 &'
c.exec_command(cmd15)
print("Launched Exp 15 (CPU-only, Dual Process)")

# Launch Phase 16 (CPU-only) immediately
cmd16 = 'nohup python3 /root/clr_paper/experiments/16_linguistic_security.py > /root/clr_paper/exp16.log 2>&1 &'
c.exec_command(cmd16)
print("Launched Exp 16 (CPU-only, Linguistic Security)")

# Create a queue script that waits for Exp 18 then runs Exp 11
queue_script = '''#!/bin/bash
echo "Waiting for Exp 18 to finish..."
while pgrep -f "18_las_behavioral" > /dev/null; do
    sleep 60
done
echo "Exp 18 done. Starting Exp 11 (Scaled Adversarial) on remaining models..."
python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1
echo "Exp 11 complete."
'''

s = c.open_sftp()
with s.file('/root/clr_paper/queue_exp11.sh', 'w') as f:
    f.write(queue_script)
s.close()

c.exec_command('chmod +x /root/clr_paper/queue_exp11.sh')
c.exec_command('nohup bash /root/clr_paper/queue_exp11.sh > /root/clr_paper/queue.log 2>&1 &')
print("Queued Exp 11 (Scaled Adversarial) — will start after Exp 18 finishes")

# Verify everything is running
time.sleep(3)
_, out, _ = c.exec_command('ps aux | grep -E "15_dual|16_linguistic|18_las|queue_exp" | grep -v grep')
print("\nRunning processes:")
print(out.read().decode())

c.close()
