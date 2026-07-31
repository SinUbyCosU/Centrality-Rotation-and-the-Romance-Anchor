import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Kill any lingering processes
c.exec_command('pkill -f 18_las_behavioral')
c.exec_command('pkill -f queue_exp11')

# Re-launch Exp 18
cmd = 'nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key all --n-prompts 30 > /root/clr_paper/exp18_las_v3.log 2>&1 &'
c.exec_command(cmd)
print("Relaunched Exp 18")

# Re-queue Exp 11 after Exp 18
queue_script = '''#!/bin/bash
echo "Waiting for Exp 18 v3 to finish..."
while pgrep -f "18_las_behavioral" > /dev/null; do
    sleep 60
done
echo "Exp 18 done. Starting Exp 11..."
python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1
echo "Exp 11 complete."
'''
s = c.open_sftp()
with s.file('/root/clr_paper/queue_exp11_v3.sh', 'w') as f:
    f.write(queue_script)
s.close()
c.exec_command('chmod +x /root/clr_paper/queue_exp11_v3.sh')
c.exec_command('nohup bash /root/clr_paper/queue_exp11_v3.sh > /root/clr_paper/queue_v3.log 2>&1 &')
print("Re-queued Exp 11 after Exp 18")

c.close()
