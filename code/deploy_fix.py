import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# 1. Upload the patched script
print("Uploading patched 18_las_behavioral.py...")
s = c.open_sftp()
s.put(r'C:\Users\Tanushree\Downloads\work\experiments\18_las_behavioral.py',
      '/root/clr_paper/experiments/18_las_behavioral.py')
s.close()

# 2. Kill ALL hung processes
print("Killing hung processes...")
c.exec_command("pkill -9 -f '18_las_behavioral.py'")
c.exec_command("pkill -9 -f 'dynamic_scheduler.py'")
c.exec_command("pkill -9 -f 'compile_worker'")

import time; time.sleep(2)

# 3. Clean partial results for the 4 models that were deadlocked
print("Cleaning partial results for deadlocked models...")
for model in ['mistral-7b', 'openhermes-2.5-mistral-7b', 'zephyr-7b', 'qwen-7b']:
    c.exec_command(f"rm -rf /root/clr_paper/results/steering_vectors_{model}_*/phase18_las_behavioral")

# 4. Restart the dynamic scheduler (watchdog will keep it alive)
print("Restarting dynamic scheduler...")
c.exec_command("nohup python3 /root/clr_paper/dynamic_scheduler.py > /dev/null 2>&1 &")

import time; time.sleep(3)

# 5. Verify
_, out, _ = c.exec_command('pgrep -a -f 18_las_behavioral')
print("Running 18_las processes:", out.read().decode().strip())
_, out, _ = c.exec_command('tail -n 5 /root/clr_paper/scheduler.log')
print("Scheduler log:", out.read().decode().strip())

print("\nDone! Patched script deployed and scheduler restarted.")
c.close()
