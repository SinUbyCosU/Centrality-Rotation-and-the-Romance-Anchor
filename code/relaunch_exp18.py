import paramiko, json, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Delete errored phase18 results (those with 0 valid data)
remote_cleanup = r"""
import glob, json, os

deleted = 0
kept = 0
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase18_las_behavioral/phase18_results.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        langs = d.get('languages', {})
        has_valid = False
        for lang, ld in langs.items():
            baseline = ld.get('conditions', {}).get('baseline', {})
            if baseline.get('total', 0) > 0:
                has_valid = True
                break
        if not has_valid:
            os.remove(f)
            os.rmdir(os.path.dirname(f))
            deleted += 1
            print(f"Deleted: {f}")
        else:
            kept += 1
            print(f"Kept: {d.get('model', '?')}")
    except Exception as e:
        print(f"Error: {e}")

print(f"\nKept: {kept}, Deleted: {deleted}")
"""

s = c.open_sftp()
with s.file('/root/cleanup_exp18.py', 'w') as f:
    f.write(remote_cleanup)
s.close()

_, out, _ = c.exec_command('python3 /root/cleanup_exp18.py')
print(out.read().decode())

# Upload fixed script
s = c.open_sftp()
s.put('C:\\Users\\Tanushree\\Downloads\\work\\experiments\\18_las_behavioral.py',
      '/root/clr_paper/experiments/18_las_behavioral.py')
s.close()
print("Uploaded fixed 18_las_behavioral.py")

# Kill old queue script
c.exec_command('pkill -f queue_exp11')

# Re-launch Exp 18
cmd = 'nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key all --n-prompts 30 > /root/clr_paper/exp18_las_v2.log 2>&1 &'
c.exec_command(cmd)
print("Relaunched Exp 18 with device fix")

# Re-queue Exp 11 after Exp 18
queue_script = '''#!/bin/bash
echo "Waiting for Exp 18 v2 to finish..."
while pgrep -f "18_las_behavioral" > /dev/null; do
    sleep 60
done
echo "Exp 18 done. Starting Exp 11..."
python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1
echo "Exp 11 complete."
'''
s = c.open_sftp()
with s.file('/root/clr_paper/queue_exp11_v2.sh', 'w') as f:
    f.write(queue_script)
s.close()
c.exec_command('chmod +x /root/clr_paper/queue_exp11_v2.sh')
c.exec_command('nohup bash /root/clr_paper/queue_exp11_v2.sh > /root/clr_paper/queue_v2.log 2>&1 &')
print("Re-queued Exp 11 after Exp 18")

# Verify
time.sleep(3)
_, out, _ = c.exec_command('ps aux | grep -E "18_las|queue_exp" | grep -v grep')
print("\nRunning:")
print(out.read().decode())

c.close()
