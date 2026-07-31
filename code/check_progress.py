import paramiko
import sys

sys.stdout.reconfigure(encoding='utf-8')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

print("1. Checking running processes...")
_, out, _ = c.exec_command("ps aux | grep -v grep | grep run_batch_gpu0.py || echo 'Batch runner not running'")
print(out.read().decode('utf-8', errors='replace').strip())
_, out, _ = c.exec_command("ps aux | grep -v grep | grep 11_scaled_adversarial_v5.py || echo 'Exp 11 not running'")
print(out.read().decode('utf-8', errors='replace').strip())

print("\n2. Checking batch log tail...")
_, out, _ = c.exec_command("tail -n 20 /root/clr_paper/batch_gpu0.log")
print(out.read().decode('utf-8', errors='replace').strip())

print("\n3. Checking completed models...")
_, out, _ = c.exec_command("ls -la /root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial/phase11_results.json | wc -l")
completed_count = out.read().decode('utf-8', errors='replace').strip()
print(f"Total completed models (with results.json): {completed_count}")

print("\n4. Checking models in progress or finished (raw jsonl exists):")
_, out, _ = c.exec_command("ls -lh /root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial/phase11_raw_responses.jsonl 2>/dev/null")
print(out.read().decode('utf-8', errors='replace').strip())

print("\n5. Checking errors...")
_, out, _ = c.exec_command("find /root/clr_paper/results -name 'LOAD_FAILED.txt'")
fails = out.read().decode('utf-8', errors='replace').strip()
if fails:
    print(f"Failed models:\n{fails}")
    for fail in fails.split('\n'):
        _, out2, _ = c.exec_command(f"cat {fail}")
        print(f"--- {fail} ---\n{out2.read().decode('utf-8', errors='replace').strip()}")
else:
    print("No LOAD_FAILED.txt found.")

c.close()
