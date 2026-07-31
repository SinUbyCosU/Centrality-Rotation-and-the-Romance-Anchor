import paramiko, time, sys

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

print("Checking batch_gpu0.log for startup...")
for _ in range(5):
    _, out, _ = c.exec_command('tail -n 15 /root/clr_paper/batch_gpu0.log')
    print(out.read().decode().strip())
    print("-" * 50)
    time.sleep(2)

print("Let's wait for the first translation_attack prompts to be written...")
time.sleep(10)
_, out, _ = c.exec_command('find /root/clr_paper/results -name "phase11_raw_responses.jsonl"')
files = out.read().decode().strip().split('\n')
for f in files:
    if f.strip():
        print(f"\nChecking raw data in {f}:")
        _, out2, _ = c.exec_command(f'grep translation_attack {f} | head -n 3')
        print(out2.read().decode().strip())

c.close()
