import paramiko, time, sys

sys.stdout.reconfigure(encoding='utf-8')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

print("Checking batch_gpu0.log for startup...")
_, out, _ = c.exec_command('tail -n 20 /root/clr_paper/batch_gpu0.log')
print(out.read().decode('utf-8', errors='replace').strip())

print("\nWaiting for the first translation_attack prompts to be written...")
time.sleep(15)

_, out, _ = c.exec_command('find /root/clr_paper/results -name "phase11_raw_responses.jsonl"')
files = out.read().decode('utf-8', errors='replace').strip().split('\n')
for f in files:
    if f.strip():
        print(f"\nChecking raw data in {f}:")
        # Grep for translation attack and only print the prompt to avoid long response unicode errors
        _, out2, _ = c.exec_command(f'grep translation_attack {f} | head -n 5')
        lines = out2.read().decode('utf-8', errors='replace').strip().split('\n')
        for line in lines:
            if '"prompt":' in line:
                import json
                try:
                    obj = json.loads(line)
                    print(f"  [{obj.get('language')}] -> {repr(obj.get('prompt')[:100])}")
                except:
                    print(f"  {line[:150]}")

c.close()
