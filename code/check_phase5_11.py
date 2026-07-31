import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

cmds = [
    # Check phase5_las results
    'cat /root/clr_paper/results/steering_vectors_bloomz-7b1_20260622_092805/phase5_las/*.json 2>/dev/null | head -100',
    # Check how many models have phase5
    'ls -d /root/clr_paper/results/steering_vectors_*/phase5_las 2>/dev/null | wc -l',
    # Check phase11 scaled adversarial (has behavioral data)
    'ls -d /root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial 2>/dev/null | wc -l',
    # Check what phase11 contains
    'cat /root/clr_paper/results/steering_vectors_bloomz-7b1_20260622_092805/phase11_scaled_adversarial/phase11_results.json 2>/dev/null | head -80',
]

for cmd in cmds:
    print(f"\n=== {cmd[:80]} ===")
    _, out, _ = c.exec_command(cmd)
    result = out.read().decode('utf-8', errors='replace')
    print(result if result.strip() else "(empty)")
