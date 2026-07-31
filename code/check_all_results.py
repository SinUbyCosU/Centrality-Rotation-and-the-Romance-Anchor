import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Check for any LAS behavioral results
cmds = [
    'find /root/clr_paper/results -name "*.json" | xargs grep -l "las_steered\\|las_refusal\\|post_las_refusal\\|las_intervention\\|las_behavioral" 2>/dev/null',
    'ls /root/clr_paper/results/experiment14*',
    'ls /root/clr_paper/results/steering_vectors_*/phase14* 2>/dev/null',
    'find /root/clr_paper/results -name "phase11*" -o -name "phase10*" | head -20',
    'find /root/clr_paper/results -type d | sort | head -60',
]

for cmd in cmds:
    print(f"\n=== {cmd[:80]} ===")
    _, out, _ = c.exec_command(cmd)
    result = out.read().decode('utf-8', errors='replace')
    print(result if result.strip() else "(empty)")
