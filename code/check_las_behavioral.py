import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

remote_script = r"""
import glob, json, numpy as np

# Check Phase 5 LAS keys thoroughly
print("=== PHASE 5: LAS - FULL KEYS ===")
f = sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase5_las/*.json"))[0]
with open(f) as fp:
    d = json.load(fp)
print(f"Keys: {list(d.keys())}")
for k, v in d.items():
    if isinstance(v, dict):
        for k2, v2 in list(v.items())[:2]:
            print(f"  {k}.{k2}: {list(v2.keys()) if isinstance(v2, dict) else type(v2).__name__}")

# Check if there's ANY result file with "las" AND "refusal" or "behavioral"
print("\n=== SEARCHING FOR LAS+BEHAVIORAL ===")
import os
for root, dirs, files in os.walk("/root/clr_paper/results"):
    for f in files:
        if f.endswith('.json'):
            path = os.path.join(root, f)
            try:
                with open(path) as fp:
                    content = fp.read()
                if 'las' in content.lower() and ('refusal' in content.lower() or 'behavioral' in content.lower()):
                    print(f"FOUND: {path}")
            except:
                pass

# Phase 6 has safety prediction — does it correlate SCD with BEHAVIORAL jailbreak rates?
print("\n=== PHASE 6: Safety Prediction Detail ===")
f = sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase6_safety_prediction/phase6_results.json"))[0]
with open(f) as fp:
    d = json.load(fp)
print(json.dumps(d, indent=2, default=str)[:1500])
"""

s = c.open_sftp()
with s.file('/root/check_las_behavioral.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, err = c.exec_command('python3 /root/check_las_behavioral.py')
print(out.read().decode('utf-8', errors='replace'))
