import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

remote_script = r"""
import glob, json, numpy as np

# Check Phase 6 results more carefully
print("=== PHASE 6: SAFETY PREDICTION (full keys) ===")
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase6_safety_prediction/phase6_results.json"))[:3]:
    try:
        with open(f) as fp:
            d = json.load(fp)
        print(f"\nFile: {f}")
        print(f"Keys: {list(d.keys())}")
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
            elif isinstance(v, dict):
                print(f"  {k}: {list(v.keys())[:5]}...")
    except Exception as e:
        print(f"  Error: {e}")

# Check Phase 9: Cross-Model Transfer
print("\n=== PHASE 9: CROSS-MODEL TRANSFER ===")
for f in sorted(glob.glob("/root/clr_paper/results/phase9_cross_model_transfer/*.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        print(f"\nFile: {f.split('/')[-1]}")
        print(f"Keys: {list(d.keys())[:10]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check Phase 12: Statistical Rigor
print("\n=== PHASE 12: STATISTICAL RIGOR ===")
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase12*/*.json"))[:3]:
    try:
        with open(f) as fp:
            d = json.load(fp)
        print(f"\nFile: {f}")
        print(f"Keys: {list(d.keys())[:10]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check Phase 13: CKA Transfer
print("\n=== PHASE 13: CKA ===")
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase13*/*.json"))[:3]:
    try:
        with open(f) as fp:
            d = json.load(fp)
        print(f"\nFile: {f.split('phase13')[1]}")
        print(f"Keys: {list(d.keys())[:10]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check Phase 15: Dual Process
print("\n=== PHASE 15: DUAL PROCESS ===")
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase15*/*.json"))[:3]:
    try:
        with open(f) as fp:
            d = json.load(fp)
        print(f"\nFile: {f.split('phase15')[1]}")
        print(f"Keys: {list(d.keys())[:10]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check Phase 16: Linguistic Security
print("\n=== PHASE 16: LINGUISTIC SECURITY ===")
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase16*/*.json"))[:3]:
    try:
        with open(f) as fp:
            d = json.load(fp)
        print(f"\nFile: {f.split('phase16')[1]}")
        print(f"Keys: {list(d.keys())[:10]}")
    except Exception as e:
        print(f"  Error: {e}")

# Comprehensive: what directories exist?
print("\n=== ALL PHASE DIRECTORIES (counts) ===")
import os
phase_counts = {}
for d in glob.glob("/root/clr_paper/results/steering_vectors_*/phase*"):
    phase = os.path.basename(d)
    phase_counts[phase] = phase_counts.get(phase, 0) + 1
for k in sorted(phase_counts.keys()):
    print(f"  {k}: {phase_counts[k]} models")
"""

s = c.open_sftp()
with s.file('/root/aggregate_all2.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, err = c.exec_command('python3 /root/aggregate_all2.py')
print(out.read().decode('utf-8', errors='replace'))
stderr = err.read().decode('utf-8', errors='replace')
if stderr.strip():
    print("STDERR:", stderr[:300])
