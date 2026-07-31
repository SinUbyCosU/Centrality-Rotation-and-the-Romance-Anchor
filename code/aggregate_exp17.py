import paramiko
import json
import numpy as np

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Get all phase17_results.json paths
stdin, stdout, stderr = client.exec_command("find /root/clr_paper/results -name phase17_results.json")
paths = stdout.read().decode().strip().split('\n')

weird_within_all = []
non_weird_within_all = []
cross_all = []

foundation_alignment = {
    "care_harm": [], "fairness_cheating": [],
    "loyalty_betrayal": [], "authority_subversion": [], "sanctity_degradation": []
}

for path in paths:
    if not path: continue
    s, o, e = client.exec_command(f"cat {path}")
    data = json.loads(o.read().decode())
    
    w = data.get("weird_clustering", {})
    if w.get("weird_within") is not None: weird_within_all.append(w["weird_within"])
    if w.get("non_weird_within") is not None: non_weird_within_all.append(w["non_weird_within"])
    if w.get("cross") is not None: cross_all.append(w["cross"])
    
    for f_name, f_data in data.get("foundations", {}).items():
        if f_data.get("alignment_with_overall") is not None:
            foundation_alignment[f_name].append(f_data["alignment_with_overall"])

print(f"Aggregated Results over {len(paths)} models:")
print(f"Mean WEIRD-within similarity: {np.mean(weird_within_all):.3f}")
print(f"Mean Non-WEIRD-within similarity: {np.mean(non_weird_within_all):.3f}")
print(f"Mean Cross-cluster similarity: {np.mean(cross_all):.3f}")

print("\nMean Alignment with Overall Safety Vector:")
for f_name, vals in foundation_alignment.items():
    print(f"  {f_name:25s}: {np.mean(vals):.3f}")

client.close()
