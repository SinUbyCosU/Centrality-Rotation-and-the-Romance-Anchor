import paramiko

remote_script = """
import glob, json, numpy as np

weird_w = []
non_weird_w = []
cross = []
align = {'care_harm': [], 'fairness_cheating': [], 'loyalty_betrayal': [], 'authority_subversion': [], 'sanctity_degradation': []}

for f in glob.glob("/root/clr_paper/results/steering_vectors_*/phase17_moral_foundations/phase17_results.json"):
    try:
        with open(f, 'r') as fp:
            d = json.load(fp)
            wc = d.get('weird_clustering', {})
            if wc.get('weird_within') is not None: weird_w.append(wc['weird_within'])
            if wc.get('non_weird_within') is not None: non_weird_w.append(wc['non_weird_within'])
            if wc.get('cross') is not None: cross.append(wc['cross'])
            
            founds = d.get('foundations', {})
            for k in align:
                if k in founds and founds[k].get('alignment_with_overall') is not None:
                    align[k].append(founds[k]['alignment_with_overall'])
    except Exception as e:
        pass

print("--- CLUSTERING ---")
print(f"WEIRD-WEIRD: {np.mean(weird_w):.3f} (n={len(weird_w)})")
print(f"NonWEIRD-NonWEIRD: {np.mean(non_weird_w):.3f}")
print(f"Cross (W-NW): {np.mean(cross):.3f}")

print("--- ALIGNMENT WITH OVERALL SAFETY ---")
for k in align:
    print(f"{k}: {np.mean(align[k]):.3f}")
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

s = c.open_sftp()
with s.file('/root/aggregate17.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, _ = c.exec_command('python3 /root/aggregate17.py')
print(out.read().decode('utf-8'))
