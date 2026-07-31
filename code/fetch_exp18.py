import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

remote_script = r"""
import glob, json, numpy as np

valid = 0
errored = 0
all_raw_lifts = []
all_las_lifts = []
all_las_adv = []
model_summaries = []

for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase18_las_behavioral/phase18_results.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        model = d.get('model', '?')
        agg = d.get('aggregate', {})
        langs = d.get('languages', {})
        
        # Check if any language has valid data (total > 0 for baseline)
        has_valid = False
        for lang, ld in langs.items():
            baseline = ld.get('conditions', {}).get('baseline', {})
            if baseline.get('total', 0) > 0:
                has_valid = True
                break
        
        if has_valid and agg.get('n_languages', 0) > 0:
            valid += 1
            raw_lift = agg.get('mean_raw_lift', 0)
            las_lift = agg.get('mean_las_lift', 0)
            las_adv = agg.get('mean_las_advantage', 0)
            all_raw_lifts.append(raw_lift)
            all_las_lifts.append(las_lift)
            all_las_adv.append(las_adv)
            
            # Per-lang detail for this model
            lang_detail = []
            for lang, ld in sorted(langs.items()):
                b = ld.get('conditions', {}).get('baseline', {}).get('refusal_rate', 0)
                r = ld.get('conditions', {}).get('raw_english', {}).get('refusal_rate', 0)
                l = ld.get('conditions', {}).get('las_rotated', {}).get('refusal_rate', 0)
                bt = ld.get('conditions', {}).get('baseline', {}).get('total', 0)
                lang_detail.append(f"{lang}: base={b*100:.0f}% raw={r*100:.0f}% las={l*100:.0f}% (n={bt})")
            
            model_summaries.append(f"{model}: raw_lift={raw_lift*100:+.1f}%, las_lift={las_lift*100:+.1f}%, las_adv={las_adv*100:+.1f}%")
            if valid <= 5:  # Print detail for first 5 models
                for ld in lang_detail:
                    print(f"    {ld}")
        else:
            errored += 1
            model_summaries.append(f"{model}: ALL ERRORED (device mismatch)")
    except Exception as e:
        errored += 1

print(f"=== EXP 18 SUMMARY ===")
print(f"Valid models: {valid}")
print(f"Errored models: {errored}")
print()
for s in model_summaries:
    print(f"  {s}")

if all_raw_lifts:
    print(f"\n=== AGGREGATE (valid models only) ===")
    print(f"Mean Raw English Lift:   {np.mean(all_raw_lifts)*100:+.1f}%")
    print(f"Mean LAS-Rotated Lift:   {np.mean(all_las_lifts)*100:+.1f}%")
    print(f"Mean LAS Advantage:      {np.mean(all_las_adv)*100:+.1f}%")
"""

s = c.open_sftp()
with s.file('/root/aggregate_exp18.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, err = c.exec_command('python3 /root/aggregate_exp18.py')
print(out.read().decode('utf-8', errors='replace'))
stderr = err.read().decode('utf-8', errors='replace')
if stderr.strip():
    print("STDERR:", stderr[:300])

c.close()
