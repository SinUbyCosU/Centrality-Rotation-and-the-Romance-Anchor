import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

remote_script = r"""
import glob, json, numpy as np

results_by_model = {}
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase8_causal_intervention/phase8_results.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        model = d.get('model', f.split('steering_vectors_')[1].split('_2026')[0])
        langs = d.get('languages', {})
        
        for lang, ld in langs.items():
            if lang == 'en':
                continue
            ar = ld.get('alpha_results', {})
            baseline = ar.get('alpha_0.0', {}).get('refusal_rate', None)
            steered_1 = ar.get('alpha_1.0', {}).get('refusal_rate', None)
            steered_2 = ar.get('alpha_2.0', {}).get('refusal_rate', None)
            steered_4 = ar.get('alpha_4.0', {}).get('refusal_rate', None)
            scd = ld.get('scd_degrees', None)
            lift = ld.get('safety_lift', None)
            
            if model not in results_by_model:
                results_by_model[model] = []
            results_by_model[model].append({
                'lang': lang,
                'baseline': baseline,
                'alpha_1': steered_1,
                'alpha_2': steered_2,
                'alpha_4': steered_4,
                'scd': scd,
                'lift': lift
            })
    except Exception as e:
        pass

# Aggregate
all_baselines = []
all_steered = []
all_lifts = []
all_scds = []
lang_agg = {}

for model, entries in results_by_model.items():
    for e in entries:
        if e['baseline'] is not None and e['alpha_4'] is not None:
            all_baselines.append(e['baseline'])
            all_steered.append(e['alpha_4'])
            all_lifts.append(e['lift'] if e['lift'] is not None else e['alpha_4'] - e['baseline'])
            if e['scd'] is not None:
                all_scds.append(e['scd'])
            
            lang = e['lang']
            if lang not in lang_agg:
                lang_agg[lang] = {'baselines': [], 'steered': [], 'lifts': [], 'scds': []}
            lang_agg[lang]['baselines'].append(e['baseline'])
            lang_agg[lang]['steered'].append(e['alpha_4'])
            lang_agg[lang]['lifts'].append(e['lift'] if e['lift'] is not None else e['alpha_4'] - e['baseline'])
            if e['scd'] is not None:
                lang_agg[lang]['scds'].append(e['scd'])

print(f"=== OVERALL (n={len(all_baselines)} lang-model pairs, {len(results_by_model)} models) ===")
print(f"Mean Baseline Refusal Rate: {np.mean(all_baselines)*100:.1f}%")
print(f"Mean Steered (alpha=4) Refusal Rate: {np.mean(all_steered)*100:.1f}%")
print(f"Mean Safety Lift: {np.mean(all_lifts)*100:+.1f}%")
print()

print("=== PER LANGUAGE ===")
print(f"{'Lang':>6} | {'n':>3} | {'Baseline':>8} | {'Steered':>8} | {'Lift':>8} | {'SCD':>6}")
print("-" * 55)
for lang in sorted(lang_agg.keys()):
    la = lang_agg[lang]
    n = len(la['baselines'])
    b = np.mean(la['baselines']) * 100
    s = np.mean(la['steered']) * 100
    l = np.mean(la['lifts']) * 100
    scd = np.mean(la['scds']) if la['scds'] else 0
    print(f"{lang:>6} | {n:>3} | {b:>7.1f}% | {s:>7.1f}% | {l:>+7.1f}% | {scd:>5.1f}")

# Correlation
if len(all_scds) >= 3:
    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(all_scds[:len(all_lifts)], all_lifts[:len(all_scds)])
    rs, ps = spearmanr(all_scds[:len(all_lifts)], all_lifts[:len(all_scds)])
    print(f"\n=== SCD vs SAFETY LIFT CORRELATION ===")
    print(f"Pearson r={r:.3f}, p={p:.4f}")
    print(f"Spearman rho={rs:.3f}, p={ps:.4f}")
"""

s = c.open_sftp()
with s.file('/root/aggregate_exp8.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, err = c.exec_command('python3 /root/aggregate_exp8.py')
print(out.read().decode('utf-8', errors='replace'))
stderr = err.read().decode('utf-8', errors='replace')
if stderr.strip():
    print("STDERR:", stderr[:500])
