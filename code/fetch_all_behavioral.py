import paramiko, json, numpy as np

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

remote_script = r"""
import glob, json, numpy as np

# ========= PHASE 10: Adversarial Probing ASR (all 20 models) =========
print("=== PHASE 10: ADVERSARIAL PROBING (ASR by language) ===")
lang_asrs = {}
n_models_p10 = 0
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase10_adversarial_probing/phase10_results.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        n_models_p10 += 1
        langs = d.get('languages', {})
        for lang, ld in langs.items():
            asr = ld.get('overall_asr', None)
            scd = ld.get('scd_degrees', None)
            if asr is not None:
                if lang not in lang_asrs:
                    lang_asrs[lang] = {'asrs': [], 'scds': []}
                lang_asrs[lang]['asrs'].append(asr)
                if scd is not None:
                    lang_asrs[lang]['scds'].append(scd)
    except:
        pass

print(f"Models with Phase 10: {n_models_p10}")
print(f"{'Lang':>6} | {'Mean ASR':>8} | {'Mean SCD':>8} | n")
print("-" * 40)
all_asrs = []
all_scds = []
for lang in sorted(lang_asrs.keys()):
    la = lang_asrs[lang]
    m_asr = np.mean(la['asrs'])
    m_scd = np.mean(la['scds']) if la['scds'] else 0
    print(f"{lang:>6} | {m_asr*100:>7.1f}% | {m_scd:>7.1f} | {len(la['asrs'])}")
    if lang != 'en':
        all_asrs.extend(la['asrs'])
        all_scds.extend(la['scds'])

if len(all_asrs) >= 3 and len(all_scds) >= 3:
    from scipy.stats import pearsonr, spearmanr
    r, p = pearsonr(all_scds[:len(all_asrs)], all_asrs[:len(all_scds)])
    rs, ps = spearmanr(all_scds[:len(all_asrs)], all_asrs[:len(all_scds)])
    print(f"\nSCD vs ASR: Pearson r={r:.3f} p={p:.4f}, Spearman rho={rs:.3f} p={ps:.4f}")

# ========= PHASE 11: Scaled Adversarial =========
print("\n=== PHASE 11: SCALED ADVERSARIAL (5 models) ===")
n_models_p11 = 0
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial/phase11_results.json")):
    try:
        with open(f) as fp:
            d = json.load(fp)
        model = d.get('model', '?')
        n_models_p11 += 1
        print(f"  {model}: ", end="")
        langs = d.get('languages', {})
        for lang in ['en', 'hi', 'ar', 'sw']:
            asr = langs.get(lang, {}).get('overall_asr', None)
            if asr is not None:
                print(f"{lang}={asr*100:.0f}% ", end="")
        print()
    except:
        pass
print(f"Total Phase 11 models: {n_models_p11}")

# ========= PHASE 6: Safety Prediction =========
print("\n=== PHASE 6: SAFETY PREDICTION ===")
for f in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/phase6_safety_prediction/phase6_results.json"))[:3]:
    try:
        with open(f) as fp:
            d = json.load(fp)
        auc = d.get('logreg_auc', d.get('auc', None))
        r2 = d.get('linreg_r2', d.get('r2', None))
        print(f"  AUC={auc}, R2={r2}")
    except Exception as e:
        print(f"  Error: {e}")
"""

s = c.open_sftp()
with s.file('/root/aggregate_all.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, err = c.exec_command('python3 /root/aggregate_all.py')
print(out.read().decode('utf-8', errors='replace'))
stderr = err.read().decode('utf-8', errors='replace')
if stderr.strip():
    print("STDERR:", stderr[:300])
