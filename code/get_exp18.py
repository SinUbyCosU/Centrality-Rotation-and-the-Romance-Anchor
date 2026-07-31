import paramiko, json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

remote_script = r"""
import glob, json

for f in glob.glob("/root/clr_paper/results/steering_vectors_*/phase18_las_behavioral/phase18_results.json"):
    try:
        model = f.split('steering_vectors_')[1].split('_2026')[0]
        with open(f) as fp:
            d = json.load(fp)
        agg = d.get('aggregate', {})
        print(f"\n=== {model} ===")
        print(f"Raw English Vector Lift: {agg.get('mean_raw_lift', 0)*100:+.1f}%")
        print(f"LAS-Rotated Vector Lift: {agg.get('mean_las_lift', 0)*100:+.1f}%")
        print(f"LAS Advantage (Net):     {agg.get('mean_las_advantage', 0)*100:+.1f}%")
        
        print("\nBreakdown by language (Baseline -> Raw English -> LAS Rotated):")
        langs = d.get('languages', {})
        for lang, ld in sorted(langs.items()):
            b = ld.get('conditions', {}).get('baseline', {}).get('refusal_rate', 0)
            r = ld.get('conditions', {}).get('raw_english', {}).get('refusal_rate', 0)
            l = ld.get('conditions', {}).get('las_rotated', {}).get('refusal_rate', 0)
            print(f"  {lang:>5}: {b*100:>5.1f}% -> {r*100:>5.1f}% -> {l*100:>5.1f}%")
            
    except Exception as e:
        print(f"Error on {f}: {e}")
"""

s = c.open_sftp()
with s.file('/root/get_exp18_results.py', 'w') as f:
    f.write(remote_script)
s.close()

_, out, _ = c.exec_command('python3 /root/get_exp18_results.py')
print(out.read().decode('utf-8'))
c.close()
