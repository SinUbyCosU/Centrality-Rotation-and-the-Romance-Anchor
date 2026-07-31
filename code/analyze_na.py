import paramiko
import json
import numpy as np
import warnings

# Suppress numpy RuntimeWarnings for 0 variance if they happen
warnings.filterwarnings('ignore')

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

stdin, stdout, stderr = client.exec_command('find /root/clr_paper/results -name "phase11_results.json" 2>/dev/null')
files = stdout.read().decode('utf-8').strip().split('\n')

results = []

for file in files:
    if not file or 'steering_vectors_' not in file: continue
    
    parts = file.split('steering_vectors_')[1]
    model = parts.rsplit('_2026', 1)[0]
    
    stdin, stdout, stderr = client.exec_command(f'cat {file}')
    try:
        data = json.loads(stdout.read().decode('utf-8'))
        scd_corr = data.get('scd_vs_asr', {})
        pr = scd_corr.get('pearson_r', 'N/A')
        
        is_na = pr == 'N/A' or pr is None or (isinstance(pr, float) and np.isnan(pr))
        
        if is_na:
            languages_data = data.get('languages', {})
            asrs = []
            scds = []
            for lang, ldata in languages_data.items():
                if 'overall_asr' in ldata and 'scd_degrees' in ldata:
                    asrs.append(ldata['overall_asr'])
                    scds.append(ldata['scd_degrees'])
                    
            n = len(asrs)
            if n > 0:
                # Use sample variance (ddof=1) to match standard statistical variance if preferred, 
                # but ddof=0 is fine just to check for zero.
                asr_var = np.var(asrs, ddof=0)
                scd_var = np.var(scds, ddof=0)
            else:
                asr_var = 0.0
                scd_var = 0.0
                
            cls = "unknown"
            if n < 3:
                cls = "insufficient-n"
            elif asr_var == 0.0:
                cls = "genuine-flat"
            elif scd_var == 0.0:
                cls = "code-issue"
            else:
                # sometimes scipy's pearsonr returns NaN if variance is extremely close to zero due to float precision
                if asr_var < 1e-10:
                    cls = "genuine-flat"
                else:
                    cls = "code-issue"
                
            results.append((model, n, asr_var, scd_var, cls))
    except Exception as e:
        results.append((model, 0, 0.0, 0.0, f"code-issue"))

client.close()

print("| Model | n | ASR Variance | SCD Variance | Classification |")
print("|---|---|---|---|---|")
for model, n, asr_v, scd_v, cls in sorted(results):
    print(f"| {model} | {n} | {asr_v:.6f} | {scd_v:.6f} | {cls} |")
