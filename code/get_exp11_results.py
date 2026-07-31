import paramiko
import json
import os

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
    
    # Extract model name from path like /root/clr_paper/results/steering_vectors_mistral-7b_2026.../phase11...
    parts = file.split('steering_vectors_')[1]
    model = parts.rsplit('_2026', 1)[0]
    
    stdin, stdout, stderr = client.exec_command(f'cat {file}')
    try:
        data = json.loads(stdout.read().decode('utf-8'))
        scd_corr = data.get('scd_vs_asr', {})
        pr = scd_corr.get('pearson_r', 'N/A')
        pp = scd_corr.get('pearson_p', 'N/A')
        
        # Calculate average ASR across languages
        asrs = [ld.get('overall_asr', 0) for ld in data.get('languages', {}).values()]
        avg_asr = sum(asrs)/len(asrs) if asrs else 0
        
        en_asr = data.get('languages', {}).get('en', {}).get('overall_asr', 0)
        
        pr_str = f"{pr:.3f}" if isinstance(pr, float) else str(pr)
        pp_str = f"{pp:.3f}" if isinstance(pp, float) else str(pp)
        
        results.append((model, en_asr, avg_asr, pr_str, pp_str))
    except Exception as e:
        pass

client.close()

# Print markdown table
print("| Model | En ASR | Avg Multilingual ASR | Pearson r (SCD vs ASR) | p-value |")
print("|---|---|---|---|---|")
for model, en_asr, avg_asr, pr_str, pp_str in sorted(results):
    print(f"| {model} | {en_asr*100:.1f}% | {avg_asr*100:.1f}% | {pr_str} | {pp_str} |")
