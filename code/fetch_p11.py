import paramiko
import json
import time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

for model in ['mistral-7b', 'phi-4-mini-instruct', 'tinyllama']:
    c.exec_command(f'cat /root/clr_paper/results/steering_vectors_{model}_*/phase11_scaled_adversarial/phase11_results.json > /root/clr_paper/temp_p11_{model}.json')

time.sleep(2)

for model in ['mistral-7b', 'phi-4-mini-instruct', 'tinyllama']:
    try:
        sftp.get(f'/root/clr_paper/temp_p11_{model}.json', f'temp_p11_{model}.json')
        with open(f'temp_p11_{model}.json') as f:
            d = json.load(f)
            
        print(f'\nModel: {model}')
        print(f'Pearson r: {d.get("scd_vs_asr", {}).get("pearson_r", "N/A")}')
        print(f'Spearman r: {d.get("scd_vs_asr", {}).get("spearman_rho", "N/A")}')
        
        # Calculate overall ASR
        total_complied = 0
        total_total = 0
        for lang, ldata in d.get("languages", {}).items():
            for atk, atk_data in ldata.get("attacks", {}).items():
                total_complied += atk_data.get("complied", 0)
                total_total += atk_data.get("total", 0)
        
        if total_total > 0:
            print(f'Overall ASR: {total_complied / total_total * 100:.2f}%')
        else:
            print('Overall ASR: N/A')
            
    except Exception as ex:
        print(f'Failed to parse {model}: {ex}')

sftp.close()
c.close()
