import paramiko

script = """
import os, json, glob

results_dir = '/root/clr_paper/results'
exp18_files = glob.glob(os.path.join(results_dir, 'steering_vectors_*', 'phase18_las_behavioral', 'phase18_results.json'))

print(f'Found {len(exp18_files)} completed models.\\n')

print(f'| {"Model":<30} | {"Base Refusal":<15} | {"Raw Eng Refusal":<17} | {"LAS Refusal":<15} | {"LAS Sim Increase":<16} |')
print('|' + '-'*32 + '|' + '-'*17 + '|' + '-'*19 + '|' + '-'*17 + '|' + '-'*18 + '|')

for file in exp18_files:
    try:
        with open(file, 'r') as f:
            data = json.load(f)
        
        model = file.split('steering_vectors_')[1].split('_202')[0]
        
        total_base_refusal = 0
        total_raw_refusal = 0
        total_las_refusal = 0
        total_prompts = 0
        pre_sim_sum = 0
        post_sim_sum = 0
        lang_count = 0
        
        langs_dict = data.get('lang_results', data)
        
        for lang, res in langs_dict.items():
            if lang in ['aggregate', 'model', 'en', 'timestamp', 'n_prompts', 'best_layer', 'lang_results']: continue
            if not isinstance(res, dict) or 'conditions' not in res: continue
            
            base_rate = res['conditions']['baseline']['refusal_rate']
            raw_rate = res['conditions'].get('raw_english', {}).get('refusal_rate', 0)
            las_rate = res['conditions'].get('las_rotated', {}).get('refusal_rate', 0)
            
            total_base_refusal += base_rate
            total_raw_refusal += raw_rate
            total_las_refusal += las_rate
            total_prompts += 1
            
            pre_sim_sum += res.get('las_pre_sim', 0)
            post_sim_sum += res.get('las_post_sim', 0)
            lang_count += 1
            
        if total_prompts == 0: continue
            
        avg_base = total_base_refusal / total_prompts * 100
        avg_raw = total_raw_refusal / total_prompts * 100
        avg_las = total_las_refusal / total_prompts * 100
        
        avg_pre = pre_sim_sum / lang_count if lang_count > 0 else 0
        avg_post = post_sim_sum / lang_count if lang_count > 0 else 0
        sim_increase = avg_post - avg_pre
        
        print(f'| {model:<30} | {avg_base:>13.1f}% | {avg_raw:>15.1f}% | {avg_las:>13.1f}% | {sim_increase:>15.3f} |')
    except Exception as e:
        print(f'Error reading {file}: {e}')
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

s = c.open_sftp()
with s.file('/root/agg3.py', 'w') as f:
    f.write(script)
s.close()

_, out, _ = c.exec_command('python3 /root/agg3.py')
print(out.read().decode())
c.close()
