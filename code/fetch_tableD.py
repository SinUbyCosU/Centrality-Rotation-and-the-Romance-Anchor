import paramiko
import json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

code = """
import os, json, glob
import numpy as np
print('| Model | WEIRD Mean Alignment | Non-WEIRD Mean Alignment |')
print('|---|---|---|')
for p in sorted(glob.glob('/root/clr_paper/results/steering_vectors_*/phase17_moral_foundations/phase17_results.json')):
    model = p.split('steering_vectors_')[1].split('_2026')[0]
    if model in ['bloomz-7b1', 'stablelm-3b', 'yi-6b', 'mistral-7b', 'qwen-7b']: # exclude base models. Wait! Did the user say qwen-7b is valid? YES. So exclude only the first 4.
        pass
    if model in ['bloomz-7b1', 'stablelm-3b', 'yi-6b', 'mistral-7b']: continue

    with open(p) as f:
        d = json.load(f)
    
    langs = d.get('languages', {})
    weird_all = []
    nonweird_all = []
    
    for lang, ldata in langs.items():
        foundations = ldata.get('foundations', {})
        for k, f in foundations.items():
            if f['category'] == 'WEIRD':
                weird_all.append(f['alignment_with_overall'])
            elif f['category'] == 'Non-WEIRD':
                nonweird_all.append(f['alignment_with_overall'])
    
    w_mean = sum(weird_all)/len(weird_all) if weird_all else 0
    nw_mean = sum(nonweird_all)/len(nonweird_all) if nonweird_all else 0
    
    print(f'| {model} | {w_mean:+.3f} | {nw_mean:+.3f} |')
"""
_, stdout, _ = c.exec_command(f'python3 -c """{code}"""')
with open('tableD.md', 'w', encoding='utf-8') as f:
    f.write(stdout.read().decode('utf-8'))
