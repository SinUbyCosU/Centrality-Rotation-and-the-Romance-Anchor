import paramiko
import json
import os

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    cmd = """
python3 -c "
import os, glob, json

models = [
    'mistral-7b', 'openhermes-2.5-mistral-7b', 'zephyr-7b', 'smollm3-3b', 
    'mistral-7b-instruct-v0.3', 'stablelm-2-1.6b-chat', 'qwen3-4b-instruct', 
    'phi-3-mini-4k-instruct', 'qwen2.5-3b-instruct', 'qwen2.5-7b-instruct', 
    'qwen2-7b-instruct', 'falcon3-7b-instruct', 'tinyllama', 'phi-4-mini-instruct'
]

results_dir = '/root/clr_paper/results'
print('================== FULL STATUS CHECK ==================')
for model in models:
    model_dirs = glob.glob(f'{results_dir}/steering_vectors_{model}_*')
    if not model_dirs:
        print(f'{model:30s} | NOT STARTED')
        continue
        
    target_dir = model_dirs[0]
    p11_dir = os.path.join(target_dir, 'phase11_scaled_adversarial')
    
    results_json = os.path.join(p11_dir, 'phase11_results.json')
    raw_responses = os.path.join(p11_dir, 'phase11_raw_responses.jsonl')
    judged_responses = os.path.join(p11_dir, 'phase11_raw_responses_llm_judged.jsonl')
    
    status = 'PENDING'
    if os.path.exists(results_json) and os.path.exists(raw_responses):
        status = 'GENERATED'
    elif os.path.exists(raw_responses):
        status = 'GENERATING NOW...'
        
    asr_str = ''
    if os.path.exists(judged_responses):
        status = 'JUDGED'
        try:
            complied = 0
            refused = 0
            with open(judged_responses, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    d = json.loads(line)
                    if d.get('llm_classification') == 'complied':
                        complied += 1
                    else:
                        refused += 1
            if complied + refused > 0:
                asr = (complied / (complied + refused)) * 100
                asr_str = f' | ASR: {asr:.1f}%'
        except:
            pass
            
    print(f'{model:30s} | {status:15s}{asr_str}')
print('=======================================================')
"
"""
    _, out, err = c.exec_command(cmd)
    res = out.read().decode('utf-8')
    err_res = err.read().decode('utf-8')
    if err_res:
        print("ERR:", err_res)
    print(res)
    c.close()
except Exception as e:
    print('SSH Error:', e)
