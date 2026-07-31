import paramiko
import json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

sftp = c.open_sftp()
sftp.get('/root/clr_paper/results/steering_vectors_tinyllama_20260622_080426/phase11_scaled_adversarial/phase11_raw_responses_llm_judged.jsonl', 'tinyllama_judged.jsonl')
sftp.close()
c.close()

complied = 0
refused = 0
for line in open('tinyllama_judged.jsonl', encoding='utf-8'):
    d = json.loads(line)
    if d.get('is_safe', False):
        refused += 1
    else:
        complied += 1

print('TOTAL:', complied + refused)
print('COMPLIED:', complied)
print('REFUSED (SAFE):', refused)
