import paramiko, json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, _ = c.exec_command('python3 -c "import json; d=json.load(open(\'/root/clr_paper/results/unified_results.json\')); print(json.dumps(list(d.keys()))); [print(k, type(d[k]).__name__, len(str(d[k]))[:6]) for k in d.keys()]"')
print(out.read().decode('utf-8', errors='replace'))

# Also get a sample
_, out, _ = c.exec_command('head -200 /root/clr_paper/results/unified_results.json')
print(out.read().decode('utf-8', errors='replace'))
