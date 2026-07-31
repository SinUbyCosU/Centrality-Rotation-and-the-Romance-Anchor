import paramiko, json

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

# Check which models have scd_degrees=0 for ALL non-en languages
stdin, stdout, stderr = client.exec_command('find /root/clr_paper/results -name "phase11_results.json" 2>/dev/null')
files = stdout.read().decode('utf-8').strip().split('\n')

for f in sorted(files):
    if not f.strip() or 'steering_vectors_' not in f: continue
    model = f.split('steering_vectors_')[1].rsplit('_2026', 1)[0]
    stdin, stdout, stderr = client.exec_command(f'cat {f}')
    data = json.loads(stdout.read().decode('utf-8'))
    langs = data.get('languages', {})
    scds = {l: ld.get('scd_degrees', -1) for l, ld in langs.items()}
    non_en_scds = {l: v for l, v in scds.items() if l != 'en'}
    all_zero = all(v == 0 for v in non_en_scds.values())
    print(f"{model}: all_zero={all_zero}, en_scd={scds.get('en', '?')}, sample_hi_scd={scds.get('hi', '?')}")

client.close()
