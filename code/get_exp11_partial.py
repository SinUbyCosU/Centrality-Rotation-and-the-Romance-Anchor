import paramiko, json

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

for model in ["bloomz-7b1", "falcon3-7b-instruct"]:
    cmd = f"find /root/clr_paper/results/steering_vectors_{model}_* -name 'phase11_results.json' -exec cat {{}} \\;"
    stdin, stdout, stderr = client.exec_command(cmd)
    data = stdout.read().decode("utf-8", errors="replace")
    if data.strip():
        with open(f"exp11_{model}.json", "w", encoding="utf-8") as f:
            f.write(data)
        print(f"{model}: downloaded ({len(data)} bytes)")
        d = json.loads(data)
        print(f"  SCD-ASR correlation: r={d.get('scd_vs_asr',{}).get('pearson_r','N/A')}")
        for lang in ['en','hi','ar','sw','ja']:
            lang_data = d.get('languages',{}).get(lang,{})
            if lang_data:
                asrs = {k: v.get('asr',0) for k,v in lang_data.get('attack_types',{}).items()}
                print(f"  {lang}: {asrs}")

client.close()
