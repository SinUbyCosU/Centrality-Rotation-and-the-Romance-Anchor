"""Download and inspect the ORIGINAL phase11 results from Jul 12 (Table C)"""
import paramiko, json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

# The original Table C result (Jul 12 timestamp)
sftp.get(
    '/root/clr_paper/results/phase11_scaled_adversarial/phase11_results.json',
    'table_c_original.json'
)

# Check what raw data exists for the original
_, out, _ = c.exec_command(
    'ls -la /root/clr_paper/results/phase11_scaled_adversarial/'
)
print("Original phase11 dir contents:")
print(out.read().decode('utf-8', errors='replace'))

c.close()

# Analyze
with open('table_c_original.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("\nOriginal Table C keys:", list(data.keys()))
print()

if 'models' in data:
    for model_name, model_data in data['models'].items():
        if isinstance(model_data, dict):
            asr = model_data.get('overall_asr', model_data.get('asr', 'N/A'))
            print(f"  {model_name}: ASR={asr}")
            if 'per_language' in model_data:
                for lang, ld in model_data['per_language'].items():
                    if isinstance(ld, dict):
                        print(f"    {lang}: {ld}")
elif 'per_language' in data:
    print("Single-model format. Per-language ASR:")
    for lang, ld in data['per_language'].items():
        if isinstance(ld, dict):
            asr = ld.get('overall_asr', 'N/A')
            print(f"  {lang}: ASR={asr}")
else:
    # Just dump it
    print(json.dumps(data, indent=2, default=str)[:3000])
