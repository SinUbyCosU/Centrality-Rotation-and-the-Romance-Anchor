import paramiko, json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, _ = c.exec_command('cat /root/clr_paper/results/causal_dose_response/bloomz-7b1_dose_response.json')
d = json.loads(out.read().decode())
print("BLOOMZ-7B1:")
for lang in d['languages']:
    if lang != 'en':
        print(f"  {lang} Lift: {d['languages'][lang]['safety_lift']:.3f}, SCD: {d['languages'][lang]['scd_degrees']:.1f}, Base Refusal: {d['languages'][lang]['baseline_refusal_rate']:.2f}")

print("\n")
_, out, _ = c.exec_command('cat /root/clr_paper/results/causal_dose_response/openhermes-2.5-mistral-7b_dose_response.json')
d = json.loads(out.read().decode())
print("OPENHERMES:")
for lang in d['languages']:
    if lang != 'en':
        print(f"  {lang} Lift: {d['languages'][lang]['safety_lift']:.3f}, SCD: {d['languages'][lang]['scd_degrees']:.1f}, Base Refusal: {d['languages'][lang]['baseline_refusal_rate']:.2f}")


