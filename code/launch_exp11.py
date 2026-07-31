import paramiko

script = """
import os, glob

results_dir = '/root/clr_paper/results'
pkls = glob.glob(f'{results_dir}/steering_vectors_*/all_steering_vectors.pkl')

for pkl in sorted(pkls):
    d = os.path.dirname(pkl)
    base = os.path.basename(d)
    model = base.replace('steering_vectors_', '').rsplit('_2026', 1)[0]
    
    out_file = os.path.join(d, 'phase11_scaled_adversarial', 'phase11_results.json')
    if not os.path.exists(out_file):
        print(f"Queueing {model}...")
        cmd = f"TORCH_COMPILE_DISABLE=1 TRITON_CACHE_DIR=/tmp/triton_exp11_{model} python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key {model} --results-dir {results_dir} >> /root/clr_paper/exp11_final.log 2>&1"
        os.system(cmd)
"""

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

s = c.open_sftp()
with s.file('/root/run_exp11_final.py', 'w') as f:
    f.write(script)
s.close()

c.exec_command('nohup python3 /root/run_exp11_final.py > /dev/null 2>&1 &')
c.close()
