import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

sftp.get('/root/clr_paper/experiments/11_scaled_adversarial_v5.py', '11_remote.py')

with open('11_remote.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add monkeypatch right after the imports
patch = """
# Monkeypatch for older models (like Phi-3) expecting seen_tokens in newer transformers
try:
    from transformers import DynamicCache
    if not hasattr(DynamicCache, 'seen_tokens'):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
except ImportError:
    pass
"""

if "DynamicCache.seen_tokens = property" not in code:
    code = code.replace("import torch", f"import torch\n{patch}")
    with open('11_remote.py', 'w', encoding='utf-8') as f:
        f.write(code)
    sftp.put('11_remote.py', '/root/clr_paper/experiments/11_scaled_adversarial_v5.py')

# Clean up and restart
c.exec_command("rm -rf /root/clr_paper/results/steering_vectors_phi-3-mini-4k-instruct_*/phase11_scaled_adversarial")
_, out, _ = c.exec_command(
    'nohup bash -c "'
    'export CUDA_VISIBLE_DEVICES=0 && '
    'python3 /root/clr_paper/experiments/11_scaled_adversarial_v5.py --model-key phi-3-mini-4k-instruct'
    '" > /root/clr_paper/phi3_fix3.log 2>&1 & echo "PID: $!"'
)
print("Phi-3 PID:", out.read().decode().strip())
c.close()
