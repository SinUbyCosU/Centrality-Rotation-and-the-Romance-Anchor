import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

print("1. Fetching model_loader.py...")
sftp.get('/root/clr_paper/models/model_loader.py', 'model_loader_remote.py')

with open('model_loader_remote.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Add the config patching code right before the AutoModelForCausalLM initialization
patch = """
        if 'phi-3' in model_id.lower() or 'phi3' in model_id.lower():
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            if hasattr(config, 'rope_scaling') and isinstance(config.rope_scaling, dict):
                if 'rope_type' in config.rope_scaling and 'type' not in config.rope_scaling:
                    config.rope_scaling['type'] = config.rope_scaling['rope_type']
            kwargs['config'] = config
"""

if "kwargs['config'] = config" not in code:
    code = code.replace(
        "if model_key in self.SEQ2SEQ_MODELS:",
        patch + "\n        if model_key in self.SEQ2SEQ_MODELS:"
    )
    with open('model_loader_remote.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("2. Uploading patched model_loader.py...")
    sftp.put('model_loader_remote.py', '/root/clr_paper/models/model_loader.py')
else:
    print("2. Already patched.")

print("3. Starting Phi-3-mini-4k-instruct generation in nohup...")
# Clean up any failed dir
c.exec_command("rm -rf /root/clr_paper/results/steering_vectors_phi-3-mini-4k-instruct_*/phase11_scaled_adversarial")

# Start just the phi-3 run
_, out, _ = c.exec_command(
    'nohup bash -c "'
    'export CUDA_VISIBLE_DEVICES=0 && '
    'python3 /root/clr_paper/experiments/11_scaled_adversarial_v5.py --model-key phi-3-mini-4k-instruct'
    '" > /root/clr_paper/phi3_fix.log 2>&1 & echo "PID: $!"'
)
print("Phi-3 generation PID:", out.read().decode().strip())

c.close()
