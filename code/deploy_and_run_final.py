import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

print("1. Uploading fixed 11_scaled_adversarial_v5.py...")
sftp.put('11_v5_patched.py', '/root/clr_paper/experiments/11_scaled_adversarial_v5.py')

print("2. Fixing model_loader.py for phi-3-mini...")
sftp.get('/root/clr_paper/models/model_loader.py', 'model_loader.py')
with open('model_loader.py', 'r') as f:
    loader_code = f.read()

# Add attn_implementation='eager' to kwargs to avoid the flash attention / rope scaling bug
if "kwargs['attn_implementation'] = 'eager'" not in loader_code:
    # Find the line: kwargs = {}
    loader_code = loader_code.replace(
        "kwargs = {}",
        "kwargs = {}\n    kwargs['attn_implementation'] = 'eager'"
    )
    with open('model_loader.py', 'w') as f:
        f.write(loader_code)
    sftp.put('model_loader.py', '/root/clr_paper/models/model_loader.py')
    print("   model_loader.py patched!")

print("3. Cleaning up invalid Phase 11 directories...")
# Delete all phase11_scaled_adversarial directories in results to force a clean re-run
c.exec_command('rm -rf /root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial')

print("4. Starting run_batch_gpu0.py in nohup...")
# Run the batch script on GPU 0
_, out, _ = c.exec_command(
    'nohup bash -c "'
    'export CUDA_VISIBLE_DEVICES=0 && '
    'python3 /root/clr_paper/run_batch_gpu0.py '
    '" > /root/clr_paper/batch_gpu0.log 2>&1 & echo "PID: $!"'
)
print("Batch PID:", out.read().decode().strip())

c.close()
print("Deployment and run started successfully!")
