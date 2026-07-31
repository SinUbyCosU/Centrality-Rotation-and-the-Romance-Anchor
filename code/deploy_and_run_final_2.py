import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

print("1. Stopping old batch processes...")
c.exec_command("pkill -f run_batch_gpu0.py")
c.exec_command("pkill -f 11_scaled_adversarial_v5.py")

print("2. Uploading fully fixed 11_scaled_adversarial_v5.py...")
sftp.put('11_v5_patched.py', '/root/clr_paper/experiments/11_scaled_adversarial_v5.py')

print("3. Cleaning up invalid Phase 11 directories...")
c.exec_command('rm -rf /root/clr_paper/results/steering_vectors_*/phase11_scaled_adversarial')

print("4. Starting run_batch_gpu0.py in nohup...")
_, out, _ = c.exec_command(
    'nohup bash -c "'
    'export CUDA_VISIBLE_DEVICES=0 && '
    'python3 /root/clr_paper/run_batch_gpu0.py '
    '" > /root/clr_paper/batch_gpu0.log 2>&1 & echo "PID: $!"'
)
print("Batch PID:", out.read().decode().strip())

c.close()
print("Clean deployment started.")
