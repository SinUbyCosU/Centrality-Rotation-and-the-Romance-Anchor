import paramiko
import os

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()
    
    # 1. Upload the fixed experiment script
    print('Uploading fixed 11_scaled_adversarial_v5.py...')
    sftp.put('11_v5_patched.py', '/root/clr_paper/experiments/11_scaled_adversarial_v5.py')
    print('  Done.')
    
    # 2. Upload the fixed batch runner
    print('Uploading fixed run_batch_gpu0.py...')
    sftp.put('run_batch_gpu0_fixed.py', '/root/clr_paper/run_batch_gpu0.py')
    print('  Done.')
    
    # 3. Clear out any broken phase11 directories (partial runs without phase11_results.json)
    print('Clearing partial/failed phase11 directories...')
    stdin, stdout, stderr = c.exec_command(
        'find /root/clr_paper/results -type d -name "phase11_scaled_adversarial" | while read d; do '
        '  if [ ! -f "$d/phase11_results.json" ]; then '
        '    echo "Removing partial: $d"; rm -rf "$d"; '
        '  else '
        '    echo "Keeping complete: $d"; '
        '  fi; '
        'done'
    )
    stdout.channel.recv_exit_status()
    print(stdout.read().decode('utf-8', errors='replace'))
    
    # 4. Kill any stale GPU0 python processes (but NOT the GPU1 extract_structured_local.py with PID 1135505)
    print('Checking for stale GPU0 python jobs...')
    stdin, stdout, stderr = c.exec_command('CUDA_VISIBLE_DEVICES=0 python3 -c "import torch; print(torch.cuda.device_count())"')
    out = stdout.read().decode('utf-8', errors='replace').strip()
    print(f'  GPU0 torch devices visible: {out}')
    
    # 5. Start the batch with proper nohup + disown so it survives SSH disconnect
    print('Starting batch on GPU 0 (nohup + disown)...')
    stdin, stdout, stderr = c.exec_command(
        'nohup python3 /root/clr_paper/run_batch_gpu0.py '
        '> /root/clr_paper/batch_gpu0.log 2>&1 & '
        'echo "Batch PID: $!"'
    )
    pid_out = stdout.read().decode('utf-8', errors='replace').strip()
    print(f'  {pid_out}')
    
    c.close()
    print('\nAll done! Batch is running.')
    
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
