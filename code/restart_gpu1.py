import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    print('Checking extracted.jsonl line count...')
    _, out, _ = c.exec_command('wc -l /root/clr_paper/data/processed/extracted.jsonl')
    print(out.read().decode('utf-8', errors='replace').strip())
    
    print('\nRestarting extract_structured_local.py on GPU 1...')
    # Run in nohup on GPU 1
    _, out, _ = c.exec_command(
        'cd /root/clr_paper && nohup bash -c "'
        'export CUDA_VISIBLE_DEVICES=1 && '
        'python3 src/extract_structured_local.py '
        '--input data/raw/factchecks.csv '
        '--output data/processed/extracted.jsonl '
        '--model_name Qwen/Qwen2.5-7B-Instruct '
        '--batch_size 4 --load_in_4bit" > extract.log 2>&1 & echo "PID: $!"'
    )
    print(out.read().decode('utf-8', errors='replace').strip())
    c.close()
    
except Exception as e:
    print('Error:', e)
