import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
    sftp = c.open_sftp()

    # Upload the judge script
    print('Uploading llm_judge_v2.py...')
    sftp.put('llm_judge_v2.py', '/root/clr_paper/llm_judge_v2.py')
    print('  Done.')

    # Count total responses to estimate time
    print('Counting total responses to judge...')
    _, out, _ = c.exec_command(
        'wc -l /root/clr_paper/results/*/phase11_scaled_adversarial/phase11_raw_responses.jsonl 2>/dev/null | tail -1'
    )
    count_out = out.read().decode('utf-8', errors='replace').strip()
    print(f'  Total response lines: {count_out}')

    # Check GPU 1 is still safe
    print('\nGPU status:')
    _, out, _ = c.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader')
    print(out.read().decode('utf-8', errors='replace').strip())

    # Start the judge on GPU 0 with nohup
    print('\nStarting LLM judge on GPU 0...')
    _, out, _ = c.exec_command(
        'nohup python3 /root/clr_paper/llm_judge_v2.py '
        '> /root/clr_paper/judge.log 2>&1 & echo "PID: $!"'
    )
    pid = out.read().decode('utf-8', errors='replace').strip()
    print(f'  {pid}')

    c.close()
    print('\nJudge started!')

except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc()
