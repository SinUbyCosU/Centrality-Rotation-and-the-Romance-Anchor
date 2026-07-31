"""
Poll the judge log every 60s, and when done, SCP all judge results back locally.
Run this locally: python poll_and_scp.py
"""
import paramiko
import time
import os
import subprocess

HOST = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'
REMOTE_DIR = '/root/clr_paper/judge_results'
LOCAL_DEST = r'C:\Users\Tanushree\Downloads\work\judge_results'
LOG_PATH = '/root/clr_paper/judge.log'

os.makedirs(LOCAL_DEST, exist_ok=True)

def check_and_download():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS)
    sftp = c.open_sftp()

    # Tail the log
    _, out, _ = c.exec_command(f'tail -n 5 {LOG_PATH}')
    tail = out.read().decode('utf-8', errors='replace').strip()
    print(f'\n[Log tail]:\n{tail}')

    # Check if done
    _, out, _ = c.exec_command(f'grep "All results saved" {LOG_PATH}')
    done_line = out.read().decode('utf-8', errors='replace').strip()

    if done_line:
        print('\n[DONE] Judge finished! Downloading results...')
        # List all files in judge_results
        _, out, _ = c.exec_command(f'ls {REMOTE_DIR}/')
        files = out.read().decode('utf-8', errors='replace').strip().split('\n')
        for fname in files:
            fname = fname.strip()
            if not fname:
                continue
            remote_path = f'{REMOTE_DIR}/{fname}'
            local_path = os.path.join(LOCAL_DEST, fname)
            print(f'  Downloading {fname}...')
            try:
                sftp.get(remote_path, local_path)
                print(f'  -> Saved to {local_path}')
            except Exception as e:
                print(f'  [WARN] Could not download {fname}: {e}')
        c.close()
        return True

    c.close()
    return False


print(f'Polling judge status every 60s...')
print(f'Results will be saved to: {LOCAL_DEST}')
while True:
    try:
        done = check_and_download()
        if done:
            print('\nAll done! Check:', LOCAL_DEST)
            break
    except Exception as e:
        print(f'[WARN] Connection error: {e}')
    print('Waiting 60s...')
    time.sleep(60)
