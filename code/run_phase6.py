import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

def run():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=10)

    print("Finding all directories with phase11_results.json...")
    stdin, stdout, stderr = client.exec_command('find /root/clr_paper/results/steering_vectors_* -name "phase11_results.json" 2>/dev/null')
    files = stdout.read().decode('utf-8').strip().split('\n')
    
    dirs = [f.replace('/phase11_scaled_adversarial/phase11_results.json', '') for f in files if f.strip()]
    
    print(f"Found {len(dirs)} directories to process.")
    
    for i, d in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] Running 06_07_safety_codemix.py for: {d}")
        cmd = f"cd /root/clr_paper && source venv/bin/activate && python experiments/06_07_safety_codemix.py --phase both --input {d} --phase2 {d}"
        
        # Execute blocking
        stdin, stdout, stderr = client.exec_command(cmd)
        
        # Wait for the command to finish and get exit status
        exit_status = stdout.channel.recv_exit_status()
        
        if exit_status == 0:
            print("  -> Success")
        else:
            print(f"  -> Failed (Code {exit_status})")
            err_output = stderr.read().decode('utf-8')
            if err_output:
                print(f"     Error: {err_output[:200]}...")

    client.close()
    print("Done!")

if __name__ == "__main__":
    run()
