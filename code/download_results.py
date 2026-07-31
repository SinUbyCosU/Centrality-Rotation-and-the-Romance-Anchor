"""Download results from remote — resilient version.
Skips large binary files (.pkl, .npy) and only grabs analysis outputs + figures.
Auto-reconnects on connection drops.
"""
import paramiko, os, sys, stat, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"
REMOTE_DIR = "/root/clr_paper/results"
LOCAL_DIR = r"C:\Users\Tanushree\Downloads\work\results"

# Only download these extensions (skip .pkl, .npy which are huge)
KEEP_EXTENSIONS = {".json", ".csv", ".pdf", ".png", ".txt", ".log"}
MAX_FILE_SIZE = 10_000_000  # 10MB max per file

def get_connection():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = client.open_sftp()
    return client, sftp

def download_dir(sftp, remote_path, local_path, count, skipped):
    os.makedirs(local_path, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_path)
    except Exception as e:
        print(f"  SKIP dir (error): {remote_path} — {e}")
        return
    
    for entry in entries:
        remote_file = f"{remote_path}/{entry.filename}"
        local_file = os.path.join(local_path, entry.filename)
        
        if stat.S_ISDIR(entry.st_mode):
            download_dir(sftp, remote_file, local_file, count, skipped)
        else:
            ext = os.path.splitext(entry.filename)[1].lower()
            if ext not in KEEP_EXTENSIONS:
                skipped[0] += 1
                continue
            if entry.st_size > MAX_FILE_SIZE:
                skipped[0] += 1
                continue
            try:
                sftp.get(remote_file, local_file)
                count[0] += 1
                if count[0] % 50 == 0:
                    print(f"  Downloaded {count[0]} files (skipped {skipped[0]})...")
            except Exception as e:
                print(f"  ERROR downloading {entry.filename}: {e}")

def main():
    # Clean old results
    if os.path.exists(LOCAL_DIR):
        import shutil
        print(f"Removing old local results...")
        shutil.rmtree(LOCAL_DIR, ignore_errors=True)
    
    # Get list of model dirs first
    client, sftp = get_connection()
    stdin, stdout, stderr = client.exec_command(f"ls -d {REMOTE_DIR}/steering_vectors_*/ {REMOTE_DIR}/figures_*/ 2>/dev/null")
    dirs = [d.strip() for d in stdout.read().decode().strip().split("\n") if d.strip()]
    print(f"Found {len(dirs)} result directories")
    sftp.close()
    client.close()
    
    count = [0]
    skipped = [0]
    
    # Download each directory with a fresh connection (prevents timeouts)
    for i, remote_subdir in enumerate(dirs):
        dir_name = os.path.basename(remote_subdir.rstrip("/"))
        local_subdir = os.path.join(LOCAL_DIR, dir_name)
        print(f"\n[{i+1}/{len(dirs)}] {dir_name}")
        
        for attempt in range(3):
            try:
                client, sftp = get_connection()
                download_dir(sftp, remote_subdir.rstrip("/"), local_subdir, count, skipped)
                sftp.close()
                client.close()
                break
            except Exception as e:
                print(f"  Attempt {attempt+1} failed: {e}")
                time.sleep(5)
                try:
                    sftp.close()
                    client.close()
                except:
                    pass
    
    # Also grab logs
    print(f"\nDownloading logs...")
    try:
        client, sftp = get_connection()
        for log in ["run_fixed.log", "retry_failed.log", "retry2.log"]:
            try:
                sftp.get(f"/root/clr_paper/{log}", os.path.join(LOCAL_DIR, log))
            except:
                pass
        sftp.close()
        client.close()
    except:
        pass
    
    print(f"\n{'='*50}")
    print(f"Done! Downloaded {count[0]} files, skipped {skipped[0]} binary files")
    print(f"Results saved to: {LOCAL_DIR}")

if __name__ == "__main__":
    main()
