import paramiko
import os
import stat

remote_host = '216.128.144.102'
remote_user = 'root'
remote_pass = '[8eE967Lg}!(GZoz'
remote_dir = '/root/clr_paper/experiments'
local_dir = r'C:\Users\Tanushree\Downloads\work\final_eacl_code'

os.makedirs(local_dir, exist_ok=True)

try:
    transport = paramiko.Transport((remote_host, 22))
    transport.connect(username=remote_user, password=remote_pass)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    # Get all files in experiments directory
    print(f"Fetching experiments directory from {remote_host}...")
    for item in sftp.listdir_attr(remote_dir):
        if stat.S_ISREG(item.st_mode) and item.filename.endswith('.py'):
            remote_path = f"{remote_dir}/{item.filename}"
            local_path = os.path.join(local_dir, item.filename)
            print(f"Downloading {item.filename}...")
            sftp.get(remote_path, local_path)
            
    # Also fetch the dynamic scheduler and aggregation scripts from root folder
    extra_files = ['dynamic_scheduler.py', 'watchdog.sh']
    for filename in extra_files:
        remote_path = f"/root/clr_paper/{filename}"
        local_path = os.path.join(local_dir, filename)
        try:
            print(f"Downloading {filename}...")
            sftp.get(remote_path, local_path)
        except Exception as e:
            print(f"Could not fetch {filename}: {e}")
            
    # Fetch the actual launch scripts that we just wrote
    try:
        sftp.get('/root/run_exp11_final.py', os.path.join(local_dir, 'run_exp11_final.py'))
        print("Downloading run_exp11_final.py...")
    except:
        pass

    print("Success! All final code files have been synchronized.")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'sftp' in locals():
        sftp.close()
    if 'transport' in locals():
        transport.close()
