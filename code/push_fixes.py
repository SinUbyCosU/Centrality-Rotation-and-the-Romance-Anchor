import paramiko
import os

remote_host = '216.128.144.102'
remote_user = 'root'
remote_pass = '[8eE967Lg}!(GZoz'

files_to_upload = [
    (r'C:\Users\Tanushree\Downloads\work\final_eacl_code\02_pivot_language_analysis.py', '/root/clr_paper/experiments/02_pivot_language_analysis.py'),
    (r'C:\Users\Tanushree\Downloads\work\final_eacl_code\06_07_safety_codemix.py', '/root/clr_paper/experiments/06_07_safety_codemix.py'),
    (r'C:\Users\Tanushree\Downloads\work\final_eacl_code\17_moral_foundations.py', '/root/clr_paper/experiments/17_moral_foundations.py'),
    (r'C:\Users\Tanushree\Downloads\work\fast_translations.json', '/root/clr_paper/fast_translations.json')
]

try:
    transport = paramiko.Transport((remote_host, 22))
    transport.connect(username=remote_user, password=remote_pass)
    sftp = paramiko.SFTPClient.from_transport(transport)
    
    print(f"Uploading files to {remote_host}...")
    for local_path, remote_path in files_to_upload:
        print(f"Uploading {local_path} -> {remote_path}")
        sftp.put(local_path, remote_path)
            
    print("Success! Files uploaded.")
    
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'sftp' in locals(): sftp.close()
    if 'transport' in locals(): transport.close()
