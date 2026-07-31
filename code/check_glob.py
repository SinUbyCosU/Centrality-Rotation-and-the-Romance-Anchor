import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

code = '''import glob, os
search_path = os.path.join("/root/clr_paper/results", "steering_vectors_*", "phase11_scaled_adversarial", "phase11_raw_responses.jsonl")
print("Found files:", len(glob.glob(search_path)))
'''
_, out, _ = c.exec_command(f'python3 -c \'{code}\'')
print(out.read().decode().strip())
c.close()
