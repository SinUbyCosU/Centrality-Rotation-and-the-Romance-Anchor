import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

code = """
import os
path = '/root/clr_paper/models/model_loader.py'
content = open(path).read()
if 'mistral-7b' not in content.split('EAGER_ATTN_MODELS')[1].split(']')[0]:
    content = content.replace('EAGER_ATTN_MODELS = [', 'EAGER_ATTN_MODELS = [\\n        "mistral-7b",')
    open(path, 'w').write(content)
    print('Patched successfully')
"""

_, stdout, _ = c.exec_command(f'python3 -c """{code}"""')
print(stdout.read().decode())
