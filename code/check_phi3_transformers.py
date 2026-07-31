import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

code = """
from transformers import AutoConfig
config = AutoConfig.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)
print('rope_scaling:', getattr(config, 'rope_scaling', 'NOT FOUND'))
"""

_, out, _ = c.exec_command(f'python3 -c "{code}"')
print(out.read().decode().strip())
c.close()
