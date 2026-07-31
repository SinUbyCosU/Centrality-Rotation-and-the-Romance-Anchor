import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

code = """
import transformers
print(transformers.__version__)
try:
    from transformers import AutoModelForCausalLM
    AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=False)
    print("SUCCESS Native load")
except Exception as e:
    print("ERROR:", e)
"""
_, out, _ = c.exec_command(f'python3 -c "{code}"')
print(out.read().decode('utf-8').strip())
c.close()
