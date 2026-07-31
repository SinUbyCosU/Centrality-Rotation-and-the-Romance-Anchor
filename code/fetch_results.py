import paramiko
import re

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

_, out, _ = client.exec_command('grep -A 20 "Final Outgroup" /root/clr_paper/outgroup_fast_fixed.log')
text = out.read().decode('utf-8', errors='ignore')

# Strip out progress bars if they bleed in
clean_text = re.sub(r'[\u258f]+', '', text)
print(clean_text)

client.close()
