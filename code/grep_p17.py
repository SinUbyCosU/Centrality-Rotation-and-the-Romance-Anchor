import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, o, _ = c.exec_command('grep -A 5 -B 5 "Already complete, skipping" /root/clr_paper/experiments/17_moral_foundations.py')
with open('p17_code.txt', 'w', encoding='utf-8') as f:
    f.write(o.read().decode())
