import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, out, err = c.exec_command('export CUDA_VISIBLE_DEVICES=0 && python3 /root/clr_paper/llm_judge_v3.py --test-only')
for line in out:
    print(line, end="")
for line in err:
    print(line, end="")

c.close()
