import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')
sftp = c.open_sftp()

print("Uploading llm_judge_v3.py...")
sftp.put('llm_judge_v3.py', '/root/clr_paper/llm_judge_v3.py')

print("Running validation test...")
_, out, err = c.exec_command('export CUDA_VISIBLE_DEVICES=0 && python3 /root/clr_paper/llm_judge_v3.py --test-only')

# We wait for the test to complete and print its output
for line in iter(out.readline, ""):
    print(line, end="")

for line in iter(err.readline, ""):
    print(line, end="")

c.close()
