import paramiko
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

print("Killing hung processes...")
c.exec_command("pkill -9 -f '18_las_behavioral.py'")
c.exec_command("pkill -9 -f 'dynamic_scheduler.py'")
c.exec_command("pkill -9 -f 'compile_worker'")

print("Deleting partial result directories...")
c.exec_command("rm -rf /root/clr_paper/results/steering_vectors_mistral-7b_*/phase18_las_behavioral")
c.exec_command("rm -rf /root/clr_paper/results/steering_vectors_openhermes-2.5-mistral-7b_*/phase18_las_behavioral")
c.exec_command("rm -rf /root/clr_paper/results/steering_vectors_zephyr-7b_*/phase18_las_behavioral")
c.exec_command("rm -rf /root/clr_paper/results/steering_vectors_qwen-7b_*/phase18_las_behavioral")

print("Updating dynamic_scheduler.py with TRITON_CACHE_DIR...")
s = c.open_sftp()
remote_file = s.file('/root/clr_paper/dynamic_scheduler.py', 'r')
code = remote_file.read().decode()
remote_file.close()

# Replace the launch command
new_cmd = 'cmd = f"TORCH_COMPILE_DISABLE=1 TRITON_CACHE_DIR=/tmp/triton_{model} CUDA_VISIBLE_DEVICES={gpu_id} nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key {model} --n-prompts 30 > /root/clr_paper/exp18_{model}.log 2>&1 &"'
code = code.replace('cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key {model} --n-prompts 30 > /root/clr_paper/exp18_{model}.log 2>&1 &"', new_cmd)

remote_file = s.file('/root/clr_paper/dynamic_scheduler.py', 'w')
remote_file.write(code)
remote_file.close()
s.close()

print("Restarting watchdog and scheduler...")
c.exec_command("nohup python3 /root/clr_paper/dynamic_scheduler.py > /dev/null 2>&1 &")

print("Done!")
c.close()
