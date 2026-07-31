import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

scheduler_code = r"""import os, glob, subprocess, time

ALL_MODELS = [
    "mistral-7b", "mistral-7b-instruct-v0.3", "openhermes-2.5-mistral-7b", "zephyr-7b",
    "qwen-7b", "qwen2-7b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct",
    "qwen3-4b-instruct", "qwen3-8b-instruct", "smollm3-3b",
    "phi-4-mini-instruct",
    "stablelm-3b", "stablelm-2-1.6b-chat",
    "yi-6b", "yi-1.5-6b-chat",
    "tinyllama",
    "bloomz-7b1",
    "phi-3-mini-4k-instruct",
    "falcon3-7b-instruct"
]

LOG_FILE = "/root/clr_paper/scheduler.log"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")
    print(msg)

def get_free_memory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
    return [int(x) for x in out.decode().strip().split('\n')]

def get_completed():
    completed = []
    for f in glob.glob("/root/clr_paper/results/steering_vectors_*/phase18_las_behavioral/phase18_results.json"):
        completed.append(f.split("steering_vectors_")[1].split("_2026")[0])
    return completed

def main():
    log("Scheduler started")
    while True:
        completed = get_completed()
        remaining = [m for m in ALL_MODELS if m not in completed]
        
        try:
            pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', '18_las_behavioral.py']).decode()
            running = []
            for line in pgrep_out.split('\n'):
                if '--model-key' in line:
                    parts = line.split('--model-key')
                    if len(parts) > 1:
                        running.extend(parts[1].strip().split()[0].split(','))
        except subprocess.CalledProcessError:
            running = []
            
        queue = [m for m in remaining if m not in running]
        
        if not remaining and not running:
            log("All models complete!")
            if os.system("pgrep -f 11_scaled_adversarial > /dev/null") != 0:
                os.system("nohup python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1 &")
            break
            
        if not queue:
            time.sleep(30)
            continue
            
        free_mems = get_free_memory()
        dispatched = False
        
        # We require 7500MB free. This allows fitting a 5-6GB model with 1.5-2.5GB margin,
        # and prevents OOM on the rare models that take up to 8.4GB.
        for gpu_id, free_mem in enumerate(free_mems):
            if free_mem > 7500:
                model = queue.pop(0)
                cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key {model} --n-prompts 30 > /root/clr_paper/exp18_{model}.log 2>&1 &"
                log(f"GPU {gpu_id} has {free_mem}MB free. Dispatching {model}.")
                os.system(cmd)
                dispatched = True
                break
                
        if dispatched:
            log("Waiting 60 seconds for model to load into memory...")
            time.sleep(60)
        else:
            time.sleep(30)

if __name__ == '__main__':
    main()
"""

s = c.open_sftp()
with s.file('/root/clr_paper/dynamic_scheduler.py', 'w') as f:
    f.write(scheduler_code)

# Update watchdog
watchdog = '''#!/bin/bash
LOG="/root/clr_paper/watchdog_scheduler.log"
echo "$(date): Watchdog started" >> $LOG

while true; do
    if ! pgrep -f "dynamic_scheduler.py" > /dev/null; then
        echo "$(date): Scheduler down, restarting..." >> $LOG
        nohup python3 /root/clr_paper/dynamic_scheduler.py > /dev/null 2>&1 &
    fi
    sleep 60
done
'''

with s.file('/root/clr_paper/watchdog.sh', 'w') as f:
    f.write(watchdog)

# Kill old
c.exec_command("pkill -f '18_las_behavioral.py'")
c.exec_command("pkill -f 'watchdog.sh'")
c.exec_command("pkill -f 'dynamic_scheduler.py'")

# Start new watchdog
c.exec_command("nohup bash /root/clr_paper/watchdog.sh > /dev/null 2>&1 &")

print("Deployed dynamic scheduler successfully.")
c.close()
