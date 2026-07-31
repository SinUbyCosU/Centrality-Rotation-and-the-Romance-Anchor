import paramiko, json, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

parallel_script = r"""
import os, glob, subprocess, json

# 1. Kill old Exp 18 process
os.system("pkill -f '18_las_behavioral.py'")
print("Killed old Exp 18.")

# 2. Find remaining models
ALL_MODELS = [
    "mistral-7b", "mistral-7b-instruct-v0.3", "openhermes-2.5-mistral-7b", "zephyr-7b",
    "qwen-7b", "qwen2-7b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct",
    "qwen3-4b-instruct", "qwen3-8b-instruct", "smollm3-3b",
    "phi-3-mini-4k-instruct", "phi-4-mini-instruct",
    "falcon3-7b-instruct",
    "stablelm-3b", "stablelm-2-1.6b-chat",
    "yi-6b", "yi-1.5-6b-chat",
    "tinyllama",
    "bloomz-7b1"
]

completed = []
for f in glob.glob("/root/clr_paper/results/steering_vectors_*/phase18_las_behavioral/phase18_results.json"):
    completed.append(f.split("steering_vectors_")[1].split("_2026")[0])

remaining = [m for m in ALL_MODELS if m not in completed]
print(f"Remaining models to run: {len(remaining)} -> {remaining}")

if len(remaining) == 0:
    print("All models complete!")
    os.system("nohup python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1 &")
    exit(0)

# 3. Split models into two lists for the two GPUs
half = len(remaining) // 2
models_gpu0 = remaining[:half]
models_gpu1 = remaining[half:]

# 4. Create runner scripts
def run_models(models, gpu_id):
    if not models: return
    model_str = ",".join(models)
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key {model_str} --n-prompts 30 > /root/clr_paper/exp18_gpu{gpu_id}.log 2>&1 &"
    os.system(cmd)
    print(f"Launched GPU {gpu_id} with models: {model_str}")

run_models(models_gpu0, 0)
run_models(models_gpu1, 1)

# Update watchdog to track both
watchdog = '''#!/bin/bash
LOG="/root/clr_paper/watchdog_parallel.log"
echo "$(date): Watchdog started" >> $LOG

while true; do
    sleep 300
    DONE=$(ls -d /root/clr_paper/results/steering_vectors_*/phase18_las_behavioral 2>/dev/null | wc -l)
    
    if [ "$DONE" -ge 20 ]; then
        echo "$(date): All 20 models complete! Exiting and starting Exp 11." >> $LOG
        if ! pgrep -f "11_scaled_adversarial" > /dev/null; then
            nohup python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1 &
        fi
        exit 0
    fi
    
    # Not enforcing strict auto-restart here since the python script now splits models.
    # We will just log progress.
    echo "$(date): $DONE/20 models complete. GPU 0 processes: $(pgrep -f exp18_gpu0 | wc -l), GPU 1 processes: $(pgrep -f exp18_gpu1 | wc -l)" >> $LOG
done
'''
with open("/root/clr_paper/watchdog.sh", "w") as f:
    f.write(watchdog)
os.system("pkill -f 'watchdog.sh'")
os.system("nohup bash /root/clr_paper/watchdog.sh > /dev/null 2>&1 &")
"""

s = c.open_sftp()
with s.file('/root/run_parallel.py', 'w') as f:
    f.write(parallel_script)
s.close()

_, out, err = c.exec_command('python3 /root/run_parallel.py')
print(out.read().decode())
if err.read().decode():
    print("ERR:", err.read().decode())

c.close()
