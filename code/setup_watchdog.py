import paramiko, time

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# 1. Check if Exp 18 is actively running
_, out, _ = c.exec_command('ps aux | grep 18_las_behavioral | grep -v grep')
proc = out.read().decode().strip()
print("=== CURRENT PROCESS ===")
print(proc if proc else "NOT RUNNING!")

# 2. Check if it's making progress (log file growing)
_, out, _ = c.exec_command('wc -l /root/clr_paper/exp18_las_v3.log')
print("\nLog lines:", out.read().decode().strip())

# 3. Create a watchdog script on the server that checks every 5 min
#    and restarts Exp 18 if it died (but not if it finished all 20)
watchdog = '''#!/bin/bash
LOG="/root/clr_paper/watchdog.log"
echo "$(date): Watchdog started" >> $LOG

while true; do
    sleep 300  # check every 5 minutes
    
    # Count completed models
    DONE=$(ls -d /root/clr_paper/results/steering_vectors_*/phase18_las_behavioral 2>/dev/null | wc -l)
    
    # If all 20 are done, stop watchdog
    if [ "$DONE" -ge 20 ]; then
        echo "$(date): All 20 models complete! Watchdog exiting." >> $LOG
        # Now start Exp 11 if not already running
        if ! pgrep -f "11_scaled_adversarial" > /dev/null; then
            echo "$(date): Starting Exp 11..." >> $LOG
            nohup python3 /root/clr_paper/experiments/11_scaled_adversarial.py --model-key all --results-dir /root/clr_paper/results > /root/clr_paper/exp11_remaining.log 2>&1 &
        fi
        exit 0
    fi
    
    # Check if Exp 18 is running
    if ! pgrep -f "18_las_behavioral" > /dev/null; then
        echo "$(date): Exp 18 DIED! Only $DONE/20 done. Restarting..." >> $LOG
        nohup python3 /root/clr_paper/experiments/18_las_behavioral.py --model-key all --n-prompts 30 > /root/clr_paper/exp18_las_restart_$(date +%s).log 2>&1 &
        echo "$(date): Restarted Exp 18." >> $LOG
    else
        echo "$(date): Exp 18 running OK. $DONE/20 models complete." >> $LOG
    fi
done
'''

s = c.open_sftp()
with s.file('/root/clr_paper/watchdog.sh', 'w') as f:
    f.write(watchdog)
s.close()

# Kill any old watchdog, then start new one
c.exec_command('pkill -f "watchdog.sh"')
time.sleep(1)
c.exec_command('chmod +x /root/clr_paper/watchdog.sh')
c.exec_command('nohup bash /root/clr_paper/watchdog.sh > /dev/null 2>&1 &')
print("\n=== WATCHDOG INSTALLED ===")
print("- Checks every 5 minutes if Exp 18 is still alive")
print("- Auto-restarts it if it died (skips already-done models)")
print("- Launches Exp 11 once all 20 are complete")
print("- Logs to /root/clr_paper/watchdog.log")

# Kill old queue scripts since watchdog replaces them
c.exec_command('pkill -f "queue_exp11"')

# Verify everything
time.sleep(2)
_, out, _ = c.exec_command('ps aux | grep -E "18_las|watchdog" | grep -v grep')
print("\n=== RUNNING PROCESSES ===")
print(out.read().decode())

c.close()
