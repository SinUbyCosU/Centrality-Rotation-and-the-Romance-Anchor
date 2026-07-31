#!/bin/bash
LOG="/root/clr_paper/watchdog_scheduler.log"
echo "$(date): Watchdog started" >> $LOG

while true; do
    if ! pgrep -f "dynamic_scheduler.py" > /dev/null; then
        echo "$(date): Scheduler down, restarting..." >> $LOG
        nohup python3 /root/clr_paper/dynamic_scheduler.py > /dev/null 2>&1 &
    fi
    sleep 60
done
