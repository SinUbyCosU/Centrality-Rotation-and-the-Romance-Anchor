import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Kill Exp 11 process
c.exec_command('pkill -f "11_scaled_adversarial"')

print("Killed Exp 11 (Scaled Adversarial).")
print("Exp 18 now has exclusive access to both GPUs.")
print("The watchdog will automatically restart Exp 11 once Exp 18 finishes all 20 models.")

c.close()
