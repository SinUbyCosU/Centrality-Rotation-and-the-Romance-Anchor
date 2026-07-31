"""SSH helper: run commands on the remote GPU server."""
import paramiko
import sys

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

def ssh_exec(cmd, timeout=300):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=30)
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    client.close()
    return out, err, exit_code

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "echo hello"
    out, err, code = ssh_exec(cmd)
    if out: print(out.encode('ascii', errors='replace').decode('ascii'))
    if err: print("STDERR:", err)
    print(f"Exit code: {code}")
