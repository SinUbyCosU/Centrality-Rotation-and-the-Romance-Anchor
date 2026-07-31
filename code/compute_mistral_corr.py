import paramiko
import json
import numpy as np
from scipy.stats import pearsonr

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

_, o, _ = c.exec_command('cat /root/clr_paper/results/steering_vectors_mistral-7b_20260622_053716/phase11_scaled_adversarial/phase11_results.json')
p11_data = json.loads(o.read().decode())

_, o, _ = c.exec_command('cat /root/clr_paper/results/steering_vectors_mistral-7b_20260622_053716/phase2_results.json')
p2_data = json.loads(o.read().decode())

# The languages are in p11_data["languages"]
# Each language has ASR values. We average the ASR across categories/scales.
# Or wait, how is the ASR computed for the correlation?
# In Phase 11, ASR is computed per language for scale=1.0?
# Let's check how I computed the correlation in previous turns.
