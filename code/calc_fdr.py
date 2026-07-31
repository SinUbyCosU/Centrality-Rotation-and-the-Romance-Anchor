import json
import numpy as np

# Load original p11 results
with open('c:\\Users\\Tanushree\\Downloads\\work\\fix_and_recompute_p11.py', 'r') as f:
    code = f.read()
    
# I will just write a new script that parses the markdown of Table C and removes mistral-7b and qwen-7b.
# But wait, I need to recalculate the FDR for the remaining 12 models.

# Let's list the 12 instruct models and their Pearson p-values from my last Table C.
models = [
    ("falcon3-7b-instruct", 0.0435),
    ("mistral-7b-instruct-v0.3", 0.4826),
    ("openhermes-2.5-mistral-7b", 0.9943),
    ("phi-3-mini-4k-instruct", 0.0001),
    # phi-4-mini-instruct is N/A-genuine
    ("qwen2-7b-instruct", 0.8432),
    ("qwen2.5-3b-instruct", 0.1191),
    ("qwen2.5-7b-instruct", 0.5648),
    ("qwen3-4b-instruct", 0.6010),
    ("smollm3-3b", 0.5451),
    ("stablelm-2-1.6b-chat", 0.2094),
    ("tinyllama", 0.4973),
    ("yi-1.5-6b-chat", 0.1392),
    ("zephyr-7b", 0.0169),
]
# Wait, let's count:
# 1 falcon3, 2 mistral-v0.3, 3 openhermes, 4 phi-3, 5 qwen2-7b, 6 qwen2.5-3b, 7 qwen2.5-7b, 8 qwen3-4b, 9 smollm3-3b, 10 stablelm-2-1.6b, 11 tinyllama, 12 yi-1.5, 13 zephyr.
# That is 13 models!
# Wait, why 13?
# The original m=14 included mistral-7b. Removing mistral-7b gives 13.
# qwen-7b was ALREADY missing from the correlation list because it had N/A or variance issues. (Actually, qwen-7b didn't even have a Phase 11 run because I didn't see it).

# So there are 13 valid p-values to run FDR on.
# Let's run Benjamini-Hochberg on these 13.

pvals = [x[1] for x in models]
n_tests = len(pvals)
sorted_indices = np.argsort(pvals)
sorted_pvals = np.array(pvals)[sorted_indices]
bh_critical = np.array([(i+1) / n_tests * 0.05 for i in range(n_tests)])

last_sig = -1
for k in range(n_tests):
    if sorted_pvals[k] <= bh_critical[k]:
        last_sig = k

fdr_sig = {}
for orig_idx, sort_idx in enumerate(sorted_indices):
    model = models[sort_idx][0]
    is_sig = (orig_idx <= last_sig)
    fdr_sig[model] = is_sig

for m, p in models:
    sig = fdr_sig[m]
    print(f"{m}: p={p}, fdr={sig}")
