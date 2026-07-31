"""Quick remote inspection of the actual JSON structure to understand what fields exist."""
import paramiko, json, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect("216.128.144.102", username="root", password="[8eE967Lg}!(GZoz", timeout=30)

def ssh_exec(cmd):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

# Upload a small inspection script
inspect_script = '''
import json, os, glob

base = "/root/clr_paper/results"
dirs = sorted(glob.glob(f"{base}/steering_vectors_*/phase2_pivot_analysis/phase2_results.json"))

print(f"Found {len(dirs)} Phase 2 results")
print()

# Check first model in detail
for f in dirs[:3]:
    model = f.split("steering_vectors_")[1].split("/")[0].rsplit("_2026",1)[0]
    d = json.load(open(f))
    print(f"=== {model} ===")
    print(f"  Keys: {list(d.keys())}")
    print(f"  pivot_centrality: {d.get('pivot_centrality')}")
    print(f"  pivot_projection: {d.get('pivot_projection')}")
    print(f"  language_order: {d.get('language_order')}")
    print(f"  centrality_scores: {d.get('centrality_scores')}")
    scd = d.get("scd_scores_degrees", {})
    print(f"  SCD degrees: {json.dumps(scd)}")
    print()

# Check Phase 4
p4_dirs = sorted(glob.glob(f"{base}/steering_vectors_*/phase4_psycholinguistic/phase4_results.json"))
print(f"\\nFound {len(p4_dirs)} Phase 4 results")
for f in p4_dirs[:3]:
    model = f.split("steering_vectors_")[1].split("/")[0].rsplit("_2026",1)[0]
    d = json.load(open(f))
    print(f"=== {model} ===")
    print(f"  Keys: {list(d.keys())}")
    for k,v in d.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")
    print()

# Check Phase 7
p7_dirs = sorted(glob.glob(f"{base}/steering_vectors_*/phase7_codemix/phase7_results.json"))
print(f"\\nFound {len(p7_dirs)} Phase 7 results")
for f in p7_dirs[:2]:
    model = f.split("steering_vectors_")[1].split("/")[0].rsplit("_2026",1)[0]
    d = json.load(open(f))
    print(f"=== {model} ===")
    print(f"  Keys: {list(d.keys())}")
    langs = d.get("languages", {})
    print(f"  N languages: {len(langs)}")
    for lang, ldata in list(langs.items())[:2]:
        print(f"  {lang}: {json.dumps(ldata)[:200]}")
    print()

# Check Phase 8
p8_dirs = sorted(glob.glob(f"{base}/steering_vectors_*/phase8_causal_intervention/phase8_results.json"))
print(f"\\nFound {len(p8_dirs)} Phase 8 results")
for f in p8_dirs[:3]:
    model = f.split("steering_vectors_")[1].split("/")[0].rsplit("_2026",1)[0]
    d = json.load(open(f))
    print(f"=== {model} ===")
    print(f"  Keys: {list(d.keys())}")
    n_langs = len(d.get("languages", {}))
    print(f"  N languages: {n_langs}")
    for lang, ldata in list(d.get("languages", {}).items())[:2]:
        print(f"  {lang}: baseline_refusal={ldata.get('baseline_refusal_rate')}, best_steered={ldata.get('best_steered_refusal_rate')}, lift={ldata.get('safety_lift')}")
        alphas = ldata.get("alpha_results", {})
        for a, adata in alphas.items():
            print(f"    alpha={a}: refused={adata.get('refused')}/{adata.get('total')} = {adata.get('refusal_rate')}")
    print()

# Check Phase 9
p9_f = f"{base}/phase9_cross_model_transfer/phase9_results.json"
if os.path.exists(p9_f):
    d = json.load(open(p9_f))
    print(f"\\n=== PHASE 9 ===")
    print(f"  Keys: {list(d.keys())}")
    print(f"  n_models: {d.get('n_models')}")
    within = d.get("within_family_sim", {})
    cross = d.get("cross_family_sim", {})
    print(f"  within_family: {within}")
    print(f"  cross_family: {cross}")
    print(f"  mann_whitney_p: {d.get('mann_whitney_p')}")
    per_lang = d.get("per_language_transfer", {})
    for lang, stats in list(per_lang.items())[:3]:
        print(f"  {lang}: {stats}")
'''

# Write and execute the script on the remote server
sftp = client.open_sftp()
with sftp.open("/root/clr_paper/inspect_results.py", "w") as f:
    f.write(inspect_script)
sftp.close()

out, err = ssh_exec("cd /root/clr_paper && python3 inspect_results.py")
if err:
    print(f"STDERR: {err[:500]}")
print(out)

client.close()
