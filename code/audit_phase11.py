"""
Audit script — TASK 1-4 investigation.
Downloads raw JSONL data and phase11_results.json for each model.
Computes the REAL classification breakdown (from classify_safety in the generation script)
vs the LLM judge classification.
Also checks phi-3-mini status.
"""
import paramiko
import json
import os
import glob
from collections import Counter

HOST = '216.128.144.102'
USER = 'root'
PASS = '[8eE967Lg}!(GZoz'

LOCAL_AUDIT = r'C:\Users\Tanushree\Downloads\work\audit'
os.makedirs(LOCAL_AUDIT, exist_ok=True)

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS)
sftp = c.open_sftp()

# ============================================================
# TASK 1: Identify sources — timestamps and file paths
# ============================================================
print("=" * 70)
print("TASK 1: SOURCE IDENTIFICATION")
print("=" * 70)

_, out, _ = c.exec_command(
    'ls -la /root/clr_paper/results/*/phase11_scaled_adversarial/phase11_results.json '
    '/root/clr_paper/results/*/phase11_scaled_adversarial/phase11_raw_responses.jsonl '
    '2>/dev/null | head -30'
)
print(out.read().decode('utf-8', errors='replace'))

# Check script hash/timestamp
_, out, _ = c.exec_command('ls -la /root/clr_paper/experiments/11_scaled_adversarial_v5.py')
print("Script on server:")
print(out.read().decode('utf-8', errors='replace'))

_, out, _ = c.exec_command('md5sum /root/clr_paper/experiments/11_scaled_adversarial_v5.py')
print("Script MD5:", out.read().decode('utf-8', errors='replace'))

# ============================================================
# TASK 2: Classification breakdown audit
# ============================================================
print("=" * 70)
print("TASK 2: CLASSIFICATION BREAKDOWN AUDIT")
print("=" * 70)

# Download raw JSONL for each model and compute breakdown
_, out, _ = c.exec_command(
    'ls /root/clr_paper/results/*/phase11_scaled_adversarial/phase11_raw_responses.jsonl 2>/dev/null'
)
raw_files = [l.strip() for l in out.read().decode('utf-8', errors='replace').strip().split('\n') if l.strip()]

for rf in raw_files:
    model_dir = rf.split('/results/')[1].split('/phase11')[0]
    model_name = model_dir.replace('steering_vectors_', '').rsplit('_2026', 1)[0]
    
    local_path = os.path.join(LOCAL_AUDIT, f"{model_name}_raw.jsonl")
    
    print(f"\n--- {model_name} ---")
    try:
        sftp.get(rf, local_path)
    except Exception as e:
        print(f"  Download error: {e}")
        continue
    
    # Read and analyze
    classifications = Counter()
    judge_classifications = Counter()  # from the judge file if exists
    responses_sample = []
    total = 0
    
    with open(local_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                classifications[obj.get('classification', 'MISSING')] += 1
                total += 1
                if total <= 3:
                    responses_sample.append({
                        'attack': obj.get('attack_type'),
                        'lang': obj.get('language'),
                        'classification': obj.get('classification'),
                        'response_preview': obj.get('response', '')[:150]
                    })
            except:
                pass
    
    print(f"  Total responses: {total}")
    print(f"  classify_safety breakdown:")
    for cls, cnt in classifications.most_common():
        pct = cnt / total * 100 if total else 0
        print(f"    {cls}: {cnt} ({pct:.1f}%)")
    
    # Also check the LLM judge file
    judge_local = os.path.join(r'C:\Users\Tanushree\Downloads\work\judge_results', f"{model_name}_judged.jsonl")
    if os.path.exists(judge_local):
        jc = Counter()
        judge_raw_samples = []
        with open(judge_local, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    jc[obj.get('llm_judge', 'MISSING')] += 1
                    if len(judge_raw_samples) < 5:
                        judge_raw_samples.append(obj.get('judge_raw', ''))
                except:
                    pass
        print(f"  LLM judge breakdown:")
        for cls, cnt in jc.most_common():
            pct = cnt / sum(jc.values()) * 100 if jc else 0
            print(f"    {cls}: {cnt} ({pct:.1f}%)")
        print(f"  LLM judge raw samples: {judge_raw_samples}")
    
    print(f"  First 3 response samples:")
    for s in responses_sample:
        print(f"    [{s['lang']}/{s['attack']}] cls={s['classification']}")
        print(f"      Response: {s['response_preview'][:100]}...")


# ============================================================
# TASK 3: phi-3-mini exclusion reason
# ============================================================
print("\n" + "=" * 70)
print("TASK 3: PHI-3-MINI EXCLUSION INVESTIGATION")
print("=" * 70)

# Check if phi-3-mini has any phase11 directory at all
_, out, _ = c.exec_command(
    'ls -la /root/clr_paper/results/steering_vectors_phi-3-mini*/ 2>/dev/null'
)
print("phi-3-mini directories:", out.read().decode('utf-8', errors='replace'))

# Check for LOAD_FAILED.txt marker
_, out, _ = c.exec_command(
    'find /root/clr_paper/results -name "LOAD_FAILED.txt" -exec echo "Found: {}" \\; -exec cat {} \\;'
)
print("LOAD_FAILED markers:", out.read().decode('utf-8', errors='replace'))

# Check batch log for phi-3-mini entries
_, out, _ = c.exec_command(
    'grep -A 10 "phi-3-mini" /root/clr_paper/batch_gpu0.log'
)
print("Batch log for phi-3-mini:", out.read().decode('utf-8', errors='replace'))

# Check if phi-3-mini had results from a PREVIOUS run (Table C)
_, out, _ = c.exec_command(
    'ls -la /root/clr_paper/results/steering_vectors_phi-3-mini*/phase11_scaled_adversarial/ 2>/dev/null'
)
print("phi-3-mini phase11 dir:", out.read().decode('utf-8', errors='replace'))


# ============================================================
# TASK 4: Compare with Table C (existing results)
# ============================================================
print("\n" + "=" * 70)
print("TASK 4: COMPARISON WITH EXISTING TABLE C")
print("=" * 70)

# Check if there are any PREVIOUS phase11 results (from the original run)
_, out, _ = c.exec_command(
    'find /root/clr_paper/results -name "phase11_results.json" -exec ls -la {} \\;'
)
print("All phase11_results.json files with timestamps:")
print(out.read().decode('utf-8', errors='replace'))

# Download the phase11_results.json (generation-time summary) for each model
_, out, _ = c.exec_command(
    'ls /root/clr_paper/results/*/phase11_scaled_adversarial/phase11_results.json 2>/dev/null'
)
result_files = [l.strip() for l in out.read().decode('utf-8', errors='replace').strip().split('\n') if l.strip()]

for rf in result_files:
    model_dir = rf.split('/results/')[1].split('/phase11')[0]
    model_name = model_dir.replace('steering_vectors_', '').rsplit('_2026', 1)[0]
    
    local_path = os.path.join(LOCAL_AUDIT, f"{model_name}_results.json")
    try:
        sftp.get(rf, local_path)
        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the overall ASR from the generation-time classify_safety
        print(f"\n{model_name}:")
        if 'overall_asr' in data:
            print(f"  Generation-time overall_asr: {data['overall_asr']*100:.1f}%")
        if 'per_language' in data:
            for lang, lang_data in data['per_language'].items():
                if isinstance(lang_data, dict) and 'overall_asr' in lang_data:
                    print(f"  Lang {lang}: ASR={lang_data['overall_asr']*100:.1f}%")
                if isinstance(lang_data, dict) and 'attacks' in lang_data:
                    for atk_name, atk_data in lang_data['attacks'].items():
                        if isinstance(atk_data, dict):
                            print(f"    {atk_name}: asr={atk_data.get('asr', 0)*100:.1f}% "
                                  f"(complied={atk_data.get('complied', 0)}, "
                                  f"refused={atk_data.get('refused', 0)}, "
                                  f"ambiguous={atk_data.get('ambiguous', 0)}, "
                                  f"total={atk_data.get('total', 0)})")
    except Exception as e:
        print(f"  Error: {e}")

c.close()
print("\n\nAUDIT COMPLETE.")
