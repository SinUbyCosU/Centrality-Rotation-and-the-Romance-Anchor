"""
Local audit — analyze the downloaded raw JSONL and results.json files.
Focus on classification breakdowns and response content.
"""
import json
import os
from collections import Counter

AUDIT_DIR = r'C:\Users\Tanushree\Downloads\work\audit'
JUDGE_DIR = r'C:\Users\Tanushree\Downloads\work\judge_results'

models = [
    'falcon3-7b-instruct',
    'mistral-7b-instruct-v0.3',
    'openhermes-2.5-mistral-7b',
    'qwen2-7b-instruct',
    'qwen2.5-3b-instruct',
    'qwen2.5-7b-instruct',
    'qwen3-4b-instruct',
    'smollm3-3b',
    'stablelm-2-1.6b-chat',
    'zephyr-7b',
]

print("=" * 80)
print("TASK 2: FULL CLASSIFICATION BREAKDOWN (classify_safety vs LLM Judge)")
print("=" * 80)

for model in models:
    raw_path = os.path.join(AUDIT_DIR, f"{model}_raw.jsonl")
    judge_path = os.path.join(JUDGE_DIR, f"{model}_judged.jsonl")
    results_path = os.path.join(AUDIT_DIR, f"{model}_results.json")
    
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")
    
    # --- Raw JSONL: classify_safety breakdown ---
    if os.path.exists(raw_path):
        gen_cls = Counter()
        attack_breakdown = {}  # attack_type -> Counter of classifications
        lang_breakdown = {}    # lang -> Counter of classifications
        sample_responses = []
        total = 0
        
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    cls = obj.get('classification', 'MISSING')
                    atk = obj.get('attack_type', 'unknown')
                    lang = obj.get('language', 'unknown')
                    gen_cls[cls] += 1
                    
                    if atk not in attack_breakdown:
                        attack_breakdown[atk] = Counter()
                    attack_breakdown[atk][cls] += 1
                    
                    if lang not in lang_breakdown:
                        lang_breakdown[lang] = Counter()
                    lang_breakdown[lang][cls] += 1
                    
                    total += 1
                    
                    # Collect some complied/ambiguous samples
                    if cls in ('complied', 'ambiguous') and len(sample_responses) < 5:
                        sample_responses.append({
                            'attack': atk,
                            'lang': lang,
                            'classification': cls,
                            'prompt': obj.get('prompt', '')[:100],
                            'response': obj.get('response', '')[:200]
                        })
                except:
                    pass
        
        print(f"\n  classify_safety() breakdown ({total} total):")
        for cls, cnt in gen_cls.most_common():
            pct = cnt / total * 100 if total else 0
            print(f"    {cls:>10}: {cnt:>5} ({pct:>5.1f}%)")
        
        print(f"\n  By attack type:")
        for atk in sorted(attack_breakdown.keys()):
            counts = attack_breakdown[atk]
            atk_total = sum(counts.values())
            parts = ', '.join(f"{c}={n}" for c, n in counts.most_common())
            print(f"    {atk:>20}: total={atk_total} | {parts}")
        
        print(f"\n  By language:")
        for lang in sorted(lang_breakdown.keys()):
            counts = lang_breakdown[lang]
            lang_total = sum(counts.values())
            parts = ', '.join(f"{c}={n}" for c, n in counts.most_common())
            print(f"    {lang:>8}: total={lang_total} | {parts}")
        
        if sample_responses:
            print(f"\n  Sample COMPLIED/AMBIGUOUS responses:")
            for s in sample_responses[:3]:
                print(f"    [{s['lang']}/{s['attack']}] cls={s['classification']}")
                print(f"      Prompt: {s['prompt']}")
                print(f"      Response: {s['response'][:150]}")
                print()
    
    # --- LLM Judge breakdown ---
    if os.path.exists(judge_path):
        judge_cls = Counter()
        judge_raw_vals = Counter()
        with open(judge_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    judge_cls[obj.get('llm_judge', 'MISSING')] += 1
                    judge_raw_vals[obj.get('judge_raw', 'MISSING')] += 1
                except:
                    pass
        jtotal = sum(judge_cls.values())
        print(f"\n  LLM Judge breakdown ({jtotal} total):")
        for cls, cnt in judge_cls.most_common():
            pct = cnt / jtotal * 100 if jtotal else 0
            print(f"    {cls:>10}: {cnt:>5} ({pct:>5.1f}%)")
        print(f"  Judge raw token distribution: {dict(judge_raw_vals.most_common(5))}")
    
    # --- Generation-time results.json ASR ---
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n  phase11_results.json keys: {list(data.keys())}")
        if 'per_language' in data:
            for lang, ld in data['per_language'].items():
                if isinstance(ld, dict):
                    asr = ld.get('overall_asr', 'N/A')
                    if isinstance(asr, (int, float)):
                        asr_str = f"{asr*100:.1f}%"
                    else:
                        asr_str = str(asr)
                    print(f"    Lang {lang}: generation-time ASR = {asr_str}")
                    if 'attacks' in ld:
                        for aname, ad in ld['attacks'].items():
                            if isinstance(ad, dict):
                                print(f"      {aname}: complied={ad.get('complied',0)}, refused={ad.get('refused',0)}, ambiguous={ad.get('ambiguous',0)}, total={ad.get('total',0)}")

print("\n\nLOCAL AUDIT COMPLETE.")
