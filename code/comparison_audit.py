"""Quick summary: classify_safety vs LLM judge for all models"""
import json, os
from collections import Counter

AUDIT_DIR = r'C:\Users\Tanushree\Downloads\work\audit'
JUDGE_DIR = r'C:\Users\Tanushree\Downloads\work\judge_results'

models = [
    'falcon3-7b-instruct', 'mistral-7b-instruct-v0.3', 'openhermes-2.5-mistral-7b',
    'qwen2-7b-instruct', 'qwen2.5-3b-instruct', 'qwen2.5-7b-instruct',
    'qwen3-4b-instruct', 'smollm3-3b', 'stablelm-2-1.6b-chat', 'zephyr-7b',
]

print(f"{'Model':<35} {'refused':>8} {'ambiguous':>9} {'complied':>9} | {'J-refused':>10} {'J-complied':>11}")
print("-" * 100)

for model in models:
    raw_path = os.path.join(AUDIT_DIR, f"{model}_raw.jsonl")
    judge_path = os.path.join(JUDGE_DIR, f"{model}_judged.jsonl")
    
    gen_cls = Counter()
    if os.path.exists(raw_path):
        with open(raw_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    gen_cls[obj.get('classification', 'MISSING')] += 1
                except: pass
    
    judge_cls = Counter()
    if os.path.exists(judge_path):
        with open(judge_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    obj = json.loads(line)
                    judge_cls[obj.get('llm_judge', 'MISSING')] += 1
                except: pass
    
    total = sum(gen_cls.values())
    ref = gen_cls.get('refused', 0)
    amb = gen_cls.get('ambiguous', 0)
    com = gen_cls.get('complied', 0)
    
    jref = judge_cls.get('refused', 0)
    jcom = judge_cls.get('complied', 0)
    
    print(f"{model:<35} {ref:>8} {amb:>9} {com:>9} | {jref:>10} {jcom:>11}")

# Now show a few actual "complied" responses from the generation classifier
# that the LLM judge marked as "refused" — this reveals the discrepancy
print("\n\n" + "=" * 80)
print("EXAMPLES: Responses classified 'complied' by classify_safety but 'refused' by LLM judge")
print("=" * 80)

for model in ['stablelm-2-1.6b-chat', 'zephyr-7b', 'openhermes-2.5-mistral-7b']:
    judge_path = os.path.join(JUDGE_DIR, f"{model}_judged.jsonl")
    if not os.path.exists(judge_path): continue
    
    print(f"\n--- {model} ---")
    count = 0
    with open(judge_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                # Show cases where gen said "complied" but judge said "refused"
                if obj.get('classification') == 'complied' and obj.get('llm_judge') == 'refused':
                    count += 1
                    if count <= 3:
                        print(f"  [{obj.get('language')}/{obj.get('attack_type')}]")
                        print(f"  Prompt: {obj['prompt'][:100]}")
                        print(f"  Response: {obj['response'][:200]}")
                        print(f"  classify_safety={obj['classification']}, judge={obj['llm_judge']}")
                        print()
            except: pass
    print(f"  Total classify_safety=complied but judge=refused: {count}")

# Also show cases where gen said "ambiguous" — what did the judge say?
print("\n\n" + "=" * 80)
print("EXAMPLES: 'ambiguous' responses — what does the LLM judge say?")
print("=" * 80)

for model in ['stablelm-2-1.6b-chat', 'zephyr-7b']:
    judge_path = os.path.join(JUDGE_DIR, f"{model}_judged.jsonl")
    if not os.path.exists(judge_path): continue
    
    print(f"\n--- {model} ---")
    amb_judge_cls = Counter()
    samples = []
    with open(judge_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                obj = json.loads(line)
                if obj.get('classification') == 'ambiguous':
                    amb_judge_cls[obj.get('llm_judge', 'MISSING')] += 1
                    if len(samples) < 3:
                        samples.append(obj)
            except: pass
    print(f"  Ambiguous responses re-classified by judge: {dict(amb_judge_cls.most_common())}")
    for s in samples:
        print(f"  [{s.get('language')}/{s.get('attack_type')}] response={s['response'][:200]}")
        print(f"    judge={s['llm_judge']}\n")
