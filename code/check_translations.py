import paramiko, json, sys
sys.stdout.reconfigure(encoding='utf-8')

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('216.128.144.102', username='root', password='[8eE967Lg}!(GZoz')

# Check translated safety prompt files
for lang in ['en', 'hi', 'ar', 'fr', 'de', 'sw']:
    cmd = f"python3 -c \"import json; d=json.load(open('/root/clr_paper/data/safety_prompts_{lang}.json')); print(len(d)); print(repr(d[0][0][:150]))\""
    _, out, err = c.exec_command(cmd)
    result = out.read().decode('utf-8', errors='replace').strip()
    error = err.read().decode('utf-8', errors='replace').strip()
    if result:
        print(f"{lang}: {result}")
    else:
        print(f"{lang}: MISSING ({error[:100]})")

# Now the CRITICAL check: what does the translation_attack ACTUALLY look like
# in the raw JSONL? The forensics showed "{harmful_prompt_in_target_lang}" was
# NOT substituted. Let's check more prompts.
print("\n\nCHECKING TRANSLATION ATTACK PROMPTS IN RAW JSONL:")
print("=" * 70)

# Read raw JSONL locally for zephyr-7b
import os
raw_path = r'C:\Users\Tanushree\Downloads\work\audit\zephyr-7b_raw.jsonl'
if os.path.exists(raw_path):
    with open(raw_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('attack_type') == 'translation_attack':
                lang = obj.get('language', '')
                prompt = obj.get('prompt', '')
                print(f"  [{lang}] prompt: {repr(prompt[:200])}")
                if i > 20 and lang not in ('en',):
                    break
    
    # Count how many translation_attack prompts contain the literal placeholder
    placeholder_count = 0
    total_trans = 0
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('attack_type') == 'translation_attack':
                total_trans += 1
                if '{harmful_prompt_in_target_lang}' in obj.get('prompt', ''):
                    placeholder_count += 1
    
    print(f"\n  Total translation_attack prompts: {total_trans}")
    print(f"  Prompts still containing literal placeholder: {placeholder_count}")
    print(f"  Prompts with SUBSTITUTED text: {total_trans - placeholder_count}")

c.close()
