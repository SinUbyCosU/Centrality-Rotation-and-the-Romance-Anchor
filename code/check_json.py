import json, glob, os, numpy as np

base = '/root/clr_paper/results/steering_vectors_*'
models_dirs = sorted(glob.glob(base))

print("="*70)
print("EXP 18: CORRECTED REFUSAL RATES (refused / (refused+complied), excluding ambiguous)")
print("="*70)
print("| Model | Base Ref% | Raw Eng Ref% | LAS Ref% | LAS Advantage |")
print("|---|---|---|---|---|")

for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase18_las_behavioral', 'phase18_results.json')
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    langs = r.get('languages', {})
    
    base_refs, raw_refs, las_refs = [], [], []
    base_refs_full, raw_refs_full, las_refs_full = [], [], []
    
    for lang, ld in langs.items():
        conds = ld.get('conditions', {})
        if not conds:
            continue
        
        for cname, arr in [('baseline', base_refs), ('raw_english', raw_refs), ('las_rotated', las_refs)]:
            c = conds.get(cname, {})
            ref = c.get('refused', 0)
            comp = c.get('complied', 0)
            ambig = c.get('ambiguous', 0)
            total = ref + comp + ambig
            # Method 1: refused / total (including ambiguous)
            if cname == 'baseline':
                base_refs_full.append(ref / max(total, 1))
            elif cname == 'raw_english':
                raw_refs_full.append(ref / max(total, 1))
            else:
                las_refs_full.append(ref / max(total, 1))
            # Method 2: refused / (refused + complied), excluding ambiguous
            decided = ref + comp
            if decided > 0:
                arr.append(ref / decided)
    
    base_mean = np.mean(base_refs) * 100 if base_refs else 0
    raw_mean = np.mean(raw_refs) * 100 if raw_refs else 0
    las_mean = np.mean(las_refs) * 100 if las_refs else 0
    
    base_full = np.mean(base_refs_full) * 100 if base_refs_full else 0
    raw_full = np.mean(raw_refs_full) * 100 if raw_refs_full else 0
    las_full = np.mean(las_refs_full) * 100 if las_refs_full else 0
    
    agg = r.get('aggregate', {})
    las_adv = agg.get('mean_las_advantage', 0)
    
    print(f"| {model} | {base_full:.1f}% | {raw_full:.1f}% | {las_full:.1f}% | {las_adv:+.3f} |")

print("\n\n")
print("="*70)
print("EXP 8: CORRECTED - checking if conditions have data")
print("="*70)
for d in models_dirs[:5]:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase8_causal_intervention', 'phase8_results.json')
    if not os.path.exists(f):
        continue
    r = json.load(open(f))
    print(f"\n{model}: best_layer={r.get('best_layer')}")
    langs = r.get('languages', {})
    for lang in ['de', 'ar', 'hi']:
        ld = langs.get(lang, {})
        conds = ld.get('conditions', {})
        if conds:
            for cname, cdata in conds.items():
                print(f"  {lang}/{cname}: ref={cdata.get('refused',0)} comp={cdata.get('complied',0)} ambig={cdata.get('ambiguous',0)}")
        else:
            print(f"  {lang}: NO conditions data")
