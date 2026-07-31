import json, glob, os, numpy as np

base = '/root/clr_paper/results/steering_vectors_*'
models_dirs = sorted(glob.glob(base))

# ===== PHASE 2: Extract actual pivot + romance/germanic from the JSON =====
print("="*70)
print("PHASE 2: SCD + Pivot (from the all_experiments_tables data)")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase2_pivot_analysis', 'phase2_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        scd = r.get('scd_scores_radians', {})
        # Check all keys
        pivot = r.get('pivot_language', r.get('pivot', 'N/A'))
        romance = r.get('romance_similarity', r.get('romance_sim', 'N/A'))
        germanic = r.get('germanic_similarity', r.get('germanic_sim', 'N/A'))
        # Try nested
        if pivot == 'N/A' and 'analysis' in r:
            pivot = r['analysis'].get('pivot_language', 'N/A')
        vals = [v for k,v in scd.items() if k != 'en']
        mean_scd = np.mean(vals) if vals else 0
        max_lang = max(scd, key=scd.get) if scd else 'N/A'
        min_lang = min(scd, key=scd.get) if scd else 'N/A'
        print(f"  {model}: mean_SCD_rad={mean_scd:.3f} ({np.degrees(mean_scd):.1f}deg) max={max_lang}({np.degrees(scd.get(max_lang,0)):.1f}deg) min={min_lang}({np.degrees(scd.get(min_lang,0)):.1f}deg)")

# ===== PHASE 4: Extract actual delta_R2 =====
print("\n" + "="*70)
print("PHASE 4: Psycholinguistic delta_R2")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase4_psycholinguistic', 'phase4_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        imp = r.get('psycholinguistic_improvement', None)
        reg = r.get('regression_models', {})
        scd_only = reg.get('scd_only', {})
        combined = reg.get('combined', {})
        if isinstance(imp, dict):
            delta = imp.get('delta_r2', imp.get('delta_R2', 'N/A'))
        else:
            delta = imp
        scd_r2 = scd_only.get('r2', 'N/A') if isinstance(scd_only, dict) else scd_only
        comb_r2 = combined.get('r2', 'N/A') if isinstance(combined, dict) else combined
        print(f"  {model}: delta_R2={delta} scd_r2={scd_r2} combined_r2={comb_r2}")

# ===== PHASE 5: LAS fidelity =====
print("\n" + "="*70)
print("PHASE 5: LAS Fidelity")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase5_las', 'phase5_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        fid = r.get('fidelity', 'N/A')
        rot_corr = r.get('rotation_psycholing_correlation', {})
        print(f"  {model}: fidelity={fid} rot_corr_r={rot_corr.get('r', 'N/A')} rot_corr_p={rot_corr.get('p', 'N/A')}")

# ===== PHASE 8: Causal Intervention =====
print("\n" + "="*70)
print("PHASE 8: Causal Intervention (SCD vs refusal lift)")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase8_causal_intervention', 'phase8_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        corr = r.get('scd_vs_lift_correlation', {})
        langs = r.get('languages', {})
        # Get mean baseline and steered refusal
        base_refs = []
        steered_refs = []
        for lang, ld in langs.items():
            conds = ld.get('conditions', {})
            baseline = conds.get('baseline', {})
            steered = conds.get('steered', conds.get('english_steered', {}))
            base_refs.append(baseline.get('refused', 0) / max(baseline.get('refused', 0) + baseline.get('complied', 0) + baseline.get('ambiguous', 0), 1))
            steered_refs.append(steered.get('refused', 0) / max(steered.get('refused', 0) + steered.get('complied', 0) + steered.get('ambiguous', 0), 1))
        print(f"  {model}: scd_vs_lift r={corr.get('r', 'N/A')} p={corr.get('p', 'N/A')} mean_base_ref={np.mean(base_refs):.3f} mean_steered_ref={np.mean(steered_refs):.3f}")

# ===== PHASE 10: Adversarial Probing =====
print("\n" + "="*70)
print("PHASE 10: Adversarial Probing")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase10_adversarial_probing', 'phase10_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        scd_asr = r.get('scd_vs_asr', {})
        langs = r.get('languages', {})
        asrs = []
        for lang, ld in langs.items():
            asrs.append(ld.get('overall_asr', 0))
        print(f"  {model}: mean_ASR={np.mean(asrs)*100:.1f}% scd_vs_asr r={scd_asr.get('pearson_r', 'N/A')} p={scd_asr.get('p_value', 'N/A')}")

# ===== PHASE 17: Moral Foundations =====
print("\n" + "="*70)
print("PHASE 17: Moral Foundations (WEIRD vs Non-WEIRD)")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase17_moral_foundations', 'phase17_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        langs = r.get('languages', {})
        weird_aligns = []
        nonweird_aligns = []
        for lang, ld in langs.items():
            founds = ld.get('foundations', {})
            for fname, fdata in founds.items():
                cat = fdata.get('category', '')
                al = fdata.get('alignment_with_overall', 0)
                if cat == 'WEIRD':
                    weird_aligns.append(al)
                elif cat == 'Non-WEIRD':
                    nonweird_aligns.append(al)
        w = np.mean(weird_aligns) if weird_aligns else 0
        nw = np.mean(nonweird_aligns) if nonweird_aligns else 0
        print(f"  {model}: WEIRD_align={w:.4f} NonWEIRD_align={nw:.4f}")

# ===== PHASE 18: LAS Behavioral =====
print("\n" + "="*70)
print("PHASE 18: LAS Behavioral Summary")
print("="*70)
for d in models_dirs:
    model = os.path.basename(d).replace('steering_vectors_', '').rsplit('_', 1)[0]
    f = os.path.join(d, 'phase18_las_behavioral', 'phase18_results.json')
    if os.path.exists(f):
        r = json.load(open(f))
        agg = r.get('aggregate', {})
        print(f"  {model}: raw_lift={agg.get('mean_raw_lift', 0):.4f} las_lift={agg.get('mean_las_lift', 0):.4f} las_advantage={agg.get('mean_las_advantage', 0):.4f}")
