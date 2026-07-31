import json, glob, os

print('# Phase 6: Safety Prediction from SCD')
print('| Model | Pearson r | P-value | LOO-CV R² (SCD) | Bootstrap R² |')
print('|---|---|---|---|---|')
files = glob.glob('/root/clr_paper/results/steering_vectors_*/phase6_safety_prediction/phase6_results.json')
for f in files:
    model = os.path.basename(os.path.dirname(os.path.dirname(f))).replace('steering_vectors_', '').rsplit('_', 1)[0]
    with open(f) as fp:
        try:
            d = json.load(fp)
            corr = d.get('scd_jailbreak_correlation', {})
            reg = d.get('regression', {})
            print(f"| {model} | {corr.get('pearson_r', 0):.3f} | {corr.get('p_value', 0):.4f} | {reg.get('loo_r2_scd', 0):.3f} | {reg.get('bootstrap_r2_scd_mean', 0):.3f} |")
        except:
            pass

print('\n# Phase 7: Code-Mix Instability')
print('| Model | CM-EN Dist | CM-HI Dist | EN-HI Dist | Interp Dist |')
print('|---|---|---|---|---|')
files7 = glob.glob('/root/clr_paper/results/steering_vectors_*/phase7_codemix/phase7_results.json')
for f in files7:
    model = os.path.basename(os.path.dirname(os.path.dirname(f))).replace('steering_vectors_', '').rsplit('_', 1)[0]
    with open(f) as fp:
        try:
            d = json.load(fp)
            geo = d.get('geometric_position', {})
            if geo:
                print(f"| {model} | {geo.get('cm_en_degrees', 0):.1f}° | {geo.get('cm_hi_degrees', 0):.1f}° | {geo.get('en_hi_degrees', 0):.1f}° | {geo.get('interpolation_distance_degrees', 0):.1f}° |")
        except:
            pass
