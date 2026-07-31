"""
Comprehensive audit of ALL experiment results (Phases 1-10) across all 20 models.
Checks for: missing files, corrupt JSON, NaN values, empty results, 
suspicious statistics, and structural anomalies.
"""
import paramiko, json, sys, math
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=30)

def ssh_exec(cmd):
    try:
        stdin, stdout, stderr = client.exec_command(cmd, timeout=60)
        return stdout.read().decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"SSH_ERROR: {e}"

def is_nan(v):
    if v is None: return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return True
    if isinstance(v, str) and v.lower() in ('nan', 'inf', '-inf', 'none'): return True
    return False

# Get all completed model dirs
dirs_raw = ssh_exec("ls -d /root/clr_paper/results/steering_vectors_*/all_steering_vectors.pkl 2>/dev/null")
model_dirs = []
for line in dirs_raw.split("\n"):
    if line.strip() and not line.startswith("SSH_ERROR"):
        d = line.replace("/all_steering_vectors.pkl", "")
        name = d.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
        model_dirs.append((name, d))

print(f"{'='*70}")
print(f"COMPREHENSIVE AUDIT: {len(model_dirs)} MODELS x 10 EXPERIMENTS")
print(f"{'='*70}\n")

LANGUAGES = ['en', 'es', 'fr', 'pt', 'de', 'ru', 'ar', 'hi', 'ja', 'zh-CN', 'sw']
all_issues = []

for model, d in sorted(model_dirs):
    print(f"\n{'='*70}")
    print(f"MODEL: {model}")
    print(f"{'='*70}")
    
    # =========================================================
    # PHASE 1: Steering Vector Extraction
    # =========================================================
    sv_raw = ssh_exec(f"python3 -c \"import pickle; svs=pickle.load(open('{d}/all_steering_vectors.pkl','rb')); print(len(svs))\" 2>/dev/null")
    if sv_raw and not sv_raw.startswith("SSH_ERROR"):
        try:
            n_langs = int(sv_raw)
            if n_langs < 11:
                print(f"  Phase 1: PARTIAL - only {n_langs}/11 languages extracted")
                all_issues.append((model, "Phase 1", f"only {n_langs}/11 languages"))
            else:
                print(f"  Phase 1: OK ({n_langs} languages)")
        except:
            print(f"  Phase 1: CORRUPT pkl")
            all_issues.append((model, "Phase 1", "corrupt pkl"))
    else:
        print(f"  Phase 1: MISSING")
        all_issues.append((model, "Phase 1", "missing"))
    
    # =========================================================
    # PHASE 2: Pivot Language + SCD
    # =========================================================
    p2_raw = ssh_exec(f"cat {d}/phase2_pivot_analysis/phase2_results.json 2>/dev/null")
    if p2_raw and not p2_raw.startswith("SSH_ERROR"):
        try:
            p2 = json.loads(p2_raw)
            pivot = p2.get("pivot_language", "MISSING")
            scd = p2.get("scd_scores_degrees", {})
            n_scd = len(scd)
            
            # Check for NaN SCD values
            nan_scds = [l for l, v in scd.items() if is_nan(v)]
            # Check for suspiciously identical SCD values
            scd_vals = [v for v in scd.values() if not is_nan(v)]
            all_same = len(set(round(v, 2) for v in scd_vals)) <= 2 if scd_vals else False
            # Check for SCD > 90 (impossible for cosine-derived angle)
            extreme_scds = [l for l, v in scd.items() if not is_nan(v) and v > 90]
            
            issues_here = []
            if pivot == "MISSING" or pivot not in LANGUAGES:
                issues_here.append(f"invalid pivot={pivot}")
            if n_scd < 10:
                issues_here.append(f"only {n_scd}/10 SCD scores")
            if nan_scds:
                issues_here.append(f"NaN SCD for: {nan_scds}")
            if all_same:
                issues_here.append(f"all SCD scores nearly identical ({scd_vals[:3]}...)")
            if extreme_scds:
                issues_here.append(f"SCD>90deg for: {extreme_scds}")
            
            if issues_here:
                print(f"  Phase 2: ISSUES - {'; '.join(issues_here)}")
                for iss in issues_here:
                    all_issues.append((model, "Phase 2", iss))
            else:
                print(f"  Phase 2: OK (pivot={pivot}, {n_scd} SCDs, range={min(scd_vals):.1f}-{max(scd_vals):.1f})")
        except json.JSONDecodeError:
            print(f"  Phase 2: CORRUPT JSON")
            all_issues.append((model, "Phase 2", "corrupt JSON"))
    else:
        print(f"  Phase 2: MISSING")
        all_issues.append((model, "Phase 2", "missing"))
    
    # =========================================================
    # PHASE 3: Per-language steering vectors (checked via phase2 SCD)
    # =========================================================
    lang_dirs = ssh_exec(f"ls -d {d}/en {d}/es {d}/fr {d}/pt {d}/de {d}/ru {d}/ar {d}/hi {d}/ja {d}/zh-CN {d}/sw 2>/dev/null | wc -l")
    try:
        n_lang_dirs = int(lang_dirs)
        if n_lang_dirs < 11:
            print(f"  Phase 3: PARTIAL - {n_lang_dirs}/11 language directories")
            all_issues.append((model, "Phase 3", f"only {n_lang_dirs}/11 lang dirs"))
        else:
            print(f"  Phase 3: OK ({n_lang_dirs} language dirs)")
    except:
        print(f"  Phase 3: UNKNOWN")
    
    # =========================================================
    # PHASE 4: Psycholinguistic Correlation
    # =========================================================
    p4_raw = ssh_exec(f"cat {d}/phase4_psycholinguistic/phase4_results.json 2>/dev/null")
    if p4_raw and not p4_raw.startswith("SSH_ERROR"):
        try:
            p4 = json.loads(p4_raw)
            rom_sim = p4.get("romance_similarity")
            germ_sim = p4.get("germanic_similarity")
            rho = p4.get("psycholinguistic_rho")
            loo_r2 = p4.get("loo_r2")
            boot_r2 = p4.get("bootstrap_r2")
            cohen_d = p4.get("cohens_d")
            
            issues_here = []
            if is_nan(rom_sim): issues_here.append("NaN romance_sim")
            if is_nan(germ_sim): issues_here.append("NaN germanic_sim")
            if is_nan(rho): issues_here.append("NaN psycholinguistic_rho")
            if is_nan(loo_r2): issues_here.append("NaN loo_r2")
            if is_nan(boot_r2): issues_here.append("NaN bootstrap_r2")
            if is_nan(cohen_d): issues_here.append("NaN cohens_d")
            if not is_nan(loo_r2) and loo_r2 < -5:
                issues_here.append(f"extreme negative LOO R2={loo_r2:.3f}")
            if not is_nan(cohen_d) and abs(cohen_d) > 3:
                issues_here.append(f"extreme Cohen's d={cohen_d:.3f}")
            
            if issues_here:
                print(f"  Phase 4: ISSUES - {'; '.join(issues_here)}")
                for iss in issues_here:
                    all_issues.append((model, "Phase 4", iss))
            else:
                print(f"  Phase 4: OK (rom={rom_sim:.3f}, germ={germ_sim:.3f}, rho={rho:.3f}, d={cohen_d:.3f})")
        except json.JSONDecodeError:
            print(f"  Phase 4: CORRUPT JSON")
            all_issues.append((model, "Phase 4", "corrupt JSON"))
    else:
        print(f"  Phase 4: MISSING")
        all_issues.append((model, "Phase 4", "missing"))
    
    # =========================================================
    # PHASE 5: LAS
    # =========================================================
    p5_files = ssh_exec(f"ls {d}/phase5_las/ 2>/dev/null")
    if p5_files and not p5_files.startswith("SSH_ERROR"):
        flist = [f for f in p5_files.split("\n") if f.strip()]
        print(f"  Phase 5: OK ({len(flist)} files: {', '.join(flist[:3])}...)")
    else:
        print(f"  Phase 5: MISSING")
        all_issues.append((model, "Phase 5", "missing"))
    
    # =========================================================
    # PHASE 6: Jailbreak simulation
    # =========================================================
    p6_raw = ssh_exec(f"cat {d}/phase6_jailbreak/phase6_results.json 2>/dev/null")
    if p6_raw and not p6_raw.startswith("SSH_ERROR"):
        try:
            p6 = json.loads(p6_raw)
            n_langs = len(p6.get("languages", {}))
            print(f"  Phase 6: OK ({n_langs} languages)")
        except:
            print(f"  Phase 6: CORRUPT JSON")
            all_issues.append((model, "Phase 6", "corrupt JSON"))
    else:
        # Phase 6 might just be a directory with figures
        p6_dir = ssh_exec(f"ls {d}/phase6_jailbreak/ 2>/dev/null | wc -l")
        try:
            n = int(p6_dir)
            if n > 0:
                print(f"  Phase 6: PARTIAL ({n} files, no JSON)")
            else:
                print(f"  Phase 6: MISSING")
                all_issues.append((model, "Phase 6", "missing"))
        except:
            print(f"  Phase 6: MISSING")
            all_issues.append((model, "Phase 6", "missing"))
    
    # =========================================================
    # PHASE 7: Code-mixed subversion
    # =========================================================
    p7_raw = ssh_exec(f"cat {d}/phase7_codemix/phase7_results.json 2>/dev/null")
    if p7_raw and not p7_raw.startswith("SSH_ERROR"):
        try:
            p7 = json.loads(p7_raw)
            n_langs = len(p7.get("languages", {}))
            print(f"  Phase 7: OK ({n_langs} languages)")
        except:
            print(f"  Phase 7: CORRUPT JSON")
            all_issues.append((model, "Phase 7", "corrupt JSON"))
    else:
        p7_dir = ssh_exec(f"ls {d}/phase7_codemix/ 2>/dev/null | wc -l")
        try:
            n = int(p7_dir)
            if n > 0:
                print(f"  Phase 7: PARTIAL ({n} files, no JSON)")
            else:
                print(f"  Phase 7: MISSING")
                all_issues.append((model, "Phase 7", "missing"))
        except:
            print(f"  Phase 7: MISSING")
            all_issues.append((model, "Phase 7", "missing"))
    
    # =========================================================
    # PHASE 8: Causal Intervention
    # =========================================================
    p8_raw = ssh_exec(f"cat {d}/phase8_causal_intervention/phase8_results.json 2>/dev/null")
    if p8_raw and not p8_raw.startswith("SSH_ERROR"):
        try:
            p8 = json.loads(p8_raw)
            n_langs = len(p8.get("languages", {}))
            best_layer = p8.get("best_layer", "?")
            corr = p8.get("scd_vs_lift_correlation", {})
            r_val = corr.get("pearson_r")
            p_val = corr.get("pearson_p")
            
            issues_here = []
            if n_langs == 0:
                issues_here.append("0 languages processed (EMPTY)")
            if is_nan(r_val):
                issues_here.append("NaN pearson_r")
            if is_nan(p_val):
                issues_here.append("NaN pearson_p")
            
            # Check per-language data quality
            for lang, ldata in p8.get("languages", {}).items():
                baseline = ldata.get("baseline_refusal_rate")
                best_steered = ldata.get("best_steered_refusal_rate")
                if is_nan(baseline): issues_here.append(f"{lang}: NaN baseline")
                if is_nan(best_steered): issues_here.append(f"{lang}: NaN steered rate")
                # Check if all alphas produced 0 refusals (generation may have failed)
                alphas = ldata.get("alpha_results", {})
                all_zero = all(a.get("refused", 0) == 0 and a.get("total", 0) == 0 for a in alphas.values()) if alphas else True
                if all_zero and alphas:
                    issues_here.append(f"{lang}: all alphas have 0/0 results (generation failed)")
            
            if issues_here:
                print(f"  Phase 8: ISSUES - {'; '.join(issues_here[:5])}")
                for iss in issues_here[:5]:
                    all_issues.append((model, "Phase 8", iss))
            else:
                lift_info = f"r={r_val:.3f}, p={p_val:.4f}" if not is_nan(r_val) else "no correlation"
                print(f"  Phase 8: OK ({n_langs} langs, layer={best_layer}, {lift_info})")
        except json.JSONDecodeError:
            print(f"  Phase 8: CORRUPT JSON")
            all_issues.append((model, "Phase 8", "corrupt JSON"))
    else:
        print(f"  Phase 8: MISSING")
        all_issues.append((model, "Phase 8", "missing"))
    
    # =========================================================
    # PHASE 10: Adversarial Probing
    # =========================================================
    p10_raw = ssh_exec(f"cat {d}/phase10_adversarial_probing/phase10_results.json 2>/dev/null")
    if p10_raw and not p10_raw.startswith("SSH_ERROR"):
        try:
            p10 = json.loads(p10_raw)
            n_langs = len(p10.get("languages", {}))
            corr = p10.get("scd_vs_asr", {})
            r_val = corr.get("pearson_r")
            p_val = corr.get("pearson_p")
            
            issues_here = []
            if n_langs == 0:
                issues_here.append("0 languages (EMPTY)")
            if is_nan(r_val):
                issues_here.append(f"NaN pearson_r (correlation failed)")
            
            # Check per-language ASRs
            for lang, ldata in p10.get("languages", {}).items():
                overall_asr = ldata.get("overall_asr")
                if is_nan(overall_asr):
                    issues_here.append(f"{lang}: NaN ASR")
                attacks = ldata.get("attacks", {})
                for atk_name, atk_data in attacks.items():
                    asr = atk_data.get("asr")
                    total = atk_data.get("total", 0)
                    if total == 0:
                        issues_here.append(f"{lang}/{atk_name}: 0 total prompts")
            
            if issues_here:
                print(f"  Phase 10: ISSUES - {'; '.join(issues_here[:5])}")
                for iss in issues_here[:5]:
                    all_issues.append((model, "Phase 10", iss))
            else:
                r_str = f"r={r_val:.3f}" if not is_nan(r_val) else "no corr"
                print(f"  Phase 10: OK ({n_langs} langs, SCD-ASR {r_str})")
        except json.JSONDecodeError:
            print(f"  Phase 10: CORRUPT JSON")
            all_issues.append((model, "Phase 10", "corrupt JSON"))
    else:
        print(f"  Phase 10: MISSING")
        all_issues.append((model, "Phase 10", "missing"))

# =========================================================
# PHASE 9: Cross-Model Transfer (global, not per-model)
# =========================================================
print(f"\n{'='*70}")
print(f"PHASE 9: Cross-Model Transfer (global)")
print(f"{'='*70}")
p9_raw = ssh_exec("cat /root/clr_paper/results/phase9_cross_model_transfer/phase9_results.json 2>/dev/null")
if p9_raw and not p9_raw.startswith("SSH_ERROR"):
    try:
        p9 = json.loads(p9_raw)
        n_models = p9.get("n_models", 0)
        within = p9.get("within_family_sim", {})
        cross = p9.get("cross_family_sim", {})
        mann_p = p9.get("mann_whitney_p")
        
        issues_here = []
        if n_models < 20:
            issues_here.append(f"only {n_models}/20 models")
        if is_nan(within.get("mean")):
            issues_here.append("NaN within-family mean")
        if is_nan(cross.get("mean")):
            issues_here.append("NaN cross-family mean")
        if is_nan(mann_p):
            issues_here.append("NaN Mann-Whitney p-value")
        
        # Check per-language transfer
        per_lang = p9.get("per_language_transfer", {})
        for lang, stats in per_lang.items():
            if is_nan(stats.get("mean")):
                issues_here.append(f"{lang}: NaN transfer mean")
        
        if issues_here:
            print(f"  ISSUES - {'; '.join(issues_here)}")
            for iss in issues_here:
                all_issues.append(("GLOBAL", "Phase 9", iss))
        else:
            print(f"  OK ({n_models} models, within={within.get('mean',0):.3f}, cross={cross.get('mean',0):.3f}, p={mann_p:.6f})")
    except json.JSONDecodeError:
        print(f"  CORRUPT JSON")
        all_issues.append(("GLOBAL", "Phase 9", "corrupt JSON"))
else:
    print(f"  MISSING")
    all_issues.append(("GLOBAL", "Phase 9", "missing"))

# =========================================================
# SUMMARY
# =========================================================
print(f"\n\n{'#'*70}")
print(f"ISSUE SUMMARY: {len(all_issues)} total issues found")
print(f"{'#'*70}")

# Group by severity
critical = [(m, p, i) for m, p, i in all_issues if "EMPTY" in i or "missing" in i.lower() or "corrupt" in i.lower() or "generation failed" in i.lower()]
warnings = [(m, p, i) for m, p, i in all_issues if "NaN" in i]
anomalies = [(m, p, i) for m, p, i in all_issues if (m, p, i) not in critical and (m, p, i) not in warnings]

if critical:
    print(f"\n--- CRITICAL ({len(critical)}) ---")
    for m, p, i in sorted(critical):
        print(f"  {m:40s} | {p:10s} | {i}")

if warnings:
    print(f"\n--- WARNINGS ({len(warnings)}) ---")
    for m, p, i in sorted(warnings):
        print(f"  {m:40s} | {p:10s} | {i}")

if anomalies:
    print(f"\n--- ANOMALIES ({len(anomalies)}) ---")
    for m, p, i in sorted(anomalies):
        print(f"  {m:40s} | {p:10s} | {i}")

if not all_issues:
    print("\n  ALL CLEAN! No issues detected across any experiments.")

client.close()
print("\nAudit complete.")
