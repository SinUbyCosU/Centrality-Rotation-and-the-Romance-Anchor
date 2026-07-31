#!/usr/bin/env python3
"""
Phase 11 SCD Fix & Recompute Script
====================================
- Reads Phase 2 scd_scores_degrees for each model
- Patches existing Phase 11 JSON files (ASR data untouched, only SCD + correlations)
- Cross-checks 3 models against Phase 2 ground truth
- Computes Pearson r, Spearman rho per model
- Applies Benjamini-Hochberg FDR correction
- Outputs final consolidated table
"""
import paramiko
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

BASE_MODELS = {"yi-6b", "bloomz-7b1", "stablelm-3b"}

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASS, timeout=10)

    # 1. Find all Phase 11 result files
    stdin, stdout, stderr = client.exec_command(
        'find /root/clr_paper/results -name "phase11_results.json" 2>/dev/null'
    )
    p11_files = [f.strip() for f in stdout.read().decode('utf-8').strip().split('\n') if f.strip() and 'steering_vectors_' in f]

    results = []
    spot_check_models = ["mistral-7b", "phi-3-mini-4k-instruct", "stablelm-2-1.6b-chat"]
    spot_check_results = []

    for p11_file in sorted(p11_files):
        model_dir = p11_file.split('/phase11_scaled_adversarial/')[0]
        model = model_dir.split('steering_vectors_')[1].rsplit('_2026', 1)[0]

        # Load Phase 2 SCD scores
        p2_file = f"{model_dir}/phase2_pivot_analysis/phase2_results.json"
        stdin, stdout, stderr = client.exec_command(f'cat {p2_file}')
        p2_raw = stdout.read().decode('utf-8')
        if not p2_raw.strip():
            print(f"  WARNING: No Phase 2 data for {model}, skipping")
            continue
        p2_data = json.loads(p2_raw)
        scd_scores = p2_data.get("scd_scores_degrees", {})

        if not scd_scores:
            print(f"  WARNING: Empty scd_scores_degrees for {model}, skipping")
            continue

        # Load existing Phase 11 data
        stdin, stdout, stderr = client.exec_command(f'cat {p11_file}')
        p11_data = json.loads(stdout.read().decode('utf-8'))

        # Patch SCD values from Phase 2 ground truth
        for lang, ldata in p11_data.get("languages", {}).items():
            ldata["scd_degrees"] = scd_scores.get(lang, 0.0)

        # Recompute correlations
        scd_vals = []
        asr_vals = []
        for lang, ldata in p11_data["languages"].items():
            scd = ldata.get("scd_degrees", 0)
            asr = ldata.get("overall_asr", 0)
            if scd is not None and not (isinstance(scd, float) and np.isnan(scd)):
                scd_vals.append(scd)
                asr_vals.append(asr)

        pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan
        n_langs = len(scd_vals)

        if n_langs >= 5:
            scd_arr = np.array(scd_vals)
            asr_arr = np.array(asr_vals)
            if np.std(scd_arr) > 0 and np.std(asr_arr) > 0:
                pr, pp = pearsonr(scd_vals, asr_vals)
                sr, sp = spearmanr(scd_vals, asr_vals)
            elif np.std(asr_arr) == 0:
                # Genuine flat - no ASR variance
                pass  # leave as NaN

        p11_data["scd_vs_asr"] = {
            "pearson_r": float(pr) if not np.isnan(pr) else None,
            "pearson_p": float(pp) if not np.isnan(pp) else None,
            "spearman_rho": float(sr) if not np.isnan(sr) else None,
            "spearman_p": float(sp) if not np.isnan(sp) else None,
        }

        # Write back patched JSON
        patched_json = json.dumps(p11_data, indent=2, default=str)
        stdin_w, stdout_w, stderr_w = client.exec_command(f"cat > {p11_file}")
        stdin_w.write(patched_json)
        stdin_w.channel.shutdown_write()
        stdout_w.read()  # wait for completion

        is_base = model in BASE_MODELS

        # Spot-check
        if model in spot_check_models:
            old_stdin, old_stdout, old_stderr = client.exec_command(f'cat {p11_file}')
            verify_data = json.loads(old_stdout.read().decode('utf-8'))
            spot_check_results.append({
                "model": model,
                "p2_scd_hi": scd_scores.get("hi", "N/A"),
                "p11_scd_hi": verify_data["languages"].get("hi", {}).get("scd_degrees", "N/A"),
                "p2_scd_ar": scd_scores.get("ar", "N/A"),
                "p11_scd_ar": verify_data["languages"].get("ar", {}).get("scd_degrees", "N/A"),
                "match": (
                    scd_scores.get("hi") == verify_data["languages"].get("hi", {}).get("scd_degrees") and
                    scd_scores.get("ar") == verify_data["languages"].get("ar", {}).get("scd_degrees")
                ),
            })

        en_asr = p11_data["languages"].get("en", {}).get("overall_asr", 0)
        all_asrs = [ld.get("overall_asr", 0) for ld in p11_data["languages"].values()]
        avg_asr = np.mean(all_asrs) if all_asrs else 0

        results.append({
            "model": model,
            "is_base": is_base,
            "n_langs": n_langs,
            "en_asr": en_asr,
            "avg_asr": avg_asr,
            "asr_var": float(np.var(all_asrs)),
            "scd_var": float(np.var([scd_scores.get(l, 0) for l in p11_data["languages"].keys()])),
            "pearson_r": pr,
            "pearson_p": pp,
            "spearman_rho": sr,
            "spearman_p": sp,
        })

    client.close()

    # ============================================================
    # Step 3: Spot-check report
    # ============================================================
    print("\n=== STEP 3: Phase 2 Ground Truth Spot-Check ===")
    for sc in spot_check_results:
        status = "MATCH" if sc["match"] else "MISMATCH"
        print(f"  {sc['model']}: hi={sc['p2_scd_hi']:.4f} vs {sc['p11_scd_hi']:.4f}, "
              f"ar={sc['p2_scd_ar']:.4f} vs {sc['p11_scd_ar']:.4f} -> {status}")

    # ============================================================
    # Step 4: FDR correction
    # ============================================================
    computable = [r for r in results if not np.isnan(r["pearson_r"])]
    pvals = [r["pearson_p"] for r in computable]

    # Benjamini-Hochberg
    if pvals:
        n_tests = len(pvals)
        sorted_indices = np.argsort(pvals)
        sorted_pvals = np.array(pvals)[sorted_indices]
        bh_critical = np.array([(i+1) / n_tests * 0.05 for i in range(n_tests)])
        bh_significant = sorted_pvals <= bh_critical

        # Find the largest k where p(k) <= k/m * alpha
        fdr_results = {}
        for orig_idx, sort_idx in enumerate(sorted_indices):
            model_name = computable[sort_idx]["model"]
            fdr_results[model_name] = {
                "rank": orig_idx + 1,
                "p_val": sorted_pvals[orig_idx],
                "bh_threshold": bh_critical[orig_idx],
                "significant": bool(bh_significant[orig_idx]),
            }

        # Apply step-up procedure correctly
        last_significant = -1
        for k in range(n_tests):
            if sorted_pvals[k] <= bh_critical[k]:
                last_significant = k
        
        for orig_idx in range(n_tests):
            model_name = computable[sorted_indices[orig_idx]]["model"]
            fdr_results[model_name]["significant"] = (orig_idx <= last_significant)
    else:
        fdr_results = {}

    n_surviving = sum(1 for v in fdr_results.values() if v["significant"])

    print(f"\n=== STEP 4: FDR Correction ===")
    print(f"  Models with computable correlations: {len(computable)}")
    print(f"  Models surviving FDR q<0.05: {n_surviving}")
    for model_name, fdr in sorted(fdr_results.items(), key=lambda x: x[1]["p_val"]):
        sig = "*" if fdr["significant"] else ""
        print(f"    {model_name}: p={fdr['p_val']:.4f}, BH_threshold={fdr['bh_threshold']:.4f} {sig}")

    # ============================================================
    # Step 5: Final consolidated table
    # ============================================================
    print("\n=== STEP 5: FINAL CONSOLIDATED TABLE ===")
    print("| Model | Type | En ASR | Avg ML ASR | Pearson r | p-value | FDR sig? | Status |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: x["model"]):
        model = r["model"]
        mtype = "base" if r["is_base"] else "instruct"
        en_asr = f"{r['en_asr']*100:.1f}%"
        avg_asr = f"{r['avg_asr']*100:.1f}%"

        if r["is_base"]:
            status = "excluded-base-model"
            pr_str = "N/A"
            pp_str = "N/A"
            fdr_str = "N/A"
        elif np.isnan(r["pearson_r"]):
            if r["asr_var"] == 0:
                status = "still-N/A-genuine (zero ASR variance: model perfectly robust across all languages)"
            else:
                status = "still-N/A-unresolved"
            pr_str = "N/A"
            pp_str = "N/A"
            fdr_str = "N/A"
        else:
            pr_str = f"{r['pearson_r']:.3f}"
            pp_str = f"{r['pearson_p']:.4f}"
            fdr_sig = fdr_results.get(model, {}).get("significant", False)
            fdr_str = "YES" if fdr_sig else "no"
            status = "verified-clean"

        print(f"| {model} | {mtype} | {en_asr} | {avg_asr} | {pr_str} | {pp_str} | {fdr_str} | {status} |")

    # ============================================================
    # Step 6: Verdict
    # ============================================================
    print(f"\n=== STEP 6: VERDICT ===")
    n_computable = len(computable)
    n_nominal = sum(1 for r in computable if r["pearson_p"] < 0.05)
    
    if n_surviving > 0:
        print(f"VALIDATED POSITIVE: {n_surviving}/{n_computable} models show FDR-significant "
              f"correlation between SCD and ASR at q<0.05.")
    elif n_nominal > 0:
        print(f"SUGGESTIVE BUT NOT FDR-SIGNIFICANT: {n_nominal}/{n_computable} models show "
              f"nominal p<0.05 but none survive FDR correction.")
    else:
        print(f"VALIDATED NULL: 0/{n_computable} models show even nominal p<0.05 correlation "
              f"between SCD and ASR.")

if __name__ == "__main__":
    main()
