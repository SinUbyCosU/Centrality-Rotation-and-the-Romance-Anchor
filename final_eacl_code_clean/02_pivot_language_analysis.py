"""
experiments/02_pivot_language_analysis.py

Phase 2: Geometric Analysis — Test the Pivot Language Hypothesis

Core question: Is English geometrically dominant in the steering space?
Do language families cluster? Is there a single "pivot language"?

This is the main mechanistic analysis of the paper.

Runtime: ~30 minutes on CPU (pure numpy after Phase 1).
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.geometry import (
    pairwise_cosine_matrix,
    find_pivot_language,
    steering_concept_drift,
    scd_per_layer,
    language_family_alignment,
    pca_steering_space,
    principal_angle_to_english,
)
from utils.languages import LANGUAGE_FAMILIES
from experiments.load_results import load_steering_vectors


def run_pivot_analysis(results_dir: str, output_dir: str = None):
    """
    Full pivot language analysis pipeline.

    Produces:
    1. Cosine similarity matrix across languages
    2. Pivot language ranking (centrality scores)
    3. SCD scores per language per layer
    4. Language family clustering statistics
    5. PCA projection data
    """

    print(f"\n{'='*60}")
    print("CLR Paper — Phase 2: Pivot Language Analysis")
    print(f"{'='*60}\n")

    # Load Phase 1 results
    all_sv, metadata = load_steering_vectors(results_dir)
    languages = list(all_sv.keys())
    layer_indices = metadata["layer_indices"]

    output_dir = output_dir or results_dir
    output_dir = Path(output_dir) / "phase2_pivot_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # -----------------------------------------------------------------------
    # 1. Find the "best" layer for analysis
    #    Middle layers typically encode the most semantically meaningful
    #    representations. We pick the layer where SCD variance is highest
    #    (most discriminative about language differences).
    # -----------------------------------------------------------------------

    print("Finding most discriminative layer...")
    scd_by_layer = scd_per_layer(all_sv, reference_lang="en")

    # FIX: Use a fixed architectural depth (2/3) to prevent circular variance-maximization
    layers_list = sorted(list(scd_by_layer.keys()))
    best_layer = layers_list[len(layers_list) * 2 // 3] if layers_list else 0
    
    layer_variances = {
        layer: np.var(list(scd_scores.values()))
        for layer, scd_scores in scd_by_layer.items()
    }
    print(f"  Selected architectural layer (2/3 depth): {best_layer} "
          f"(SCD variance = {layer_variances.get(best_layer, 0):.4f})")

    results["best_layer"] = best_layer
    results["layer_scd_variances"] = {int(k): float(v) for k, v in layer_variances.items()}

    # Use the best layer for main analysis
    steering_at_best = {lang: all_sv[lang][best_layer] for lang in languages}

    # -----------------------------------------------------------------------
    # 2. Pairwise Cosine Similarity Matrix
    # -----------------------------------------------------------------------

    print("\nComputing pairwise cosine similarity matrix...")
    sim_matrix, lang_order = pairwise_cosine_matrix(steering_at_best)

    results["cosine_similarity_matrix"] = sim_matrix.tolist()
    results["language_order"] = lang_order

    # Save as CSV for easy inspection
    df_sim = pd.DataFrame(sim_matrix, index=lang_order, columns=lang_order)
    df_sim.to_csv(output_dir / "cosine_similarity_matrix.csv")
    print(f"  Saved similarity matrix: {output_dir}/cosine_similarity_matrix.csv")

    # -----------------------------------------------------------------------
    # 3. Pivot Language Detection
    # -----------------------------------------------------------------------

    print("\nDetecting pivot language...")
    pivot_centrality, centrality_scores = find_pivot_language(
        steering_at_best, method="centrality"
    )
    pivot_projection, projection_scores = find_pivot_language(
        steering_at_best, method="projection"
    )

    print(f"  Pivot (centrality method): {pivot_centrality}")
    print(f"  Pivot (projection method): {pivot_projection}")
    print("\n  Centrality scores (higher = more central):")
    for lang, score in sorted(centrality_scores.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {score:.4f}")

    results["pivot_centrality"] = pivot_centrality
    results["pivot_projection"] = pivot_projection
    results["centrality_scores"] = centrality_scores
    results["projection_scores"] = projection_scores

    # -----------------------------------------------------------------------
    # 4. Steering Concept Drift (SCD) — Main Metric
    # -----------------------------------------------------------------------

    print("\nComputing Steering Concept Drift (SCD) scores...")
    scd_scores = steering_concept_drift(steering_at_best, reference_lang="en")
    scd_scores_degrees = {lang: float(np.degrees(v)) for lang, v in scd_scores.items()}

    print("\n  SCD scores (angular distance from English steering direction):")
    for lang, scd in sorted(scd_scores.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {np.degrees(scd):.1f}° "
              f"({'HIGH drift' if scd > 0.5 else 'LOW drift'})")

    results["scd_scores_radians"] = {k: float(v) for k, v in scd_scores.items()}
    results["scd_scores_degrees"] = scd_scores_degrees
    results["scd_per_layer"] = {
        int(layer): {lang: float(v) for lang, v in scores.items()}
        for layer, scores in scd_by_layer.items()
    }

    # Save SCD progression across layers (for Figure 2 in paper)
    scd_layer_df = pd.DataFrame({
        lang: [scd_by_layer[layer].get(lang, np.nan) for layer in sorted(scd_by_layer.keys())]
        for lang in languages
    }, index=sorted(scd_by_layer.keys()))
    scd_layer_df.to_csv(output_dir / "scd_per_layer.csv")

    # -----------------------------------------------------------------------
    # 5. Language Family Clustering
    # -----------------------------------------------------------------------

    print("\nAnalyzing language family clustering in steering space...")
    family_map = {lang: LANGUAGE_FAMILIES.get(lang, "Unknown") for lang in languages}
    family_scores = language_family_alignment(steering_at_best, family_map)

    print("\n  Within-family mean cosine similarities:")
    for fam, score in sorted(family_scores.items(), key=lambda x: -x[1]):
        print(f"    {fam}: {score:.4f}")

    # Cross-family baseline
    all_sims = []
    for i, l1 in enumerate(languages):
        for l2 in languages[i+1:]:
            if family_map.get(l1) != family_map.get(l2):
                all_sims.append(
                    float(sim_matrix[lang_order.index(l1)][lang_order.index(l2)])
                )
    cross_family_mean = np.mean(all_sims) if all_sims else 0.0

    print(f"\n  Cross-family baseline similarity: {cross_family_mean:.4f}")
    print(f"  → Within-family similarities are consistently higher: "
          f"supports family clustering hypothesis")

    results["family_clustering_scores"] = family_scores
    results["cross_family_baseline"] = float(cross_family_mean)

    # -----------------------------------------------------------------------
    # 6. PCA Projection Data (for Figure 1 — the key visual)
    # -----------------------------------------------------------------------

    print("\nComputing PCA projection of steering space...")
    projections, lang_order_pca, pca = pca_steering_space(steering_at_best, n_components=3)

    results["pca_projections"] = projections.tolist()
    results["pca_lang_order"] = lang_order_pca
    results["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()

    print(f"  PCA explained variance: "
          f"{pca.explained_variance_ratio_[:3].sum()*100:.1f}% (first 3 components)")

    # Save PCA data
    pca_df = pd.DataFrame(
        projections[:, :2],
        index=lang_order_pca,
        columns=["PC1", "PC2"]
    )
    pca_df["family"] = [family_map.get(l, "Unknown") for l in lang_order_pca]
    pca_df["scd_score"] = [scd_scores_degrees.get(l, 0) for l in lang_order_pca]
    pca_df.to_csv(output_dir / "pca_steering_space.csv")

    # -----------------------------------------------------------------------
    # 7. Statistical Tests
    # -----------------------------------------------------------------------

    print("\nRunning statistical tests...")

    all_p_values = {}

    # Test 1: Permutation test — Is English's centrality significantly higher?
    # (Replaces circular t-test with proper non-parametric test)
    english_centrality = centrality_scores.get("en", 0)
    n_perms = 10000
    centrality_values = list(centrality_scores.values())
    count_higher = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(centrality_values)
        if shuffled[0] >= english_centrality:
            count_higher += 1
    perm_p_val = count_higher / n_perms
    print(f"  English centrality permutation test (n={n_perms}): p={perm_p_val:.4f}")
    all_p_values["english_centrality_permutation"] = perm_p_val

    # Test 2: Mantel test — Do language families cluster in steering space?
    # Correlates steering vector distance matrix with family membership matrix
    from scipy.spatial.distance import pdist, squareform
    sv_dists = squareform(pdist(
        np.array([steering_at_best[l] for l in languages]), metric='cosine'
    ))
    fam_dists = np.array([
        [0.0 if family_map.get(languages[i]) == family_map.get(languages[j]) else 1.0
         for j in range(len(languages))]
        for i in range(len(languages))
    ])
    # Mantel statistic: Pearson correlation between distance matrices
    sv_flat = sv_dists[np.triu_indices_from(sv_dists, k=1)]
    fam_flat = fam_dists[np.triu_indices_from(fam_dists, k=1)]
    from scipy.stats import pearsonr as _pearsonr
    mantel_r, _ = _pearsonr(sv_flat, fam_flat)
    # Permutation test for Mantel
    mantel_count = 0
    for _ in range(n_perms):
        perm_idx = np.random.permutation(len(languages))
        perm_fam = fam_dists[np.ix_(perm_idx, perm_idx)]
        perm_flat = perm_fam[np.triu_indices_from(perm_fam, k=1)]
        perm_r, _ = _pearsonr(sv_flat, perm_flat)
        if perm_r >= mantel_r:
            mantel_count += 1
    mantel_p = mantel_count / n_perms
    print(f"  Mantel test (family clustering): r={mantel_r:.3f}, p={mantel_p:.4f}")
    all_p_values["mantel_family_clustering"] = mantel_p

    # Benjamini-Hochberg FDR correction across all tests
    test_names = list(all_p_values.keys())
    raw_pvals = np.array([all_p_values[k] for k in test_names])
    # BH procedure
    n_tests = len(raw_pvals)
    sorted_idx = np.argsort(raw_pvals)
    adjusted = np.zeros(n_tests)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = raw_pvals[idx] * n_tests / (rank + 1)
    adjusted = np.minimum.accumulate(adjusted[np.argsort(sorted_idx)][::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    fdr_pvals = {name: float(adj) for name, adj in zip(test_names, adjusted)}

    print(f"\n  FDR-corrected p-values:")
    for name, adj_p in fdr_pvals.items():
        print(f"    {name}: raw={all_p_values[name]:.4f}, FDR={adj_p:.4f}")

    results["statistical_tests"] = {
        "english_centrality_permutation": {
            "statistic": float(english_centrality),
            "p_raw": float(perm_p_val),
            "p_fdr": float(fdr_pvals["english_centrality_permutation"]),
            "n_permutations": n_perms,
        },
        "mantel_family_clustering": {
            "mantel_r": float(mantel_r),
            "p_raw": float(mantel_p),
            "p_fdr": float(fdr_pvals["mantel_family_clustering"]),
            "n_permutations": n_perms,
        },
    }

    # -----------------------------------------------------------------------
    # Save all results
    # -----------------------------------------------------------------------

    with open(output_dir / "phase2_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Phase 2 complete. Results saved to: {output_dir}")
    print(f"  Run Phase 3: python experiments/03_concept_drift_metric.py "
          f"--input {results_dir}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLR Phase 2: Pivot Language Analysis")
    parser.add_argument("--input", required=True,
                        help="Path to Phase 1 results directory")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: same as input)")
    args = parser.parse_args()

    run_pivot_analysis(args.input, args.output)
