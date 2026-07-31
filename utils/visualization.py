"""
utils/visualization.py

All paper figures. Run after all experiment phases complete.

Figure map:
    Figure 1: PCA of steering space — language family clustering
    Figure 2: SCD score progression across transformer layers
    Figure 3: SCD vs Jailbreak rate scatter (main result)
    Figure 4: LAS rotation magnitudes vs psycholinguistic distance
    Figure 5: Code-mixed instability diagram
    Figure 6: Conceptual diagram — Sapir-Whorf analog
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Optional


# ── Style ────────────────────────────────────────────────────────────────────
FAMILY_COLORS = {
    "Germanic":     "#2563EB",
    "Romance":      "#16A34A",
    "Slavic":       "#9333EA",
    "Semitic":      "#EA580C",
    "Indo-Aryan":   "#0891B2",
    "Dravidian":    "#BE185D",
    "Sino-Tibetan": "#CA8A04",
    "Turkic":       "#65A30D",
    "Bantu":        "#78350F",
    "Austronesian": "#0F766E",
    "Austroasiatic":"#4338CA",
    "Koreanic":     "#B45309",
    "Japonic":      "#7C3AED",
    "Iranian":      "#1D4ED8",
    "Unknown":      "#6B7280",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def _save(fig, output_dir: Path, name: str):
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(str(path).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Figure 1: PCA Steering Space ─────────────────────────────────────────────

def fig1_steering_pca(
    pca_df: pd.DataFrame,        # columns: PC1, PC2, language, family, scd_score
    output_dir: Path,
    highlight_lang: str = "en",
):
    """
    Main visual: languages in 2D PCA of steering space.
    Color = language family, size = SCD score (larger = more drift).
    This is Figure 1 of the paper — the "money shot".
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in pca_df.iterrows():
        color = FAMILY_COLORS.get(row["family"], "#6B7280")
        size = 80 + row["scd_score"] * 300  # scale by SCD
        marker = "*" if row["language"] == highlight_lang else "o"
        ax.scatter(row["PC1"], row["PC2"], c=color, s=size,
                   marker=marker, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(
            row["language"],
            (row["PC1"], row["PC2"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=9, fontweight="bold" if row["language"] == highlight_lang else "normal",
        )

    # Draw convex hulls for language families (optional visual grouping)
    from scipy.spatial import ConvexHull
    for family, color in FAMILY_COLORS.items():
        subset = pca_df[pca_df["family"] == family]
        if len(subset) >= 3:
            try:
                points = subset[["PC1", "PC2"]].values
                hull = ConvexHull(points)
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1],
                            color=color, alpha=0.25, linewidth=1)
            except Exception:
                pass

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=family)
        for family, color in FAMILY_COLORS.items()
        if family in pca_df["family"].values
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              title="Language Family", title_fontsize=8)

    ax.set_xlabel("PC1 (Steering Direction Variance)", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_title("Steering Vector Space: Language Family Clustering\n"
                 "(★ = English, size ∝ Steering Concept Drift)",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="#E5E7EB", linewidth=0.5)
    ax.axvline(0, color="#E5E7EB", linewidth=0.5)

    _save(fig, output_dir, "fig1_steering_pca.pdf")


# ── Figure 2: SCD Across Layers ───────────────────────────────────────────────

def fig2_scd_layers(
    scd_layer_df: pd.DataFrame,   # index = layer, columns = languages
    output_dir: Path,
    highlight_langs: list = None,
):
    """
    SCD score progression across transformer layers.
    Shows that safety concept drift is LAYER-DEPENDENT
    and emerges most strongly in middle layers.
    """
    if highlight_langs is None:
        highlight_langs = ["hi", "ar", "ta", "fr", "de"]

    fig, ax = plt.subplots(figsize=(9, 5))
    layers = scd_layer_df.index.tolist()

    for lang in scd_layer_df.columns:
        if lang == "en":
            continue
        values = np.degrees(scd_layer_df[lang].values)
        alpha = 0.9 if lang in highlight_langs else 0.25
        lw = 2.0 if lang in highlight_langs else 0.8
        ax.plot(layers, values, alpha=alpha, linewidth=lw, label=lang if lang in highlight_langs else None)

    ax.set_xlabel("Transformer Layer", fontsize=10)
    ax.set_ylabel("Steering Concept Drift (degrees)", fontsize=10)
    ax.set_title("Safety Concept Drift Across Transformer Layers\n"
                 "(Higher = greater deviation from English safety representation)",
                 fontsize=11, fontweight="bold")

    # Shade "middle layers" region
    n_layers = len(layers)
    mid_start, mid_end = layers[n_layers//4], layers[3*n_layers//4]
    ax.axvspan(mid_start, mid_end, alpha=0.08, color="#2563EB",
               label="Middle layers\n(semantic zone)")

    ax.legend(fontsize=9, loc="upper left")
    _save(fig, output_dir, "fig2_scd_layers.pdf")


# ── Figure 3: SCD vs Jailbreak Rate ──────────────────────────────────────────

def fig3_scd_jailbreak(
    prediction_df: pd.DataFrame,  # columns: language, scd_degrees, jailbreak_rate, family
    output_dir: Path,
):
    """
    The main result figure: SCD predicts jailbreak success rate.
    Scatter plot with regression line and language labels.
    """
    from scipy.stats import pearsonr
    from sklearn.linear_model import LinearRegression

    fig, ax = plt.subplots(figsize=(8, 6))

    x = prediction_df["scd_degrees"].values
    y = prediction_df["jailbreak_rate"].values * 100  # percentage

    # Regression line
    reg = LinearRegression().fit(x.reshape(-1,1), y)
    x_line = np.linspace(x.min() - 2, x.max() + 2, 100)
    y_line = reg.predict(x_line.reshape(-1,1))
    r, p = pearsonr(x, y)
    ax.plot(x_line, y_line, color="#2563EB", linewidth=2, alpha=0.7,
            label=f"Linear fit (r={r:.2f}, p={p:.3f})")

    # 95% CI band
    from scipy import stats
    n = len(x)
    se = np.sqrt(np.sum((y - reg.predict(x.reshape(-1,1)))**2) / (n-2))
    t_val = stats.t.ppf(0.975, df=n-2)
    x_mean = np.mean(x)
    ci = t_val * se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.12, color="#2563EB")

    # Scatter
    for _, row in prediction_df.iterrows():
        color = FAMILY_COLORS.get(row.get("family", "Unknown"), "#6B7280")
        ax.scatter(row["scd_degrees"], row["jailbreak_rate"] * 100,
                   color=color, s=90, zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(
            row["language"],
            (row["scd_degrees"], row["jailbreak_rate"] * 100),
            xytext=(5, 2), textcoords="offset points", fontsize=9,
        )

    ax.set_xlabel("Steering Concept Drift — SCD (degrees)", fontsize=11)
    ax.set_ylabel("Jailbreak Success Rate (%)", fontsize=11)
    ax.set_title("SCD Predicts Multilingual Safety Failures\n"
                 "(Mechanistic geometry → behavioral outcome)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    _save(fig, output_dir, "fig3_scd_jailbreak.pdf")


# ── Figure 4: LAS Rotation Magnitudes ────────────────────────────────────────

def fig4_las_rotations(
    fidelity_df: pd.DataFrame,  # columns: language, rotation_magnitude, psycholing_distance
    output_dir: Path,
):
    """
    LAS rotation magnitude vs psycholinguistic distance.
    Key result: languages cognitively closer to English need smaller rotations.
    This directly validates the Sapir-Whorf computational analog.
    """
    from scipy.stats import pearsonr

    fig, ax = plt.subplots(figsize=(7, 5))

    x = fidelity_df["psycholing_distance"].values
    y = fidelity_df["rotation_magnitude"].values
    r, p = pearsonr(x, y)

    ax.scatter(x, y, s=80, color="#7C3AED", alpha=0.8, edgecolors="white", zorder=3)

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x.reshape(-1,1), y)
    x_line = np.linspace(x.min()-0.02, x.max()+0.02, 100)
    ax.plot(x_line, reg.predict(x_line.reshape(-1,1)), color="#7C3AED",
            linewidth=2, alpha=0.7, label=f"r={r:.2f}, p={p:.3f}")

    for _, row in fidelity_df.iterrows():
        ax.annotate(row["language"], (row["psycholing_distance"], row["rotation_magnitude"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Psycholinguistic Distance from English (harm domain)", fontsize=11)
    ax.set_ylabel("LAS Rotation Magnitude ‖R − I‖_F", fontsize=11)
    ax.set_title("Psycholinguistically Distant Languages Need Larger Steering Corrections\n"
                 "(Computational Sapir-Whorf: distance → required rotation)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, output_dir, "fig4_las_rotations.pdf")


# ── Figure 5: Cosine Similarity Heatmap ──────────────────────────────────────

def fig5_similarity_heatmap(
    sim_matrix: np.ndarray,
    lang_order: list,
    family_map: dict,
    output_dir: Path,
):
    """
    Pairwise cosine similarity heatmap with family annotations.
    Shows that within-family similarities are higher (clustering).
    """
    # Sort by family for cleaner visualization
    families = [family_map.get(l, "Unknown") for l in lang_order]
    sort_idx = sorted(range(len(lang_order)), key=lambda i: (families[i], lang_order[i]))
    sorted_langs = [lang_order[i] for i in sort_idx]
    sorted_matrix = sim_matrix[np.ix_(sort_idx, sort_idx)]

    fig, ax = plt.subplots(figsize=(9, 8))
    mask = np.eye(len(sorted_langs), dtype=bool)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        sorted_matrix, ax=ax,
        xticklabels=sorted_langs, yticklabels=sorted_langs,
        cmap="RdYlGn", center=0.5, vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 8},
        mask=mask, square=True, linewidths=0.3,
    )

    ax.set_title("Pairwise Cosine Similarity of Safety Steering Vectors\n"
                 "(Sorted by language family — within-family blocks visible)",
                 fontsize=11, fontweight="bold")

    _save(fig, output_dir, "fig5_similarity_heatmap.pdf")


# ── Main runner ───────────────────────────────────────────────────────────────

def generate_all_figures(results_base_dir: str, output_dir: str = None):
    """
    Generate all paper figures from saved experiment results.
    Call after all 7 phases have completed.
    """
    results_base = Path(results_base_dir)
    output_dir = Path(output_dir or results_base / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating all paper figures...")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load PCA data (Phase 2)
    pca_path = results_base / "phase2_pivot_analysis" / "pca_steering_space.csv"
    if pca_path.exists():
        pca_df = pd.read_csv(pca_path, index_col=0)
        pca_df["language"] = pca_df.index
        fig1_steering_pca(pca_df, output_dir)

    # Load SCD layer data (Phase 2)
    scd_path = results_base / "phase2_pivot_analysis" / "scd_per_layer.csv"
    if scd_path.exists():
        scd_df = pd.read_csv(scd_path, index_col=0)
        fig2_scd_layers(scd_df, output_dir)

    # Load safety prediction data (Phase 6)
    pred_path = results_base / "phase6_safety_prediction" / "safety_prediction_table.csv"
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        fig3_scd_jailbreak(pred_df, output_dir)

    # LAS fidelity data (Phase 5)
    phase5_path = results_base / "phase5_las" / "phase5_results.json"
    if phase5_path.exists():
        import json
        with open(phase5_path) as f:
            phase5 = json.load(f)
        fid_data = [
            {"language": lang, **vals}
            for lang, vals in phase5["fidelity"].items()
        ]
        fid_df = pd.DataFrame(fid_data)
        from utils.psycholing import PSYCHOLINGUISTIC_DISTANCES
        fid_df["psycholing_distance"] = fid_df["language"].map(PSYCHOLINGUISTIC_DISTANCES)
        fig4_las_rotations(fid_df, output_dir)

    # Similarity matrix (Phase 2)
    sim_path = results_base / "phase2_pivot_analysis" / "cosine_similarity_matrix.csv"
    p2_path = results_base / "phase2_pivot_analysis" / "phase2_results.json"
    if sim_path.exists() and p2_path.exists():
        sim_df = pd.read_csv(sim_path, index_col=0)
        with open(p2_path) as f:
            p2 = json.load(f)
        from utils.languages import LANGUAGE_FAMILIES
        fig5_similarity_heatmap(
            sim_df.values, list(sim_df.columns), LANGUAGE_FAMILIES, output_dir
        )

    print(f"\n✓ All figures saved to: {output_dir}")
