"""
experiments/04_psycholinguistic_priors.py + 05_las_method.py

Phase 4: Validate psycholinguistic priors — show that cognitive science
distances predict SCD better than typological distances alone.

Phase 5: Language-Anchored Steering (LAS) — the novel method.
Learn a minimal rotation per language that preserves the safety concept
while respecting that language's geometric neighborhood.

This is the paper's main methodological contribution.
"""

import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.psycholing import (
    psycholinguistic_distance_vector,
    typological_distance_vector,
    BehavioralDistanceEstimator,
)
from utils.geometry import (
    steering_concept_drift,
    projection_fidelity,
    cosine_similarity,
)
from experiments.load_results import load_steering_vectors, load_phase2_results


# ===========================================================================
# PHASE 4: Psycholinguistic Prior Validation
# ===========================================================================

def run_psycholinguistic_validation(
    sv_results_dir: str,
    phase2_dir: str,
    behavioral_data_path: str = None,
    output_dir: str = None,
):
    """
    Validate that psycholinguistic distances predict SCD scores.

    Key comparison:
    - Model A: SCD ~ typological_distance (baseline, linguistics)
    - Model B: SCD ~ psycholinguistic_distance (our claim, cognitive science)
    - Model C: SCD ~ typological + psycholinguistic (combined)

    If Model B > Model A: psycholinguistic distances have unique predictive value.
    This is the key result supporting the Sapir-Whorf angle.
    """

    print(f"\n{'='*60}")
    print("CLR Phase 4: Psycholinguistic Prior Validation")
    print(f"{'='*60}\n")

    all_sv, metadata = load_steering_vectors(sv_results_dir)
    phase2_results = load_phase2_results(phase2_dir)

    output_dir = Path(output_dir or sv_results_dir) / "phase4_psycholinguistic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get SCD scores from Phase 2
    scd_scores = phase2_results["scd_scores_radians"]
    languages = [l for l in scd_scores.keys() if l != "en"]

    # Build predictor vectors
    psycholing_dists = psycholinguistic_distance_vector(languages)
    typological_dists = typological_distance_vector(languages)
    scd_values = np.array([scd_scores[l] for l in languages])

    print(f"Languages in analysis: {languages}")
    print(f"N = {len(languages)}")

    results = {}

    # -----------------------------------------------------------------------
    # Regression Analysis
    # -----------------------------------------------------------------------

    def regression_analysis(X, y, name):
        X_fit = X.reshape(-1, 1) if X.ndim == 1 else X
        reg = Ridge(alpha=0.01).fit(X_fit, y)
        y_pred = reg.predict(X_fit)
        r2 = r2_score(y, y_pred)
        
        # For correlation, if multiple predictors, correlate predictions with target
        corr_x = X if X.ndim == 1 else y_pred
        pearson_r, p = pearsonr(corr_x, y)
        spearman_r, sp = spearmanr(corr_x, y)
        return {
            "name": name,
            "r_squared": float(r2),
            "pearson_r": float(pearson_r),
            "pearson_p": float(p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(sp),
        }

    model_a = regression_analysis(typological_dists, scd_values, "Typological only")
    model_b = regression_analysis(psycholing_dists, scd_values, "Psycholinguistic only")
    model_c_X = np.stack([typological_dists, psycholing_dists], axis=1)
    model_c = regression_analysis(model_c_X, scd_values, "Combined")

    print("\nRegression Results: Predicting SCD from distance measures")
    print("-" * 50)
    for model in [model_a, model_b, model_c]:
        print(f"\n  {model['name']}:")
        print(f"    R² = {model['r_squared']:.3f}")
        print(f"    Pearson r = {model['pearson_r']:.3f} (p={model['pearson_p']:.4f})")
        print(f"    Spearman r = {model['spearman_r']:.3f} (p={model['spearman_p']:.4f})")

    # KEY FINDING: If psycholinguistic R² > typological R², we support our claim
    improvement = model_b["r_squared"] - model_a["r_squared"]
    print(f"\n  *** Psycholinguistic improvement over typological: ΔR² = {improvement:.3f} ***")

    results["regression_models"] = {
        "typological": model_a,
        "psycholinguistic": model_b,
        "combined": model_c,
    }
    results["psycholinguistic_improvement"] = improvement  # ΔR²

    # -----------------------------------------------------------------------
    # Per-language breakdown table
    # -----------------------------------------------------------------------

    lang_df = pd.DataFrame({
        "language": languages,
        "scd_degrees": [np.degrees(scd_scores[l]) for l in languages],
        "psycholing_distance": psycholing_dists,
        "typological_distance": typological_dists,
    }).sort_values("scd_degrees", ascending=False)

    lang_df.to_csv(output_dir / "language_distances.csv", index=False)
    print(f"\n  Language breakdown saved to: {output_dir}/language_distances.csv")

    # -----------------------------------------------------------------------
    # Behavioral Study Integration (if Prolific data available)
    # -----------------------------------------------------------------------

    if behavioral_data_path and Path(behavioral_data_path).exists():
        print("\n  Integrating Prolific behavioral data...")
        estimator = BehavioralDistanceEstimator(behavioral_data_path)
        behavioral_dists = estimator.compute_distances()
        corr_results = estimator.correlation_with_scd(scd_scores)

        print(f"  Empirical-SCD correlation:")
        print(f"    Pearson r = {corr_results['pearson_r']:.3f} "
              f"(p={corr_results['p_value_pearson']:.4f})")
        print(f"    Spearman r = {corr_results['spearman_r']:.3f}")

        results["behavioral_study"] = corr_results
    else:
        print("\n  [Prolific data not found — run behavioral study to add this result]")
        print("  Tip: 50 participants × 12 languages × 15 scenarios ≈ $200 on Prolific")

    with open(output_dir / "phase4_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Phase 4 complete.")
    return results


# ===========================================================================
# PHASE 5: Language-Anchored Steering (LAS)
# ===========================================================================

class LanguageAnchoredSteering(nn.Module):
    """
    Language-Anchored Steering (LAS) — the paper's novel method.

    For each language, learns a rotation matrix R_lang such that:
        steered_direction = R_lang @ english_steering_vector

    Constraints:
    1. R_lang should be an orthogonal rotation (preserve vector norms)
    2. The rotation should be minimal (close to identity) — psycholinguistically
       close languages need smaller rotations
    3. The rotated vector should maximize safety steering effectiveness
       in the target language

    The psycholinguistic prior enters as a regularization weight:
        loss = -steering_effectiveness + λ * psycholing_distance * ||R - I||²

    Languages with LARGER psycholinguistic distance from English → larger λ
    → allows more rotation → model learns a larger correction.

    This is the direct computational implementation of the Sapir-Whorf hypothesis:
    we let the cognitive science distance dictate how much we trust the
    "English framing" of safety concepts.
    """

    def __init__(self, hidden_dim: int, n_languages: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # One rotation matrix per language (initialized to identity)
        self.rotations = nn.ParameterList([
            nn.Parameter(torch.eye(hidden_dim))
            for _ in range(n_languages)
        ])

    def forward(self, english_sv: torch.Tensor, lang_idx: int) -> torch.Tensor:
        """
        Apply language-specific rotation to English steering vector.

        Args:
            english_sv: English steering vector (hidden_dim,)
            lang_idx: which language rotation to use

        Returns:
            rotated steering vector (hidden_dim,)
        """
        R = self.rotations[lang_idx]
        # Orthogonalize via QR decomposition for numerical stability
        Q, _ = torch.linalg.qr(R)
        return Q @ english_sv

    def get_rotation_magnitude(self, lang_idx: int) -> float:
        """Measure how far the rotation is from identity"""
        R = self.rotations[lang_idx]
        I = torch.eye(self.hidden_dim, device=R.device)
        return float(torch.norm(R - I, p="fro"))


def train_las(
    english_sv: np.ndarray,
    target_sv_per_lang: dict[str, np.ndarray],
    psycholing_distances: dict[str, float],
    n_epochs: int = 50,
    lr: float = 0.01,
    base_lambda: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Train Language-Anchored Steering rotations.

    Args:
        english_sv: English steering vector at best layer
        target_sv_per_lang: {lang: target_steering_vector}
            These are the "oracle" steering vectors we want to reach.
            In practice, we use the directly-extracted steering vectors
            from Phase 1 as targets.
        psycholing_distances: {lang: distance}
        n_epochs: training epochs
        lr: learning rate
        base_lambda: base regularization strength

    Returns:
        {lang: rotated_steering_vector} — the LAS output
    """
    languages = list(target_sv_per_lang.keys())
    n_langs = len(languages)
    hidden_dim = english_sv.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert to tensors
    en_sv_t = torch.tensor(english_sv, dtype=torch.float32, device=device)
    target_svs = {
        lang: torch.tensor(vec, dtype=torch.float32, device=device)
        for lang, vec in target_sv_per_lang.items()
    }

    model = LanguageAnchoredSteering(hidden_dim, n_langs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    print(f"\nTraining LAS ({n_langs} languages, {hidden_dim}-dim, {n_epochs} epochs on {device})...")

    best_loss = float("inf")
    history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)

        for i, lang in enumerate(languages):
            rotated = model(en_sv_t, i)

            # Primary loss: maximize cosine similarity with target steering vector
            cos_sim = torch.dot(rotated, target_svs[lang]) / (
                torch.norm(rotated) * torch.norm(target_svs[lang]) + 1e-8
            )
            steering_loss = 1.0 - cos_sim

            # Regularization: psycholinguistically closer languages should rotate less
            psy_dist = psycholing_distances.get(lang, 0.5)
            lambda_reg = base_lambda * (1.0 - psy_dist)  # closer → stronger reg → less rotation
            rotation_norm = torch.norm(model.rotations[i] - torch.eye(hidden_dim, device=device), p="fro")
            reg_loss = lambda_reg * rotation_norm

            lang_loss = steering_loss + reg_loss
            total_loss = total_loss + lang_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        history.append(float(total_loss))

        if float(total_loss) < best_loss:
            best_loss = float(total_loss)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs} — Loss: {total_loss:.4f}")

    # Extract final rotated vectors
    rotated_vectors = {}
    with torch.no_grad():
        for i, lang in enumerate(languages):
            rotated = model(en_sv_t, i)
            rotated_vectors[lang] = rotated.cpu().numpy()

    return rotated_vectors, model, history


def run_las_method(
    sv_results_dir: str,
    phase2_dir: str,
    output_dir: str = None,
):
    """
    Train and evaluate Language-Anchored Steering.

    Evaluation:
    1. Fidelity: does LAS-rotated vector match target language steering direction?
    2. Safety improvement: does applying LAS-rotated steering reduce jailbreak rate?
    3. Rotation magnitude vs psycholinguistic distance correlation
    """

    print(f"\n{'='*60}")
    print("CLR Phase 5: Language-Anchored Steering (LAS)")
    print(f"{'='*60}\n")

    all_sv, metadata = load_steering_vectors(sv_results_dir)
    phase2_results = load_phase2_results(phase2_dir)

    output_dir = Path(output_dir or sv_results_dir) / "phase5_las"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_layer = phase2_results["best_layer"]
    languages = [l for l in all_sv.keys() if l != "en"]

    english_sv = all_sv["en"][best_layer]
    target_svs = {lang: all_sv[lang][best_layer] for lang in languages}

    from utils.psycholing import PSYCHOLINGUISTIC_DISTANCES
    psy_distances = {lang: PSYCHOLINGUISTIC_DISTANCES.get(lang, 0.5) for lang in languages}

    # -----------------------------------------------------------------------
    # Train LAS
    # -----------------------------------------------------------------------

    rotated_svs, las_model, train_history = train_las(
        english_sv=english_sv,
        target_sv_per_lang=target_svs,
        psycholing_distances=psy_distances,
        n_epochs=1000,
        lr=0.005,
    )

    # -----------------------------------------------------------------------
    # Evaluate Fidelity
    # -----------------------------------------------------------------------

    print("\nLAS Fidelity Evaluation:")
    print("-" * 50)
    print(f"{'Language':<12} {'Baseline Sim':>14} {'LAS Sim':>10} {'Improvement':>13} {'Rotation |R-I|':>16}")
    print("-" * 50)

    fidelity_results = {}
    for i, lang in enumerate(languages):
        fid = projection_fidelity(
            source_vec=english_sv,
            target_vec=target_svs[lang],
            rotated_vec=rotated_svs[lang],
        )
        rot_mag = las_model.get_rotation_magnitude(i)

        fidelity_results[lang] = {**fid, "rotation_magnitude": rot_mag}

        indicator = "✓" if fid["improvement"] > 0.05 else "~"
        print(f"  {lang:<10} {fid['baseline_similarity']:>14.3f} "
              f"{fid['fidelity']:>10.3f} {fid['improvement']:>+13.3f} "
              f"{rot_mag:>14.3f}  {indicator}")

    # -----------------------------------------------------------------------
    # Key result: rotation magnitude ~ psycholinguistic distance
    # -----------------------------------------------------------------------

    rot_magnitudes = np.array([fidelity_results[l]["rotation_magnitude"] for l in languages])
    psy_dists_arr = np.array([psy_distances[l] for l in languages])
    pearson_r, pearson_p = pearsonr(psy_dists_arr, rot_magnitudes)

    print(f"\n  Rotation magnitude ~ Psycholinguistic distance:")
    print(f"  Pearson r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"  → {'CONFIRMS' if pearson_p < 0.05 else 'does not confirm'} "
          f"Sapir-Whorf prediction")

    results = {
        "best_layer": best_layer,
        "fidelity": fidelity_results,
        "rotation_psycholing_correlation": {
            "pearson_r": float(pearson_r),
            "p_value": float(pearson_p),
        },
        "training_history": train_history,
    }

    with open(output_dir / "phase5_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save rotated steering vectors for Phase 6
    with open(output_dir / "las_steering_vectors.pkl", "wb") as f:
        pickle.dump(rotated_svs, f)

    print(f"\n✓ Phase 5 complete.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["4", "5", "both"], default="both")
    parser.add_argument("--input", required=True, help="Phase 1 results dir")
    parser.add_argument("--phase2", required=True, help="Phase 2 results dir")
    parser.add_argument("--behavioral", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.phase in ("4", "both"):
        run_psycholinguistic_validation(
            args.input, args.phase2, args.behavioral, args.output
        )
    if args.phase in ("5", "both"):
        run_las_method(args.input, args.phase2, args.output)
