"""
utils/geometry.py

Geometric analysis utilities for steering vector spaces.

Core operations:
- Subspace angles between steering directions
- Principal angle analysis (canonical angles between subspaces)
- Projection quality metrics
- Pivot language detection
- Steering Concept Drift (SCD) metric
"""

import numpy as np
from scipy.linalg import subspace_angles
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# Pairwise Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(1 - cosine(v1, v2))


def pairwise_cosine_matrix(vectors: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise cosine similarity matrix for a set of named vectors.

    Returns:
        (matrix of shape (n, n), ordered list of language codes)
    """
    langs = sorted(vectors.keys())
    n = len(langs)
    matrix = np.zeros((n, n))

    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            matrix[i, j] = cosine_similarity(vectors[l1], vectors[l2])

    return matrix, langs


def angular_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angular distance in radians between two vectors.
    Range: [0, π]. 0 = identical direction, π/2 = orthogonal.
    """
    sim = np.clip(cosine_similarity(v1, v2), -1.0, 1.0)
    return float(np.arccos(sim))


# ---------------------------------------------------------------------------
# Pivot Language Detection
# ---------------------------------------------------------------------------

def find_pivot_language(
    steering_vectors: dict[str, np.ndarray],
    method: str = "centrality",
) -> tuple[str, dict[str, float]]:
    """
    Identify the pivot language — the language whose steering direction
    is most central (most similar to all others on average).

    This operationalizes the "dominant internal language" hypothesis.

    Args:
        steering_vectors: {lang_code: steering_vector}
        method: "centrality" (mean similarity to all others)
                "projection" (best explains other vectors via projection)

    Returns:
        (pivot_lang_code, {lang: centrality_score})
    """
    langs = list(steering_vectors.keys())
    scores = {}

    if method == "centrality":
        for lang in langs:
            similarities = [
                cosine_similarity(steering_vectors[lang], steering_vectors[other])
                for other in langs if other != lang
            ]
            scores[lang] = float(np.mean(similarities))

    elif method == "projection":
        # Measure how well each language's vector serves as a "basis"
        # by projecting all others onto it and measuring residual variance
        for lang in langs:
            basis = steering_vectors[lang]
            residuals = []
            for other in langs:
                if other == lang:
                    continue
                v = steering_vectors[other]
                # Projection of v onto basis
                proj = (np.dot(v, basis) / (np.dot(basis, basis) + 1e-8)) * basis
                residual = np.linalg.norm(v - proj)
                residuals.append(residual)
            # Lower residual = better pivot (explains more variance)
            scores[lang] = float(-np.mean(residuals))  # negate so higher=better

    pivot = max(scores, key=scores.__getitem__)
    return pivot, scores


# ---------------------------------------------------------------------------
# Steering Concept Drift (SCD) Metric — Main Novel Contribution
# ---------------------------------------------------------------------------

def steering_concept_drift(
    steering_vectors: dict[str, np.ndarray],
    reference_lang: str = "en",
) -> dict[str, float]:
    """
    Compute the Steering Concept Drift (SCD) score for each language
    relative to the reference language.

    SCD(lang) = angular_distance(sv_lang, sv_reference)

    Range: [0, π/2]. 0 = identical concept geometry, π/2 = orthogonal.
    Higher SCD → greater concept drift → higher predicted safety failure rate.

    This is the paper's core metric. We show SCD correlates with:
    - Jailbreak success rates from PuneCon paper
    - Psycholinguistic distance from English
    - Language family distance
    """
    assert reference_lang in steering_vectors, \
        f"Reference language '{reference_lang}' not in steering vectors"

    ref_vec = steering_vectors[reference_lang]
    scd_scores = {}

    for lang, vec in steering_vectors.items():
        scd_scores[lang] = angular_distance(ref_vec, vec)

    return scd_scores


def scd_per_layer(
    all_layer_vectors: dict[str, dict[int, np.ndarray]],
    reference_lang: str = "en",
) -> dict[int, dict[str, float]]:
    """
    Compute SCD at each layer.

    Args:
        all_layer_vectors: {lang: {layer_idx: steering_vector}}

    Returns:
        {layer_idx: {lang: scd_score}}
    """
    # Collect all layer indices
    layers = sorted(next(iter(all_layer_vectors.values())).keys())
    result = {}

    for layer in layers:
        layer_vecs = {
            lang: all_layer_vectors[lang][layer]
            for lang in all_layer_vectors
        }
        result[layer] = steering_concept_drift(layer_vecs, reference_lang)

    return result


# ---------------------------------------------------------------------------
# Language Family Clustering
# ---------------------------------------------------------------------------

def language_family_alignment(
    steering_vectors: dict[str, np.ndarray],
    language_families: dict[str, str],
) -> dict[str, float]:
    """
    Measure within-family vs cross-family cosine similarities.
    Tests whether language families cluster in steering space —
    the computational analog of linguistic relativity structure.

    Returns:
        {family: mean_within_family_similarity}
    """
    families = {}
    for lang, fam in language_families.items():
        if lang in steering_vectors:
            if fam not in families:
                families[fam] = []
            families[fam].append(lang)

    family_scores = {}
    for fam, langs in families.items():
        if len(langs) < 2:
            continue
        sims = []
        for i, l1 in enumerate(langs):
            for l2 in langs[i+1:]:
                sims.append(cosine_similarity(
                    steering_vectors[l1], steering_vectors[l2]
                ))
        family_scores[fam] = float(np.mean(sims))

    return family_scores


# ---------------------------------------------------------------------------
# PCA / Subspace Analysis
# ---------------------------------------------------------------------------

def pca_steering_space(
    steering_vectors: dict[str, np.ndarray],
    n_components: int = 2,
) -> tuple[np.ndarray, list[str], PCA]:
    """
    PCA projection of steering vectors for visualization.

    Returns:
        (projections of shape (n_langs, n_components), lang_list, fitted_pca)
    """
    langs = sorted(steering_vectors.keys())
    matrix = np.stack([steering_vectors[l] for l in langs])  # (n_langs, hidden_dim)

    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(matrix)

    return projections, langs, pca


def principal_angle_to_english(
    steering_vectors: dict[str, np.ndarray],
    reference_lang: str = "en",
) -> dict[str, float]:
    """
    Canonical angle between each language's steering vector and the reference.
    Equivalent to SCD but phrased as the first principal angle between
    1D subspaces — useful for the paper's geometric exposition.
    """
    # For 1D subspaces (vectors), principal angle = arccos(|cos_sim|)
    ref = steering_vectors[reference_lang].reshape(1, -1)
    angles = {}

    for lang, vec in steering_vectors.items():
        v = vec.reshape(1, -1)
        # scipy subspace_angles returns angles in radians
        try:
            angle = subspace_angles(ref.T, v.T)[0]
        except Exception:
            # fallback
            angle = angular_distance(ref.flatten(), v.flatten())
        angles[lang] = float(angle)

    return angles


# ---------------------------------------------------------------------------
# Projection Quality (for LAS evaluation)
# ---------------------------------------------------------------------------

def projection_fidelity(
    source_vec: np.ndarray,
    target_vec: np.ndarray,
    rotated_vec: np.ndarray,
) -> dict[str, float]:
    """
    Measures how well the LAS rotation recovers the target steering direction.

    Returns:
        - improvement: cos_sim(rotated, target) - cos_sim(source, target)
        - fidelity: cos_sim(rotated, target)
        - residual_angle: angular_distance(rotated, target)
    """
    baseline = cosine_similarity(source_vec, target_vec)
    after = cosine_similarity(rotated_vec, target_vec)

    return {
        "baseline_similarity": float(baseline),
        "fidelity": float(after),
        "improvement": float(after - baseline),
        "residual_angle_rad": float(angular_distance(rotated_vec, target_vec)),
    }
