"""
outgroup_homogeneity_test.py

Sharpens Exp 17's monolithic-cluster finding into a specific, falsifiable
signature of the Outgroup Homogeneity Effect (Quattrone & Jones, 1980;
Linville & Jones, 1980), and cross-checks it against real-world cultural
distance rather than a bare within/between similarity ratio.

WHY THIS MATTERS FOR NOVELTY
-----------------------------
Your current Exp 17 test is binary: West vs. non-West, similarity=0.480 vs.
0.964. That's a real result, but it can't distinguish "the model collapses
non-Western morality into one blob" from "non-Western moral concepts happen
to be genuinely more similar to each other in the training distribution."
A skeptical reviewer can raise that second possibility, and you currently
have no way to rule it out.

Two existing lines of prior work you should cite and differentiate from:
  - Work assessing homogeneity in LLM-generated GROUP DESCRIPTIONS (a
    different object of study than internal safety-concept geometry).
  - Mechanistic work establishing a shared-but-different moral subspace
    across English and Chinese using Moral Foundations Theory (two
    languages, not a fine-grained within-outgroup diversity test).
Your contribution here is a fine-grained, causally-motivated test of whether
the model is BLIND TO REAL DIVERSITY within the "outgroup," which is the
actual outgroup homogeneity signature (cf. the finding that face-selective
cortex fails to distinguish different members of a racial outgroup, even
though those individuals are just as visually distinct as ingroup members).

THE TEST
--------
1. Get fine-grained items: (moral foundation x sub-culture) pairs, e.g.
   (Loyalty, Confucian), (Loyalty, Islamic), (Authority, Hindu), etc. -- NOT
   just one pooled "non-Western" bucket. You need at least ~3 distinct
   non-Western sub-cultures and ~3 distinct Western sub-cultures (e.g.
   Nordic-secular vs. Anglo vs. Mediterranean-Catholic) to make this test
   meaningful; two points can't show a correlation.

2. Get a REAL cultural-distance number for every pair of sub-cultures, from
   an external source you did not build (Hofstede dimension scores, World
   Values Survey coordinates, etc.) -- this is what keeps the test honest.
   A generic variance-normalized distance function (in the spirit of
   Kogut & Singh, 1988) is provided; you supply the actual per-culture
   dimension scores.

3. For each group (Western sub-cultures, non-Western sub-cultures)
   separately, Mantel-correlate the model's REPRESENTATIONAL distance matrix
   against the REAL cultural-distance matrix.
       - If the model tracks real diversity: Western group shows a
         significant positive Mantel r (more culturally distant Western
         sub-cultures also look more different to the model).
       - Outgroup homogeneity effect predicts: non-Western group shows a
         near-zero, non-significant Mantel r -- the model is blind to real
         differences among non-Western sub-cultures, flattening them into
         noise regardless of how culturally distant they actually are.

   NOTE: this is a DIFFERENT use of the Mantel test than your Exp 12. There,
   you correlated SCD against linguistic/typological distance across all 11
   languages pooled together, and got a null (r ~ -0.03 to -0.14), which you
   used to rule OUT linguistic distance as the driver. Here, you run the
   same style of test SEPARATELY within each cultural group, correlating
   representational distance against real CULTURAL (not linguistic)
   distance, to test within-group sensitivity to diversity. Keep these
   clearly distinct in the writeup -- same tool, different question.

4. Compare the two group-wise Mantel correlations directly (permutation test
   for the difference, plus a Fisher r-to-z cross-check).
"""

import numpy as np
from scipy import stats
from itertools import combinations

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
# An "item" is one (foundation, sub_culture) pairing with its own activation
# vector. Expected input format:
#
#   items = [
#       {"id": "loyalty_confucian",  "group": "non_west", "vector": np.array(...)},
#       {"id": "loyalty_islamic",    "group": "non_west", "vector": np.array(...)},
#       {"id": "authority_hindu",    "group": "non_west", "vector": np.array(...)},
#       {"id": "care_nordic",        "group": "west",     "vector": np.array(...)},
#       {"id": "fairness_anglo",     "group": "west",     "vector": np.array(...)},
#       ...
#   ]
#
# real_distances: dict mapping frozenset({id_i, id_j}) -> float real-world
# cultural distance between the two sub-cultures involved (symmetric,
# diagonal is implicitly 0). You compute this from an external cultural
# index (Hofstede, WVS, etc.) -- NOT from anything the model produced.


def load_real_items_and_distances():
    """
    *** PLUG POINT ***
    Replace with your real loader. `items` should reuse the SAME activation
    vectors you already extracted for Exp 17 -- just at finer sub-cultural
    granularity than the single pooled "non-Western" bucket you used there.
    `real_distances` must come from an independent cultural index, e.g.:

        hofstede_scores = {
            "confucian": {"PDI": 80, "IDV": 20, "MAS": 66, "UAI": 30, ...},
            "islamic_mena": {...},
            "hindu_south_asian": {...},
            "nordic": {...},
            "anglo": {...},
            "mediterranean_catholic": {...},
        }
        # then combine each culture's dimension vector into a distance,
        # e.g. via `kogut_singh_distance` below.
    """
    raise NotImplementedError("Plug in your real items + real-world distance loader.")


def kogut_singh_distance(dims_a, dims_b, dim_variances):
    """
    Variance-normalized composite distance across cultural dimensions, in the
    spirit of Kogut & Singh's (1988) cultural distance index: for each
    dimension, the squared difference is scaled by that dimension's variance
    across your full set of cultures (so no single dimension with an
    arbitrarily large numeric range dominates the distance), then averaged.

    dims_a, dims_b: dict of {dimension_name: score} for two cultures
    dim_variances: dict of {dimension_name: variance across ALL cultures in
                   your sample} -- compute this once up front over your full
                   set of sub-cultures, not just the pair being compared.

    If you're citing this directly, verify against the original Kogut &
    Singh (1988) formula for your writeup -- this implements the general
    variance-normalized-squared-difference logic, not a verbatim reproduction.
    """
    total = 0.0
    for dim, var in dim_variances.items():
        if var <= 0:
            continue
        total += (dims_a[dim] - dims_b[dim]) ** 2 / var
    return total / len(dim_variances)


def build_distance_matrix(pairwise_dict, ids):
    n = len(ids)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            key = frozenset((ids[i], ids[j]))
            d = pairwise_dict[key]
            mat[i, j] = d
            mat[j, i] = d
    return mat


# ---------------------------------------------------------------------------
# Representational distance
# ---------------------------------------------------------------------------
def cosine_distance_matrix(vectors):
    """1 - cosine similarity, for a list of vectors (same order as ids)."""
    V = np.array(vectors)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ V.T
    return 1 - sim


# ---------------------------------------------------------------------------
# Mantel test (permutation-based correlation between two distance matrices)
# ---------------------------------------------------------------------------
def mantel_test(dist_a, dist_b, n_perm=10000, seed=0):
    """
    Standard Mantel test: correlate the upper-triangle entries of two
    distance matrices, then build a null distribution by permuting the row/
    column labels of one matrix jointly (preserving its internal structure,
    only scrambling which item is which).
    """
    rng = np.random.default_rng(seed)
    n = dist_a.shape[0]
    iu = np.triu_indices(n, k=1)

    a_flat = dist_a[iu]
    b_flat = dist_b[iu]
    observed_r, _ = stats.pearsonr(a_flat, b_flat)

    null_rs = np.empty(n_perm)
    idx = np.arange(n)
    for p in range(n_perm):
        perm = rng.permutation(idx)
        b_perm = dist_b[np.ix_(perm, perm)][iu]
        null_rs[p] = stats.pearsonr(a_flat, b_perm)[0]

    p_value = (np.sum(np.abs(null_rs) >= np.abs(observed_r)) + 1) / (n_perm + 1)
    return observed_r, p_value, null_rs


# ---------------------------------------------------------------------------
# Comparing the two group-wise Mantel correlations
# ---------------------------------------------------------------------------
def fisher_r_to_z_test(r1, n1, r2, n2):
    """Parametric test for whether two independent correlation coefficients
    differ. n1/n2 are the number of ITEMS (not pairs) in each group; the
    Mantel test's effective df is smaller than n(n-1)/2 due to non-
    independence of pairs sharing an item, so treat this as a rough
    cross-check, not the primary test -- use the permutation test below as
    the main event."""
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def permutation_test_group_difference(rep_dist, real_dist, group_labels, n_perm=10000, seed=1):
    """
    Primary test for "does the West group track real cultural distance more
    than the non-West group does?" Pools all items, repeatedly reshuffles
    which items are labeled 'west' vs 'non_west' (preserving group sizes),
    recomputes both groups' Mantel r's, and builds a null distribution for
    the DIFFERENCE (r_west - r_nonwest). This avoids assuming independence
    of pairs the way the Fisher z-test does.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(group_labels)
    n_west = np.sum(labels == "west")

    def group_mantel_r(mask):
        idx = np.where(mask)[0]
        sub_rep = rep_dist[np.ix_(idx, idx)]
        sub_real = real_dist[np.ix_(idx, idx)]
        iu = np.triu_indices(len(idx), k=1)
        if len(iu[0]) < 3:
            return np.nan
        return stats.pearsonr(sub_rep[iu], sub_real[iu])[0]

    r_west = group_mantel_r(labels == "west")
    r_nonwest = group_mantel_r(labels == "non_west")
    observed_diff = r_west - r_nonwest

    n_total = len(labels)
    null_diffs = np.empty(n_perm)
    idx_all = np.arange(n_total)
    for p in range(n_perm):
        shuffled = rng.permutation(idx_all)
        west_idx = shuffled[:n_west]
        mask = np.zeros(n_total, dtype=bool)
        mask[west_idx] = True
        r_w = group_mantel_r(mask)
        r_nw = group_mantel_r(~mask)
        null_diffs[p] = r_w - r_nw

    p_value = (np.sum(np.abs(null_diffs) >= np.abs(observed_diff)) + 1) / (n_perm + 1)
    return r_west, r_nonwest, observed_diff, p_value, null_diffs


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_outgroup_homogeneity_test(items, real_distances, n_perm=10000):
    """
    items: list of {"id", "group" ("west"/"non_west"), "vector"} dicts
    real_distances: dict {frozenset({id_i, id_j}): float}
    """
    ids = [it["id"] for it in items]
    groups = [it["group"] for it in items]
    vectors = [it["vector"] for it in items]

    rep_dist = cosine_distance_matrix(vectors)
    real_dist = build_distance_matrix(real_distances, ids)

    print("=" * 70)
    print("Fine-Grained Outgroup Homogeneity Test (extends Exp 17)")
    print("=" * 70)
    n_west = groups.count("west")
    n_nonwest = groups.count("non_west")
    print(f"n items: {len(items)}  (west={n_west}, non_west={n_nonwest})")
    if n_west < 4 or n_nonwest < 4:
        print("WARNING: fewer than 4 sub-cultures per group. Mantel correlations")
        print("with this few points will be very noisy -- treat results as")
        print("exploratory only until you have more sub-cultures per group.")
    print()

    r_west, r_nonwest, diff, p_diff, _ = permutation_test_group_difference(
        rep_dist, real_dist, groups, n_perm=n_perm
    )
    z, p_fisher = fisher_r_to_z_test(r_west, n_west, r_nonwest, n_nonwest)

    print(f"Western sub-cultures    : Mantel r = {r_west:+.3f}")
    print(f"Non-Western sub-cultures: Mantel r = {r_nonwest:+.3f}")
    print(f"Difference (west - nonwest) = {diff:+.3f}")
    print(f"Permutation test on the difference: p = {p_diff:.4g}  (n_perm={n_perm}, primary test)")
    print(f"Fisher r-to-z cross-check:           p = {p_fisher:.4g}  (secondary, rough)")
    print()

    if r_west > 0.2 and (r_nonwest < 0.1) and p_diff < 0.05:
        print("-> Pattern consistent with OUTGROUP HOMOGENEITY:")
        print("   The model's internal geometry tracks real cultural diversity")
        print("   among Western sub-cultures, but is blind to just-as-real")
        print("   diversity among non-Western sub-cultures -- flattening a")
        print("   genuinely varied 'outgroup' into representational noise.")
    else:
        print("-> Pattern NOT clearly consistent with outgroup homogeneity.")
        print("   Either both groups track real distance comparably, or neither")
        print("   does -- re-examine whether your sub-culture split has enough")
        print("   points and enough real cultural spread to detect this.")
    print()
    return {"r_west": r_west, "r_nonwest": r_nonwest, "diff": diff,
            "p_permutation": p_diff, "p_fisher": p_fisher}


# ---------------------------------------------------------------------------
# Synthetic demo
# ---------------------------------------------------------------------------
def make_synthetic_items(dim=32, homogeneity_strength=0.8, seed=0):
    """
    Builds synthetic sub-cultures where Western items' representational
    distances DO track a synthetic real-world distance, while non-Western
    items' representational distances are increasingly decoupled from their
    (equally real, equally large) synthetic cultural distances as
    `homogeneity_strength` increases toward 1.0. Set to 0.0 for the null
    case (no homogeneity effect -- both groups track real distance equally).
    """
    rng = np.random.default_rng(seed)
    west_cultures = ["anglo", "nordic", "mediterranean", "germanic"]
    nonwest_cultures = ["confucian", "islamic_mena", "hindu_south_asian", "west_african", "andean"]

    # give every sub-culture a real-world coordinate in a small "true
    # cultural space" -- distances in this space are the "ground truth"
    true_coords = {c: rng.normal(size=4) for c in west_cultures + nonwest_cultures}

    items = []
    real_distances = {}
    all_cultures = west_cultures + nonwest_cultures
    shared_collapse_point = rng.normal(size=dim)
    shared_collapse_point /= np.linalg.norm(shared_collapse_point)

    culture_vectors = {}
    for c in all_cultures:
        own_direction = rng.normal(size=dim)
        own_direction /= np.linalg.norm(own_direction)
        if c in west_cultures:
            # Western reps preserve real distance: derived FROM true_coords
            proj = np.zeros(dim)
            proj[:4] = true_coords[c]
            v = proj + 0.3 * rng.normal(size=dim)
        else:
            # Non-Western reps get pulled toward one shared collapse point,
            # proportionally erasing their real distinctiveness
            proj = np.zeros(dim)
            proj[:4] = true_coords[c]
            v = ((1 - homogeneity_strength) * proj
                 + homogeneity_strength * shared_collapse_point * 3.0
                 + 0.3 * rng.normal(size=dim))
        culture_vectors[c] = v / np.linalg.norm(v)

    for c in all_cultures:
        group = "west" if c in west_cultures else "non_west"
        items.append({"id": c, "group": group, "vector": culture_vectors[c]})

    for c1, c2 in combinations(all_cultures, 2):
        real_d = np.linalg.norm(true_coords[c1] - true_coords[c2])
        real_distances[frozenset((c1, c2))] = real_d

    return items, real_distances


if __name__ == "__main__":
    for strength, label in [(0.85, "SIGNAL case (strong outgroup homogeneity)"),
                             (0.0, "NULL case (no homogeneity effect)")]:
        print("#" * 70)
        print(f"# {label}")
        print("#" * 70)
        items, real_distances = make_synthetic_items(homogeneity_strength=strength)
        run_outgroup_homogeneity_test(items, real_distances)
        print()
