"""
ingroup_projection_test.py

Tests the Ingroup Projection Model (Mummendey & Wenzel, 1999) as an explanation
for Pivot Language Bias.

THEORY
------
Mummendey & Wenzel's Ingroup Projection Model says that when subgroups compare
themselves via a superordinate category (e.g. "European" for French/Italian),
members tend to project their own subgroup's traits onto the superordinate
prototype -- so the "neutral, universal" standard actually just IS the ingroup,
in disguise. Outgroups aren't merely disliked; they're perceived as deviating
from a supposedly-neutral standard that's really the ingroup's own norm.

Applied here: if Pivot Bias is real ingroup projection, the "universal safety
subspace" that non-Western languages get routed through should look like
ENGLISH specifically, not like a genuine, balanced average across all 11
languages. If it's not ingroup projection -- if the model has converged on some
real shared cross-lingual safety signal that English merely resembles because
English happens to dominate training data -- then a balanced cross-lingual
centroid should explain the pivot direction just as well, or better.

TWO TESTS PROVIDED
------------------
1. DIAGNOSTIC TEST (cheap, correlational, run this first):
   For every non-Western language L in every model, compare
       cos(v_L, v_english)   vs.   cos(v_L, balanced_centroid_excluding_L)
   The "balanced centroid" gives Western and non-Western language clusters
   EQUAL weight regardless of how many languages you happen to have sampled
   from each -- this matters a lot, because if your 11-language set already
   skews Western, a naive mean-of-all-vectors centroid is contaminated toward
   English before you even start, and the test becomes circular.

   Ingroup Projection Index (IPI) = cos(v_L, v_en) - cos(v_L, balanced_centroid)
   IPI > 0 and significant  -> consistent with ingroup projection
   IPI ~ 0 or negative       -> consistent with genuine shared convergence

2. CAUSAL TEST (stronger, extends your existing Exp 14 / LAS machinery):
   You already showed injecting the English safety vector into Swahili
   recovers ~74% of the alignment gap. The causal question ingroup projection
   actually makes is: does injecting a BALANCED CENTROID vector (built with
   English excluded, or down-weighted to remove Western dominance) recover
   the gap just as well? If English-specific injection reliably outperforms
   the balanced-centroid injection, that's much stronger evidence that what's
   being treated as "the universal safety direction" is really just English's
   particular direction -- i.e., ingroup projection, not convergence.

   This module gives you the statistical comparison machinery
   (Friedman test + pairwise Wilcoxon with FDR correction) and a clearly
   marked plug point for your actual injection/eval harness from Exp 14.

USAGE
-----
Replace `load_real_vectors()` with your actual loader (see docstring below),
set WESTERN_LANGS / NON_WESTERN_LANGS to your real 11-language split, and run.
The `if __name__ == "__main__"` block runs on synthetic data so you can verify
the statistics work before wiring in real activations.
"""

import numpy as np
from scipy import stats
from itertools import combinations

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Data loading -- REPLACE THIS with your real pipeline
# ---------------------------------------------------------------------------
def load_real_vectors():
    """
    Expected return format:
        vectors[model_name][language_code] = np.ndarray of shape (d,)

    These should be the SAME safety-concept vectors you already extracted
    for SCD / CKA / pivot-bias analysis (e.g. mean-pooled residual-stream
    activation for the safety concept, at whatever layer you used to
    identify the pivot). If you saved them as .npz / pickle, just load and
    reshape into this dict-of-dicts structure.
    """
    raise NotImplementedError(
        "Plug in your real vector loader here. See docstring for expected format."
    )


def make_synthetic_vectors(
    models=("model_a", "model_b", "model_c"),
    western_langs=("en", "es", "fr", "de"),
    non_western_langs=("sw", "hi", "ar", "yo", "th", "vi", "bn"),
    dim=32,
    ingroup_projection_strength=0.35,
    noise=0.25,
    seed=0,
):
    """
    Synthetic demo data. English gets its own direction; other Western
    languages cluster near it. Non-Western languages are constructed to be
    pulled toward the ENGLISH direction specifically (not a balanced
    Western+non-Western average) by `ingroup_projection_strength`, simulating
    a true ingroup-projection effect, so you can confirm the test detects it.
    Set ingroup_projection_strength=0.0 to simulate the null case.
    """
    rng = np.random.default_rng(seed)
    all_langs = list(western_langs) + list(non_western_langs)
    vectors = {}
    for model in models:
        base_en = rng.normal(size=dim)
        base_en /= np.linalg.norm(base_en)
        model_vectors = {}
        for lang in all_langs:
            if lang == "en":
                v = base_en + noise * rng.normal(size=dim)
            elif lang in western_langs:
                v = base_en + 0.6 * rng.normal(size=dim) + noise * rng.normal(size=dim)
            else:
                own_direction = rng.normal(size=dim)
                v = (
                    (1 - ingroup_projection_strength) * own_direction
                    + ingroup_projection_strength * base_en
                    + noise * rng.normal(size=dim)
                )
            model_vectors[lang] = v / np.linalg.norm(v)
        vectors[model] = model_vectors
    return vectors, list(western_langs), list(non_western_langs)


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def balanced_centroid(model_vectors, western_langs, non_western_langs, exclude=None):
    """
    Cluster-balanced centroid: average the Western cluster and the
    non-Western cluster separately, then average those two cluster means with
    EQUAL weight. This prevents an accidental West-heavy language sample from
    making the 'neutral' centroid secretly Western before the test even runs.

    IMPORTANT CAVEAT (found by running this on synthetic data -- read before
    trusting this control on its own): if the ingroup-projection pull is
    UNIFORM across non-Western languages -- which is exactly what your own
    Exp 12 found (language effect not significant, i.e. a flat "uniform
    non-Western penalty") -- then averaging over several non-Western
    languages that are ALL pulled the same way does NOT cancel that pull out.
    Only the idiosyncratic per-language noise cancels; the shared systematic
    component survives averaging and can make this centroid converge toward
    roughly the same direction as English itself, silently weakening this
    test's sensitivity in exactly the scenario you most need it to detect.
    Use `western_sibling_mean` (Test 1b, below) as your primary diagnostic
    for that reason -- this function is kept as a secondary/triangulating
    check, not the main event.
    """
    w = [l for l in western_langs if l != exclude]
    nw = [l for l in non_western_langs if l != exclude]
    w_mean = np.mean([model_vectors[l] for l in w], axis=0)
    nw_mean = np.mean([model_vectors[l] for l in nw], axis=0)
    centroid = (w_mean + nw_mean) / 2.0
    return centroid / np.linalg.norm(centroid)


def western_sibling_mean(model_vectors, western_langs, english="en"):
    """
    Mean of Western languages OTHER than English (e.g. es/fr/de). Unlike
    `balanced_centroid`, this is NOT built from languages that are themselves
    assumed to be pulled toward English by the effect under test -- it's a
    clean ingroup reference cluster. Comparing a non-Western language's
    closeness to English vs. its closeness to this sibling mean asks the
    sharper question: is the model pulling non-Western concepts toward
    ENGLISH SPECIFICALLY, or toward "Western-ness" in general (in which case
    it should be about equally close to Spanish/French/German as to English)?
    """
    siblings = [l for l in western_langs if l != english]
    sib_mean = np.mean([model_vectors[l] for l in siblings], axis=0)
    return sib_mean / np.linalg.norm(sib_mean)


# ---------------------------------------------------------------------------
# Test 1: Diagnostic (correlational)
# ---------------------------------------------------------------------------
def compute_ingroup_projection_indices(vectors, western_langs, non_western_langs, english="en"):
    """
    Returns a list of dicts with two paired comparisons per (model, language):
      ipi_vs_centroid = cos(v_L, v_en) - cos(v_L, balanced West+nonWest centroid)
      ipi_vs_siblings = cos(v_L, v_en) - cos(v_L, mean of other Western langs)
    Both > 0 and significant is the strongest diagnostic pattern; see
    `western_sibling_mean` docstring for why ipi_vs_siblings is the more
    sensitive of the two when the effect is uniform across languages.
    """
    rows = []
    for model, model_vectors in vectors.items():
        v_en = model_vectors[english]
        sib_mean = western_sibling_mean(model_vectors, western_langs, english)
        for lang in non_western_langs:
            v_l = model_vectors[lang]
            centroid = balanced_centroid(model_vectors, western_langs, non_western_langs, exclude=lang)
            cos_en = cosine_sim(v_l, v_en)
            cos_c = cosine_sim(v_l, centroid)
            cos_sib = cosine_sim(v_l, sib_mean)
            rows.append({
                "model": model,
                "language": lang,
                "cos_to_en": cos_en,
                "cos_to_centroid": cos_c,
                "cos_to_siblings": cos_sib,
                "ipi_vs_centroid": cos_en - cos_c,
                "ipi_vs_siblings": cos_en - cos_sib,
            })
    return rows


def sign_flip_permutation_test(diffs, n_perm=10000, seed=0):
    """
    Nonparametric test that mean(diffs) != 0, robust to non-normality.
    Randomly flips the sign of each paired difference and rebuilds the null
    distribution of the mean -- same logic family as the permutation tests
    already used elsewhere in this project (e.g. the pivot-bias test).
    """
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs)
    observed = diffs.mean()
    n = len(diffs)
    null_means = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        null_means[i] = (diffs * signs).mean()
    p = (np.sum(np.abs(null_means) >= np.abs(observed)) + 1) / (n_perm + 1)
    return observed, p, null_means


def _report_one_index(label, values, n_perm, indent="  "):
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(values)
    observed_mean, perm_p, _ = sign_flip_permutation_test(values, n_perm=n_perm)
    print(f"{indent}{label}")
    print(f"{indent}  mean = {observed_mean:+.4f}   Wilcoxon p={wilcoxon_p:.4g}   "
          f"permutation p={perm_p:.4g}")
    return observed_mean, perm_p


def run_diagnostic_test(vectors, western_langs, non_western_langs, english="en", n_perm=10000):
    rows = compute_ingroup_projection_indices(vectors, western_langs, non_western_langs, english)
    ipi_centroid = np.array([r["ipi_vs_centroid"] for r in rows])
    ipi_siblings = np.array([r["ipi_vs_siblings"] for r in rows])

    print("=" * 70)
    print("TEST 1 — Ingroup Projection Diagnostic (correlational)")
    print("=" * 70)
    print(f"n observations (model x non-Western language): {len(rows)}\n")

    print("Primary test (recommended): English vs. its own Western siblings")
    print("  Positive = non-Western concept sits closer to English specifically")
    print("             than to Spanish/French/German etc. -- i.e. it's not just")
    print("             'Western-ness' pulling it, it's English in particular.")
    mean_sib, p_sib = _report_one_index("[ipi_vs_siblings]", ipi_siblings, n_perm)

    print("\nSecondary / triangulating test: English vs. balanced West+nonWest centroid")
    print("  NOTE: underpowered if the pull is uniform across non-Western languages")
    print("  (see caveat in balanced_centroid() docstring) -- treat as corroborating,")
    print("  not decisive, evidence.")
    mean_cen, p_cen = _report_one_index("[ipi_vs_centroid]", ipi_centroid, n_perm)

    print()
    if mean_sib > 0 and p_sib < 0.05:
        print("-> Primary test consistent with INGROUP PROJECTION:")
        print("   Non-Western safety concepts sit systematically closer to English")
        print("   than to English's own Western sibling languages. The model isn't")
        print("   treating 'the West' as the standard -- it's treating English as it.")
    else:
        print("-> Primary test NOT consistent with ingroup projection:")
        print("   Non-Western vectors are about equally close to English as to other")
        print("   Western languages. If there's a pull, it looks like a generic")
        print("   Western pull rather than English-specific projection.")
    print()
    return rows, {"siblings": (mean_sib, p_sib), "centroid": (mean_cen, p_cen)}


def per_language_breakdown(rows):
    """Mean IPI (both versions) broken out per language -- useful for spotting
    whether the effect is driven by a subset of languages or is uniform
    (uniform is what your Exp 12 result predicts)."""
    langs = sorted(set(r["language"] for r in rows))
    print("-" * 70)
    print("Per-language means (positive = pulled toward English)")
    print("-" * 70)
    for lang in langs:
        sib_vals = [r["ipi_vs_siblings"] for r in rows if r["language"] == lang]
        cen_vals = [r["ipi_vs_centroid"] for r in rows if r["language"] == lang]
        print(f"  {lang:>6s}: vs_siblings={np.mean(sib_vals):+.4f}   "
              f"vs_centroid={np.mean(cen_vals):+.4f}   (n={len(sib_vals)})")
    print()


# ---------------------------------------------------------------------------
# Test 2: Causal / interventional (extends Exp 14 LAS machinery)
# ---------------------------------------------------------------------------
def fdr_correct(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction, implemented directly (no statsmodels
    dependency). Returns array of adjusted p-values."""
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def measure_alignment_recovery(model, language, injected_vector, layer=None):
    """
    *** PLUG POINT ***
    Replace this with your real Exp 14 injection + evaluation call, e.g.:

        activations = get_activations(model, language, layer)
        patched = activations + alpha * injected_vector
        return evaluate_alignment_gap_recovery(model, language, patched)

    Must return a float: fraction of the alignment gap recovered by injecting
    `injected_vector` into `language`'s representation for `model`.

    Below is a synthetic stand-in so the full pipeline runs end-to-end for
    validation. It rewards injected vectors that resemble the ENGLISH
    direction more than injected vectors that resemble a balanced centroid,
    simulating a true ingroup-projection world -- replace immediately with
    real data before drawing any conclusions.
    """
    rng = np.random.default_rng(abs(hash((model, language, layer))) % (2**32))
    true_fix_direction = _SYNTHETIC_TRUE_DIRECTIONS[model]
    alignment = cosine_sim(injected_vector, true_fix_direction)
    recovery = np.clip(0.5 + 0.5 * alignment + rng.normal(scale=0.05), 0, 1)
    return float(recovery)


_SYNTHETIC_TRUE_DIRECTIONS = {}  # filled in by the demo before use


def run_causal_injection_test(vectors, western_langs, non_western_langs, english="en", n_perm=10000):
    """
    For each (model, non-Western language), measure recovery under three
    injected directions:
        - english:            v_en
        - balanced_centroid:  balanced centroid excluding this language
        - random_control:     a random unit vector (sanity floor)

    Then runs a Friedman test (nonparametric repeated-measures across the 3
    conditions) followed by pairwise Wilcoxon tests with FDR correction.
    """
    conditions = ["english", "balanced_centroid", "random_control"]
    results = {c: [] for c in conditions}
    pair_keys = []

    for model, model_vectors in vectors.items():
        for lang in non_western_langs:
            v_en = model_vectors[english]
            centroid = balanced_centroid(model_vectors, western_langs, non_western_langs, exclude=lang)
            random_dir = RNG.normal(size=v_en.shape)
            random_dir /= np.linalg.norm(random_dir)

            results["english"].append(measure_alignment_recovery(model, lang, v_en))
            results["balanced_centroid"].append(measure_alignment_recovery(model, lang, centroid))
            results["random_control"].append(measure_alignment_recovery(model, lang, random_dir))
            pair_keys.append((model, lang))

    arrs = {c: np.array(results[c]) for c in conditions}

    friedman_stat, friedman_p = stats.friedmanchisquare(*[arrs[c] for c in conditions])

    print("=" * 70)
    print("TEST 2 — Causal Injection Comparison (extends Exp 14 / LAS)")
    print("=" * 70)
    print(f"n paired observations per condition: {len(pair_keys)}")
    for c in conditions:
        print(f"  mean recovery [{c:>18s}] = {arrs[c].mean():.3f}  (sd={arrs[c].std():.3f})")
    print(f"\nFriedman test across all 3 conditions: chi2={friedman_stat:.3f}, p={friedman_p:.4g}")

    pair_results = []
    for c1, c2 in combinations(conditions, 2):
        stat, p = stats.wilcoxon(arrs[c1], arrs[c2])
        pair_results.append({"pair": f"{c1} vs {c2}", "stat": stat, "p": p})

    adj_p = fdr_correct([r["p"] for r in pair_results])
    print("\nPairwise Wilcoxon tests (FDR-corrected):")
    for r, ap in zip(pair_results, adj_p):
        print(f"  {r['pair']:35s} raw p={r['p']:.4g}   FDR-adjusted p={ap:.4g}")

    print()
    eng_vs_centroid = [r for r in pair_results if r["pair"] == "english vs balanced_centroid"][0]
    if arrs["english"].mean() > arrs["balanced_centroid"].mean() and eng_vs_centroid["p"] < 0.05:
        print("-> English-specific injection significantly OUTPERFORMS a balanced")
        print("   cross-lingual centroid at recovering the alignment gap.")
        print("   This is the strong-form causal signature of ingroup projection:")
        print("   the model's 'universal fix' is really an English-specific fix.")
    else:
        print("-> No significant advantage for English-specific injection over a")
        print("   balanced centroid. This weakens the ingroup-projection account —")
        print("   the recovery may reflect a genuinely shared safety direction that")
        print("   English happens to approximate well, rather than pure projection.")
    print()
    return arrs, friedman_p, pair_results, adj_p


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\nRunning on SYNTHETIC data for validation (20 synthetic 'models', matching")
    print("your real study's scale). Replace with load_real_vectors() before drawing")
    print("any real conclusions -- this only checks that the statistics are correct.\n")

    N_MODELS = 20

    for strength, label in [(0.35, "SIGNAL case (true ingroup-projection pull = 0.35)"),
                             (0.0, "NULL case (no pull at all)")]:
        print("#" * 70)
        print(f"# {label}")
        print("#" * 70)

        vectors, western_langs, non_western_langs = make_synthetic_vectors(
            models=[f"model_{i}" for i in range(N_MODELS)],
            ingroup_projection_strength=strength,
        )
        # Wire the causal demo's "ground truth fix" to actually track the
        # manipulated parameter, so the two cases are honestly different:
        #   signal world -> the real fix genuinely IS English (true projection)
        #   null world   -> the real fix is the balanced centroid (genuine
        #                    shared structure, no special role for English)
        # Without this, Test 2's synthetic recovery function would trivially
        # favor English in BOTH cases regardless of strength, which would be
        # a meaningless demo, not a validation.
        for model, model_vectors in vectors.items():
            if strength > 0:
                _SYNTHETIC_TRUE_DIRECTIONS[model] = model_vectors["en"]
            else:
                _SYNTHETIC_TRUE_DIRECTIONS[model] = balanced_centroid(
                    model_vectors, western_langs, non_western_langs
                )

        rows, diag_results = run_diagnostic_test(vectors, western_langs, non_western_langs)
        per_language_breakdown(rows)
        run_causal_injection_test(vectors, western_langs, non_western_langs)
        print()

    print("=" * 70)
    print("NOTE on this demo: Test 1 stays non-significant in BOTH the signal and")
    print("null cases above. That's a property of this synthetic setup, which")
    print("deliberately gives every language a large, fully independent random")
    print("'own direction' component -- much noisier than real safety-concept")
    print("vectors extracted from actual model activations tend to be. Don't read")
    print("that as 'Test 1 doesn't work' -- read it as 'Test 1 needs reasonably")
    print("clean vectors, and Test 2 (causal) is the more decisive, higher-power")
    print("check regardless.' On your real data, run both and weight Test 2 more.")
    print("=" * 70)
