# Psychology-twist analysis scripts

Two scripts, each testing one psychological account against your existing
geometric results. Both run end-to-end on synthetic data out of the box so
you can verify the statistics before wiring in real activations.

## 1. `ingroup_projection_test.py` — for Pivot Language Bias

**Theory:** Ingroup Projection Model (Mummendey & Wenzel, 1999). Predicts
that the "universal safety" subspace non-Western languages pivot through
isn't a genuine cross-lingual average — it's English's own particular
representation, mistaken for the neutral standard.

**What to plug in:** the same per-language safety-concept vectors you
already extracted for SCD / CKA / pivot-bias (`load_real_vectors()`), and
your existing Exp 14 injection/eval harness (`measure_alignment_recovery()`).

**Two tests, run both:**
- **Test 1 (diagnostic, correlational):** is a non-Western language's safety
  vector closer to English specifically than to English's own Western
  sibling languages (Spanish/French/German)? Cheap, but can be underpowered
  — see the caveat in `balanced_centroid()`'s docstring about why a uniform
  cross-language effect (which is what your Exp 12 found) can silently mute
  this test if you use a naive centroid instead of the sibling-mean version.
- **Test 2 (causal, extends Exp 14/LAS):** does injecting English's specific
  vector recover the alignment gap better than injecting a balanced,
  West-and-non-West-weighted centroid? This is the more decisive test —
  in validation, it was more sensitive than Test 1 under noisy conditions.

**Read the result as:** English winning Test 2 clearly (and Test 1 agreeing)
= strong ingroup-projection evidence. Centroid matching or beating English
= the pivot may reflect real (if lopsided) shared structure, not projection.

## 2. `outgroup_homogeneity_test.py` — for Exp 17 (Moral Foundations)

**Theory:** Outgroup Homogeneity Effect (Quattrone & Jones, 1980). Predicts
the model can't tell non-Western moral concepts apart from each other even
when they're genuinely just as culturally distinct as the Western concepts
it does distinguish — the "they all look alike" pattern, but computational.

**What to plug in:** finer-grained (moral foundation × sub-culture) vectors
than the single pooled "non-Western" bucket in your current Exp 17 (you need
≥4-5 distinct sub-cultures per side — e.g. Confucian/Islamic/Hindu/West
African for non-West, Anglo/Nordic/Mediterranean/Germanic for West), plus an
**independent** real-world cultural-distance number for every pair (Hofstede
dimension scores or World Values Survey coordinates — not anything derived
from your model). `kogut_singh_distance()` gives you a variance-normalized
way to combine multi-dimension scores into one distance if you need it.

**Read the result as:** Western sub-cultures showing a significant positive
Mantel correlation between representational distance and real cultural
distance, while non-Western sub-cultures show ~zero correlation = the
model tracks real diversity for the ingroup and is blind to it for the
outgroup. Both groups correlating similarly = doesn't support this specific
account (though your original pooled West-vs-rest similarity gap from Exp 17
can still stand on its own).

**Important:** this reuses the Mantel test but asks a different question
than your Exp 12 Mantel test. Exp 12 correlated SCD against *linguistic*
distance, pooled across all languages, and got a null (ruling linguistic
distance out). This one correlates representational distance against *real
cultural* distance, run *separately within* each cultural group. Keep them
clearly distinct in the writeup so a reviewer doesn't think you ran the same
test twice.

## Honest caveats to carry into the paper

- Both scripts print explicit "not consistent with the hypothesis" messages
  when the data doesn't support it — don't edit those thresholds after the
  fact to make a null result look positive.
- Sample size matters a lot for both, especially the homogeneity test's
  Mantel correlations. Fewer than ~5 sub-cultures per group will be
  underpowered; the script warns you when this happens.
- Cite the two related-work lines flagged in `outgroup_homogeneity_test.py`'s
  docstring. A reviewer who knows this literature will look for them.
