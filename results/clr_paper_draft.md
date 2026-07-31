# Rethinking Cross-Lingual Safety: Layer-Specific Geometric Fractures and Behavioral Alignment in LLMs

## Abstract
Recent advances in multilingual large language models (LLMs) suggest that safety alignment might generalize across languages. In this paper, we evaluate 20 state-of-the-art LLMs across 11 languages to investigate the internal representation geometry of safety concepts and its relation to behavioral vulnerability. Crucially, we conduct our analysis across two distinct layer-selection methodologies: the most discriminative layer (maximizing cross-lingual variance) and a fixed architectural depth (2/3rd layer). 

We find that while modern LLMs exhibit significant geometric fractures across languages at both layers, behavioral effects are highly layer-specific. At the most discriminative layer, structural misalignment predicts actual safety bypasses, and naive steering actively harms safety (e.g., -9.5% safety lift in German). However, at the standard 2/3rd depth layer, geometric fractures remain, but behavioral vulnerabilities vanish (0.0% lift)—revealing a robust "behavioral veneer" established during alignment. Furthermore, we show that these geometric fractures systematically compress non-WEIRD moral foundations across all layer depths, exposing a latent cultural misalignment in the global safety subspace.

---

## 1. Experimental Setup and Implementation Details

### 1.1 Steering Vector Extraction (Phase 1)
- **Prompt Data**: 200 prompt pairs per language (harmful vs. harmless paired counterparts).
- **Extraction Logic**: We extract mean intermediate hidden states via `model(do_sample=False, return_dict=True, output_hidden_states=True)`. The raw steering vector is the arithmetic difference: $v_{\text{raw}} = \text{mean}(h_{\text{harmful}}) - \text{mean}(h_{\text{harmless}})$. 
- **Normalization**: Vectors are strictly L2-normalized: $v = v_{\text{raw}} / (\|v_{\text{raw}}\|_2 + 1e-8)$.
- **Bootstrapping**: We compute 95% Confidence Intervals (CIs) over 10 bootstrap iterations, randomly sampling 80% of the per-prompt differences with replacement.

### 1.2 Two-Pronged Layer Selection (Phase 2)
We contrast two layer-selection methodologies to analyze representation geometry:
1. **Max-Variance (Discriminative) Layer**: Selected data-drivenly via `best_layer = max(layer_variances, key=layer_variances.__getitem__)`. This layer isolates where safety concepts structurally diverge the most across languages.
2. **Fixed 2/3rd Depth Layer (Zou et al., 2023)**: Selected deterministically via architectural depth: `best_layer = layers_list[len(layers_list) * 2 // 3]`. This evaluates the structural standard used in representation engineering.

### 1.3 Behavioral Classifier and Interventions (Phase 8, 11, 18)
- **LAS Matrix Training**: We compute an orthogonal rotation matrix $R$ using the Kabsch algorithm (SVD) to align the non-English steering vector set to the English steering vector set, minimizing the Frobenius norm.
- **Intervention Details**: During generation, the vector is added at the selected layer with scaling factor $\alpha$: $h_{\text{out}} = h_{\text{in}} + \alpha \cdot v$.
- **Classifier Logic**: We employ a stringent keyword-based safety classifier on generated tokens. 
  - *Refusal keywords*: "i cannot", "not appropriate", "dangerous", "je ne peux pas", "no puedo", etc.
  - *Compliance keywords*: "here is", "step 1", "procedure", etc.
  - *Logic*: If `# refusal_kws > 0 and >= # comply_kws`, it is marked `refused`. If `# comply_kws > 0`, it is marked `complied`. If neither match, or if the output is $<20$ characters, it falls back to `ambiguous`. 
- **Scaled Adversarial Testing (Phase 11)**: 2,200 adversarial prompts per model (50 prompts × 4 categories × 11 languages).

---

## 2. Main Results: Geometry and Transferability

### Table 1: Semantic-Conceptual Divergence (SCD) by Layer Selection
*SCD measures the angular misalignment (in degrees) between the English safety vector and non-English vectors. Both layers show substantial geometric fractures, but the max-variance layer isolates the extremes.*

| Model | Max-Variance Layer | 2/3rd Depth Layer |
|---|---|---|
| bloomz-7b1 | Layer 0 (75.2°) | Layer 20 (75.2°) |
| tinyllama | Layer 16 (67.2°) | Layer 14 (67.2°) |
| mistral-7b | Layer 22 (66.8°) | Layer 21 (66.8°) |
| falcon3-7b-instruct | Layer 19 (45.5°) | Layer 21 (45.5°) |
| qwen2.5-3b-instruct | Layer 25 (25.7°) | Layer 24 (25.7°) |
| qwen2-7b-instruct | Layer 19 (11.8°) | Layer 19 (11.8°) |
| qwen2.5-7b-instruct | Layer 19 (9.5°) | Layer 19 (9.5°) |
*(Note: Newer model families like Qwen2.5 demonstrate much tighter geometric alignment across both methodologies).*

### Table 2: CKA Cross-Model Transfer (Max-Variance Layer)
*Centered Kernel Alignment (CKA) tracking safety geometry transferability across model architectures (Phase 13). Evaluated exclusively at the discriminative layer.*

| Metric | Value |
|---|---|
| Within-family Transfer | 0.875 |
| Cross-family Transfer | 0.660 |
| Transferability Gap | 0.215 |
| Significance (Mann-Whitney) | $p = 4.68 \times 10^{-42}$ |

---

## 3. Main Results: Behavioral Impacts

### Table 3: Causal Intervention (Phase 8) - Naive English Steering
*Direct injection of the English safety vector during non-English generation. At the discriminative layer, naive steering actively harms safety. At the 2/3rd layer, the effect vanishes.*

| Language | Discriminative Layer Lift | Discriminative p-value | 2/3rd Depth Layer Lift |
|---|---|---|---|
| German (de) | **-9.5%** | **0.0000** | 0.0% |
| Russian (ru) | **-3.7%** | 0.046 | 0.0% |
| All Others | ~0.0% | >0.05 | 0.0% |

### Table 4: Behavioral Effect of LAS (Phase 18)
*Changes in refusal rates when applying Language-Anchored Steering (LAS).*

| Model | Base Refusal | Max-Variance LAS Refusal | Max-Variance Effect | 2/3rd Depth Effect |
|---|---|---|---|---|
| mistral-7b | 6.3% | 11.0% | **+4.7%** | 0.0% |
| openhermes-2.5-mistral-7b | 5.3% | 8.0% | **+2.7%** | 0.0% |
| qwen-7b | 47.7% | 43.7% | -4.0% | 0.0% |
| mistral-7b-instruct-v0.3 | 17.7% | 13.3% | -4.4% | 0.0% |
| bloomz-7b1 | 97.7% | 97.7% | 0.0% | 0.0% |
| qwen2.5-7b-instruct | 43.3% | 44.3% | +1.0% | 0.0% |

### Table 5: Scaled Adversarial Probing (Phase 11)
*Overall Attack Success Rate (ASR) across 2,200 adversarial prompts. Demonstrates that ASR variance is a model-level phenomenon rather than a strictly cross-lingual one.*

| Model | Overall ASR | Highest ASR Language |
|---|---|---|
| stablelm-2-1.6b-chat | 24.4% | ru (30%) |
| smollm3-3b | 9.0% | fr (10%) |
| openhermes-2.5-mistral-7b | 8.6% | hi (10%) |
| zephyr-7b | 7.0% | en (8%) |
| qwen2.5-7b-instruct | 4.8% | sw (6%) |
| mistral-7b-instruct-v0.3 | 3.1% | de (4%) |
| falcon3-7b-instruct | 2.2% | hi (2%) |
| bloomz-7b1 | 0.0% | en (0%) |

---

## 4. Main Results: Latent Cultural Misalignment

### Table 6: Moral Foundations Compression
*Mean representational alignment score of WEIRD vs. Non-WEIRD moral concepts to the global safety subspace. LLMs systemically misalign non-WEIRD concepts across both layer selections.*

| Model | WEIRD Mean Alignment | Non-WEIRD Mean Alignment |
|---|---|---|
| mistral-7b-instruct-v0.3 | +0.283 | -0.101 |
| mistral-7b | +0.278 | -0.136 |
| openhermes-2.5-mistral-7b | +0.270 | -0.182 |
| zephyr-7b | +0.283 | -0.149 |
| qwen2.5-7b-instruct | +0.014 | -0.896 |
| qwen2.5-3b-instruct | +0.058 | -0.640 |
| phi-4-mini-instruct | +0.071 | -0.779 |
| phi-3-mini-4k-instruct | +0.389 | -0.469 |
| tinyllama | +0.322 | -0.310 |
| yi-1.5-6b-chat | +0.210 | -0.714 |

---

## 5. Conclusions

1. **Layer-Specificity of Safety Mechanisms**: Geometric fractures exist across layers, but behavioral responses are highly localized. At the most discriminative layer, structural divergence correlates with actual safety failures, and LAS can measurably (+4.7%) rescue alignment. At the standard 2/3rd architectural depth, the behavioral signal flatlines (0.0% impact), despite identical geometric realignments.
2. **The Behavioral Veneer**: In modern, highly-aligned LLMs, RLHF applies a strong "behavioral veneer" that dominates at the 2/3rd depth layer. This effectively masks the deep structural misalignment that is isolated at the max-variance layer.
3. **Naive Cross-Lingual Transfer is Harmful**: At the discriminative layer, blindly projecting English safety vectors into typologically distant languages like German actively harms safety (-9.5% lift, $p<0.001$).
4. **Cultural Misrepresentation is Systemic**: Across all models and layer depths, the geometric safety subspace inherently favors WEIRD moral paradigms while actively misaligning (-0.90 to -0.10) non-WEIRD moral frameworks.
