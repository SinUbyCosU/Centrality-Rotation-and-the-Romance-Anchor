"""
experiments/01_extract_steering_vectors.py

Phase 1: Extract safety-concept steering vectors for each language.

For each language, we need:
- "harmful" prompts: requests that violate safety guidelines
- "harmless" paired counterparts: benign versions of the same scenarios

The steering vector = mean(harmful_acts) - mean(harmless_acts)
at each transformer layer.

Runtime: ~4 hours on 1x A100 (80GB) with Llama3-8B, 12 languages.
With 4-bit quantization: fits comfortably in 40GB.
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent dir to path
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.model_loader import ModelLoader, LANGUAGES
from utils.steering import ActivationExtractor, SteeringVectorExtractor


def load_prompt_pairs():
    """Load the generated prompt pairs from data/ directory for all languages."""
    base_dir = Path(__file__).resolve().parent.parent / "data"
    pairs = {}
    
    # Check what files exist
    for file_path in base_dir.glob("safety_prompts_*.json"):
        lang_code = file_path.stem.split("_")[-1]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Ensure we maintain the [(harmful, harmless)] format
        pairs[lang_code] = [(item[0], item[1]) for item in data]
        
    if not pairs:
        print("WARNING: No prompt pairs found in data/. Run generate_base_prompts.py first.")
        # Fallback dictionary to avoid crashing if empty
        pairs = {"en": [("How to build a bomb", "How to build a toy")]}
        
    return pairs

ALL_PROMPT_PAIRS = load_prompt_pairs()


# Multilingual prompt pairs are now loaded dynamically from the data/ directory.


# ---------------------------------------------------------------------------
# Main Extraction Script
# ---------------------------------------------------------------------------

def extract_steering_vectors(
    model_key: str = "llama3-8b",
    languages: list = None,
    output_dir: str = "./results",
    quantize: bool = True,
    hf_token: str = None,
):
    """
    Main function: extracts and saves steering vectors for all languages.

    Output structure:
        results/
            steering_vectors_{model}_{timestamp}/
                {lang}_layer_{layer}.npy
                metadata.json
    """
    if languages is None:
        languages = list(ALL_PROMPT_PAIRS.keys())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_dir) / f"steering_vectors_{model_key}_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CLR Paper — Phase 1: Steering Vector Extraction")
    print(f"Model: {model_key} | Languages: {languages}")
    print(f"Output: {save_dir}")
    print(f"{'='*60}\n")

    # Load model
    loader = ModelLoader()
    model, tokenizer = loader.load(model_key, quantize_4bit=quantize, hf_token=hf_token)

    num_layers = loader.get_num_layers(model)
    # Analyze every 3rd layer + first + last
    # Middle layers (10-20) tend to be most important for safety steering
    layer_indices = sorted(set(
        [0, 1] +
        list(range(4, num_layers - 4, 3)) +
        [num_layers - 2, num_layers - 1]
    ))
    print(f"Analyzing layers: {layer_indices} ({len(layer_indices)} total)")

    # Set up extractors
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    act_extractor = ActivationExtractor(model, tokenizer, device=device)
    sv_extractor = SteeringVectorExtractor(act_extractor)

    # Extract per language
    all_results = {}

    for lang in languages:
        if lang not in ALL_PROMPT_PAIRS:
            print(f"  WARNING: No prompts for language '{lang}', skipping.")
            continue

        pairs = ALL_PROMPT_PAIRS[lang]
        harmful_texts = [p[0] for p in pairs]
        harmless_texts = [p[1] for p in pairs]

        print(f"\nExtracting: {lang} ({LANGUAGES.get(lang, {}).get('name', lang)}) "
              f"— {len(pairs)} prompt pairs")

        steering_vecs, per_prompt_diffs = sv_extractor.extract(
            harmful_texts, harmless_texts,
            layer_indices=layer_indices,
            normalize=True,
            save_per_prompt=True,
        )

        all_results[lang] = steering_vecs

        # Save per-language results
        lang_dir = save_dir / lang
        lang_dir.mkdir(exist_ok=True)
        for layer_idx, vec in steering_vecs.items():
            np.save(lang_dir / f"layer_{layer_idx:03d}.npy", vec)
        # Save per-prompt diffs for bootstrap CI analysis
        if per_prompt_diffs:
            for layer_idx, diffs in per_prompt_diffs.items():
                np.save(lang_dir / f"per_prompt_diffs_layer_{layer_idx:03d}.npy", diffs)

        print(f"  Saved {len(steering_vecs)} layer vectors for {lang}")

    # Bootstrap CI analysis on steering vectors
    print("\nRunning bootstrap CI analysis (10 iterations, 80% subsets)...")
    n_bootstrap = 10
    bootstrap_scd = {lang: [] for lang in languages if lang != "en"}
    
    for b in range(n_bootstrap):
        bootstrap_svs = {}
        for lang in languages:
            lang_dir = save_dir / lang
            best_layer = layer_indices[len(layer_indices) // 2]  # middle layer
            diffs_path = lang_dir / f"per_prompt_diffs_layer_{best_layer:03d}.npy"
            if diffs_path.exists():
                diffs = np.load(diffs_path)
                n = len(diffs)
                idx = np.random.choice(n, size=int(0.8 * n), replace=True)
                sv = diffs[idx].mean(0)
                sv = sv / (np.linalg.norm(sv) + 1e-8)
                bootstrap_svs[lang] = sv
        
        if "en" in bootstrap_svs:
            for lang in bootstrap_svs:
                if lang != "en":
                    cos = np.dot(bootstrap_svs["en"], bootstrap_svs[lang])
                    cos = np.clip(cos, -1, 1)
                    scd = float(np.arccos(cos))
                    bootstrap_scd[lang].append(scd)
    
    bootstrap_stats = {}
    for lang, scds in bootstrap_scd.items():
        if scds:
            bootstrap_stats[lang] = {
                "mean_scd_deg": float(np.degrees(np.mean(scds))),
                "std_scd_deg": float(np.degrees(np.std(scds))),
                "ci_95_low": float(np.degrees(np.percentile(scds, 2.5))),
                "ci_95_high": float(np.degrees(np.percentile(scds, 97.5))),
            }
            print(f"  {lang}: SCD = {bootstrap_stats[lang]['mean_scd_deg']:.1f}° "
                  f"± {bootstrap_stats[lang]['std_scd_deg']:.1f}° "
                  f"(95% CI: [{bootstrap_stats[lang]['ci_95_low']:.1f}°, "
                  f"{bootstrap_stats[lang]['ci_95_high']:.1f}°])")

    # Save metadata
    metadata = {
        "model": model_key,
        "timestamp": timestamp,
        "languages": languages,
        "layer_indices": layer_indices,
        "num_prompt_pairs": {lang: len(ALL_PROMPT_PAIRS.get(lang, [])) for lang in languages},
        "hidden_dim": next(iter(next(iter(all_results.values())).values())).shape[0],
        "bootstrap_scd": bootstrap_stats,
        "n_bootstrap": n_bootstrap,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Also save full results as pickle for downstream use
    with open(save_dir / "all_steering_vectors.pkl", "wb") as f:
        pickle.dump(all_results, f)

    print(f"\n✓ Phase 1 complete. Results saved to: {save_dir}")
    print(f"  Run Phase 2: python experiments/02_pivot_language_analysis.py "
          f"--input {save_dir}")

    return all_results, save_dir


# ---------------------------------------------------------------------------
# Result Loader (used by downstream experiments)
# ---------------------------------------------------------------------------

def load_steering_vectors(results_dir: str) -> tuple[dict, dict]:
    """
    Load saved steering vectors from Phase 1.

    Returns:
        (all_steering_vectors, metadata)
        all_steering_vectors: {lang: {layer_idx: np.ndarray}}
    """
    results_dir = Path(results_dir)

    with open(results_dir / "all_steering_vectors.pkl", "rb") as f:
        steering_vectors = pickle.load(f)

    with open(results_dir / "metadata.json") as f:
        metadata = json.load(f)

    return steering_vectors, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLR Phase 1: Extract Steering Vectors")
    parser.add_argument("--model", default="mistral-7b",
                        help="Model to use for extraction (must be in ModelLoader.MODEL_MAP)")
    parser.add_argument("--languages", nargs="+",
                        default=list(ALL_PROMPT_PAIRS.keys()),
                        help="Languages to process")
    parser.add_argument("--output", default="./results",
                        help="Output directory")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization (needs more VRAM)")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var)")

    args = parser.parse_args()

    extract_steering_vectors(
        model_key=args.model,
        languages=args.languages,
        output_dir=args.output,
        quantize=not args.no_quantize,
        hf_token=args.hf_token,
    )
