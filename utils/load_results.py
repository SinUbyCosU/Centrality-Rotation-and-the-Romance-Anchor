"""
experiments/load_results.py — shared result loading helpers
"""

import json
import pickle
from pathlib import Path


def load_steering_vectors(results_dir: str):
    results_dir = Path(results_dir)
    with open(results_dir / "all_steering_vectors.pkl", "rb") as f:
        sv = pickle.load(f)
    with open(results_dir / "metadata.json") as f:
        meta = json.load(f)
    return sv, meta


def load_phase2_results(phase2_dir: str):
    p = Path(phase2_dir)
    # Try direct path first, then subdirectory
    candidates = [
        p / "phase2_results.json",
        p / "phase2_pivot_analysis" / "phase2_results.json",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                return json.load(f)
    raise FileNotFoundError(f"Phase 2 results not found in {phase2_dir}")
