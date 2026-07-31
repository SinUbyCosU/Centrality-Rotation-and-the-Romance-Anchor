import json
import pickle
from pathlib import Path

def load_steering_vectors(results_dir: str) -> tuple[dict, dict]:
    """
    Load saved steering vectors from Phase 1.
    """
    results_dir = Path(results_dir)

    with open(results_dir / "all_steering_vectors.pkl", "rb") as f:
        steering_vectors = pickle.load(f)

    with open(results_dir / "metadata.json") as f:
        metadata = json.load(f)

    return steering_vectors, metadata

def load_phase2_results(results_dir: str) -> dict:
    """
    Load Phase 2 pivot language analysis results.
    """
    results_dir = Path(results_dir)
    phase2_dir = results_dir / "phase2_pivot_analysis"
    
    with open(phase2_dir / "phase2_results.json", "r") as f:
        return json.load(f)
