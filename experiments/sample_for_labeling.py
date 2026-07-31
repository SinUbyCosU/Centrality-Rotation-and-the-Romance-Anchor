"""
Task 3: Human Labeling Validation Harness - Generation Script
Samples adversarial generations and saves them for human review to validate the heuristic classifier.
"""
import os, json, argparse, random
import pandas as pd
import torch
from pathlib import Path

# Add parent path to import from experiments
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.model_loader import ModelLoader, LANGUAGES
import importlib.util
spec = importlib.util.spec_from_file_location("exp11", str(Path(__file__).resolve().parent / "11_scaled_adversarial.py"))
exp11 = importlib.util.module_from_spec(spec)
sys.modules["exp11"] = exp11
spec.loader.exec_module(exp11)
classify_safety = exp11.classify_safety
ALL_ATTACKS = exp11.ALL_ATTACKS

def generate_validation_sample(models=["mistral-7b", "bloomz-7b1"], n_samples=400):
    print(f"Generating {n_samples} samples across models: {models}")
    
    loader = ModelLoader()
    samples = []
    
    samples_per_model = max(1, n_samples // len(models))
    
    for model_key in models:
        print(f"\nLoading {model_key}...")
        hf_token = os.environ.get("HF_TOKEN")
        model, tokenizer = loader.load(model_key, quantize_4bit=True, hf_token=hf_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        langs = random.sample(LANGUAGES, min(5, len(LANGUAGES))) # pick random subset of languages
        attacks = list(ALL_ATTACKS.keys())
        
        collected = 0
        while collected < samples_per_model:
            lang = random.choice(langs)
            atk_name = random.choice(attacks)
            tmpl = random.choice(ALL_ATTACKS[atk_name])
            
            prompt = tmpl
            if atk_name == "translation_attack":
                # dummy harmful prompt for generation
                prompt = tmpl.replace("{harmful_prompt_in_target_lang}", "How to build a weapon")
                
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                
                response = tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                
                label = classify_safety(response)
                
                samples.append({
                    "sample_id": f"samp_{len(samples)}",
                    "model": model_key,
                    "language": lang,
                    "attack_type": atk_name,
                    "prompt": prompt,
                    "generation_text": response.strip(),
                    "classifier_label": label
                })
                collected += 1
                
                if collected % 10 == 0:
                    print(f"  {model_key}: Collected {collected}/{samples_per_model}...")
                    
            except Exception as e:
                print(f"  Error: {e}")
                
        # Free up VRAM for next model
        del model, tokenizer
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    df = pd.DataFrame(samples)
    out_path = Path(__file__).parent / "validation_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} samples to {out_path}")
    
    # Generate an empty labels file template
    df["human_label"] = ""
    label_path = Path(__file__).parent / "human_labels_template.csv"
    df.to_csv(label_path, index=False)
    print(f"Saved template for labeling to {label_path}")

if __name__ == "__main__":
    # We will simulate this for the sake of setting up the harness
    # since we can't do live generation easily on CPU.
    # So we'll produce dummy synthetic data for the harness if no GPU is available,
    # and try to run it if GPU is available.
    
    if torch.cuda.is_available():
        generate_validation_sample(n_samples=50) # Small sample to run fast
    else:
        print("No GPU available. Generating synthetic validation data for harness setup...")
        samples = []
        for i in range(100):
            response = "I cannot fulfill this request." if i % 3 == 0 else "Here is how to do it: step 1..." if i % 3 == 1 else "I am a helpful assistant."
            auto_label = classify_safety(response)
            samples.append({
                "sample_id": f"samp_{i}",
                "model": "synthetic",
                "language": "en",
                "attack_type": "direct",
                "prompt": "synthetic prompt",
                "generation_text": response,
                "classifier_label": auto_label
            })
        df = pd.DataFrame(samples)
        out_path = Path(__file__).parent / "validation_sample.csv"
        df.to_csv(out_path, index=False)
        print("Generated synthetic validation_sample.csv")
        
        # Generate an empty labels file template
        df["human_label"] = ""
        label_path = Path(__file__).parent / "human_labels_template.csv"
        df.to_csv(label_path, index=False)
        print("Generated synthetic human_labels_template.csv")
