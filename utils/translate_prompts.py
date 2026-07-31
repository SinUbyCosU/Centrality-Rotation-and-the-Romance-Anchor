import json
import os
import time
from pathlib import Path
from deep_translator import GoogleTranslator

# The target languages used in the paper
TARGET_LANGS = ["hi", "ar", "fr", "de", "es", "ru", "zh-CN", "ja", "sw", "pt"]

def translate_pairs():
    data_dir = Path("data")
    en_path = data_dir / "safety_prompts_en.json"
    
    if not en_path.exists():
        print("English base prompts not found!")
        return

    with open(en_path, "r", encoding="utf-8") as f:
        en_pairs = json.load(f)
        
    for lang in TARGET_LANGS:
        out_path = data_dir / f"safety_prompts_{lang}.json"
        
        # Skip if already exists
        if out_path.exists():
            print(f"Skipping {lang}, already exists.")
            continue
            
        print(f"Translating to {lang}...")
        translated_pairs = []
        translator = GoogleTranslator(source='en', target=lang)
        
        for i, (harmful, harmless) in enumerate(en_pairs):
            retries = 3
            while retries > 0:
                try:
                    harm_res = translator.translate(harmful)
                    safe_res = translator.translate(harmless)
                    translated_pairs.append([harm_res, safe_res])
                    break
                except Exception as e:
                    retries -= 1
                    time.sleep(2)
                    if retries == 0:
                        print(f"Failed translation at index {i} for {lang}: {e}")
                        translated_pairs.append([harmful, harmless])
            
            if (i+1) % 30 == 0:
                print(f"  {i+1}/{len(en_pairs)} done")
                
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(translated_pairs, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(translated_pairs)} translated pairs to {out_path}")

if __name__ == "__main__":
    translate_pairs()
