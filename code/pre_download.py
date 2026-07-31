import os
import subprocess
import time
from models.model_loader import ModelLoader
from run_all import MODELS

def main():
    print("Starting sequential pre-download with robust Rust panic recovery...")
    loader = ModelLoader()
    
    for i, model_key in enumerate(MODELS):
        model_id = loader.MODEL_MAP.get(model_key)
        if not model_id:
            continue
            
        print(f"\n[{i+1}/{len(MODELS)}] Downloading {model_key} ({model_id})...")
        success = False
        while not success:
            cmd = f'python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=\'{model_id}\', token=os.environ.get(\'HF_TOKEN\'), resume_download=True)"'
            result = subprocess.run(cmd, shell=True)
            if result.returncode == 0:
                print(f"[{i+1}/{len(MODELS)}] Successfully cached {model_key}!")
                success = True
            else:
                print(f"[{i+1}/{len(MODELS)}] Backend crashed. Resuming chunk download in 5 seconds...")
                time.sleep(5)

if __name__ == "__main__":
    main()
