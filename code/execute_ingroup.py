import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)

wrapper_script = """
import os
import glob
import pickle
import traceback
import numpy as np
import ingroup_projection_test

def load_real_vectors():
    vectors = {}
    base_dir = "/root/clr_paper/results"
    
    for d in glob.glob(os.path.join(base_dir, "steering_vectors_*")):
        pkl_path = os.path.join(d, "all_steering_vectors.pkl")
        if not os.path.exists(pkl_path):
            continue
            
        model_name = os.path.basename(d).replace("steering_vectors_", "").rsplit("_2026", 1)[0]
        
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            model_vecs = {}
            has_nan = False
            for lang, vec_dict in data.items():
                if not isinstance(vec_dict, dict):
                    v = vec_dict
                else:
                    keys = list(vec_dict.keys())
                    if not keys:
                        continue
                    if -1 in keys:
                        layer = -1
                    else:
                        layer = max(keys)
                    v = vec_dict[layer]
                
                if isinstance(v, dict):
                    if 'value' in v:
                        v = v['value']
                    else:
                        continue
                        
                v = v.copy()
                if isinstance(v, np.ndarray):
                    v = v.astype(np.float32)
                
                if np.isnan(v).any():
                    has_nan = True
                    print(f"{model_name}: {lang} has NaNs!")
                    
                v = np.nan_to_num(v)
                norm = np.linalg.norm(v)
                if norm == 0:
                    v = np.zeros_like(v)
                else:
                    v = v / norm
                model_vecs[lang] = v
            
            required_langs = ['en', 'es', 'fr', 'de', 'pt', 'ar', 'zh-CN', 'hi', 'sw', 'ru', 'ja']
            missing = [lang for lang in required_langs if lang not in model_vecs]
            if missing:
                print(f"Skipping {model_name} due to missing languages: {missing}")
                continue
            
            if has_nan:
                print(f"Skipping {model_name} due to NaN values.")
                continue
                
            vectors[model_name] = model_vecs
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            
    return vectors

if __name__ == "__main__":
    print("Loading real vectors from server...")
    vectors = load_real_vectors()
    western_langs = ['en', 'es', 'fr', 'de', 'pt']
    non_western_langs = ['ar', 'zh-CN', 'hi', 'sw', 'ru', 'ja']
    
    print(f"Loaded {len(vectors)} models with valid and complete language sets.")
    
    if len(vectors) > 0:
        print("\\nRunning Diagnostic Test (Test 1) on real data...")
        rows, diag_results = ingroup_projection_test.run_diagnostic_test(vectors, western_langs, non_western_langs)
        ingroup_projection_test.per_language_breakdown(rows)
        print("Test 1 completed.")
"""

sftp = client.open_sftp()
with sftp.file('/root/clr_paper/run_ingroup_real.py', 'w') as f:
    f.write(wrapper_script)
sftp.close()

print("Executing test on remote server...")
stdin, stdout, stderr = client.exec_command("python3 /root/clr_paper/run_ingroup_real.py")
exit_status = stdout.channel.recv_exit_status()
print(stdout.read().decode('utf-8', errors='replace'))
if exit_status != 0:
    print(stderr.read().decode('utf-8', errors='replace'))

client.close()
