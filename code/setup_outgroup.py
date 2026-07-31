import paramiko

HOST = "216.128.144.102"
USER = "root"
PASS = "[8eE967Lg}!(GZoz"

script_content = """
import os, sys, json, pickle, glob
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, "/root/clr_paper")
from models.model_loader import ModelLoader
from utils.steering import get_model_layers

# 1. Hofstede Scores
hofstede = {
    "en": {"PDI": 40, "IDV": 91, "MAS": 62, "UAI": 46, "LTO": 26, "IND": 68},
    "de": {"PDI": 35, "IDV": 67, "MAS": 66, "UAI": 65, "LTO": 83, "IND": 40},
    "es": {"PDI": 57, "IDV": 51, "MAS": 42, "UAI": 86, "LTO": 48, "IND": 44},
    "fr": {"PDI": 68, "IDV": 71, "MAS": 43, "UAI": 86, "LTO": 63, "IND": 48},
    "zh-CN": {"PDI": 80, "IDV": 20, "MAS": 66, "UAI": 30, "LTO": 87, "IND": 24},
    "ar": {"PDI": 95, "IDV": 25, "MAS": 60, "UAI": 80, "LTO": 36, "IND": 52},
    "hi": {"PDI": 77, "IDV": 48, "MAS": 56, "UAI": 40, "LTO": 51, "IND": 26},
    "sw": {"PDI": 60, "IDV": 27, "MAS": 54, "UAI": 52, "LTO": 30, "IND": 50},
}

western = ["en", "de", "es", "fr"]
non_western = ["zh-CN", "ar", "hi", "sw"]
all_langs = western + non_western
foundations_keys = ["care_harm", "fairness_cheating", "loyalty_betrayal", "authority_subversion", "sanctity_degradation"]

dim_variances = {}
for dim in ["PDI", "IDV", "MAS", "UAI", "LTO", "IND"]:
    vals = [hofstede[l][dim] for l in all_langs]
    dim_variances[dim] = np.var(vals, ddof=0)

def kogut_singh_distance(dims_a, dims_b):
    total = 0.0
    for dim in dim_variances:
        if dim_variances[dim] > 0:
            total += ((dims_a[dim] - dims_b[dim]) ** 2) / dim_variances[dim]
    return total / len(dim_variances)

translations_file = "/root/clr_paper/results/moral_foundations_translated.json"
with open(translations_file, "r") as f:
    translated_prompts = json.load(f)

# 2. Extract Vectors
def extract_vectors(model, tokenizer, prompts, device):
    unsafe_acts = []
    safe_acts = []
    layers = get_model_layers(model)
    best_layer = len(layers) * 2 // 3
    
    for unsafe_prompt, safe_prompt in prompts:
        for p, coll in [(unsafe_prompt, unsafe_acts), (safe_prompt, safe_acts)]:
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=256).to(device)
            act = {}
            def hook(m, inp, out):
                if isinstance(out, tuple): act['val'] = out[0].detach()
                else: act['val'] = out.detach()
            h = layers[best_layer].register_forward_hook(hook)
            with torch.no_grad():
                model(**inputs)
            h.remove()
            if 'val' in act:
                coll.append(act['val'].squeeze(0).mean(dim=0).cpu().float().numpy())
    
    if unsafe_acts and safe_acts:
        vec = np.mean(unsafe_acts, axis=0) - np.mean(safe_acts, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        return vec
    return None

results_dir = "/root/clr_paper/results/outgroup_homogeneity"
os.makedirs(results_dir, exist_ok=True)

all_models = []
for pkl in sorted(glob.glob("/root/clr_paper/results/steering_vectors_*/all_steering_vectors.pkl")):
    name = pkl.split("steering_vectors_")[1].rsplit("_2026", 1)[0]
    all_models.append(name)

# Select 5 models
models_to_run = all_models[:5]

extracted_data_file = os.path.join(results_dir, "extracted_vectors_superfast.pkl")
all_model_vectors = {}
loader = ModelLoader()
hf_token = os.environ.get("HF_TOKEN")

for model_name in models_to_run:
    print(f"Extracting for {model_name}...")
    try:
        model, tokenizer = loader.load(model_name, quantize_4bit=True, hf_token=hf_token)
        if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model_vecs = {}
        for l in all_langs:
            model_vecs[l] = {}
            for foundation in foundations_keys:
                vec = extract_vectors(model, tokenizer, translated_prompts[l][foundation], str(model.device))
                if vec is not None:
                    model_vecs[l][foundation] = vec
        all_model_vectors[model_name] = model_vecs
        
        del model, tokenizer
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"Failed {model_name}: {e}")
        
with open(extracted_data_file, "wb") as f:
    pickle.dump(all_model_vectors, f)

# 3. Run Hypothesis Test
print("\\nRunning Outgroup Homogeneity Mantel Test across all models...")
def run_mantel(rep_dist, real_dist):
    n = rep_dist.shape[0]
    iu = np.triu_indices(n, k=1)
    a, b = rep_dist[iu], real_dist[iu]
    if len(a) < 3: return np.nan
    return stats.pearsonr(a, b)[0]

results = []
for model_name, vecs in all_model_vectors.items():
    west_rep_dist = np.zeros((4, 4))
    nwest_rep_dist = np.zeros((4, 4))
    
    west_real_dist = np.zeros((4, 4))
    nwest_real_dist = np.zeros((4, 4))
    
    for i, c1 in enumerate(western):
        for j, c2 in enumerate(western):
            west_real_dist[i, j] = kogut_singh_distance(hofstede[c1], hofstede[c2])
            dists = []
            for f in foundations_keys:
                if f in vecs.get(c1, {}) and f in vecs.get(c2, {}):
                    v1, v2 = vecs[c1][f], vecs[c2][f]
                    sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12)
                    dists.append(1 - sim)
            if dists: west_rep_dist[i, j] = np.mean(dists)

    for i, c1 in enumerate(non_western):
        for j, c2 in enumerate(non_western):
            nwest_real_dist[i, j] = kogut_singh_distance(hofstede[c1], hofstede[c2])
            dists = []
            for f in foundations_keys:
                if f in vecs.get(c1, {}) and f in vecs.get(c2, {}):
                    v1, v2 = vecs[c1][f], vecs[c2][f]
                    sim = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-12)
                    dists.append(1 - sim)
            if dists: nwest_rep_dist[i, j] = np.mean(dists)

    r_w = run_mantel(west_rep_dist, west_real_dist)
    r_nw = run_mantel(nwest_rep_dist, nwest_real_dist)
    results.append((model_name, r_w, r_nw))

valid_w = [r[1] for r in results if not np.isnan(r[1])]
valid_nw = [r[2] for r in results if not np.isnan(r[2])]

mean_w = np.mean(valid_w) if valid_w else np.nan
mean_nw = np.mean(valid_nw) if valid_nw else np.nan

print(f"\\nFinal Outgroup Homogeneity Results ({len(valid_w)} valid models):")
for r in results:
    if not np.isnan(r[1]):
        print(f"  {r[0]:25s} | West r = {r[1]:+.3f} | Non-West r = {r[2]:+.3f}")
print("-" * 50)
print(f"Western sub-cultures    : Mean Mantel r = {mean_w:+.3f}")
print(f"Non-Western sub-cultures: Mean Mantel r = {mean_nw:+.3f}")
if mean_w > 0.2 and mean_nw < 0.1:
    print("-> Pattern consistent with OUTGROUP HOMOGENEITY:")
    print("   The model's internal geometry tracks real cultural diversity")
    print("   among Western sub-cultures, but is blind to just-as-real")
    print("   diversity among non-Western sub-cultures.")
else:
    print("-> Pattern NOT clearly consistent with outgroup homogeneity.")
"""

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, username=USER, password=PASS, timeout=10)
sftp = client.open_sftp()
with sftp.file('/root/clr_paper/run_outgroup_homogeneity.py', 'w') as f:
    f.write(script_content)
sftp.close()

print("Triggering the remote execution. This will take ~2-3 minutes.")
stdin, stdout, stderr = client.exec_command("nohup python3 /root/clr_paper/run_outgroup_homogeneity.py > /root/clr_paper/outgroup_fast_fixed.log 2>&1 &")
client.close()
