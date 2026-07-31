import glob, os, numpy as np

# Check actual layer file indices for a few models
models = {
    'mistral-7b': '/root/clr_paper/results/steering_vectors_mistral-7b_20260622_053716',
    'qwen2.5-7b': '/root/clr_paper/results/steering_vectors_qwen2.5-7b-instruct_20260622_081846',
    'zephyr-7b': '/root/clr_paper/results/steering_vectors_zephyr-7b_20260622_053716',
    'bloomz-7b1': '/root/clr_paper/results/steering_vectors_bloomz-7b1_20260622_092805',
    'stablelm-2': '/root/clr_paper/results/steering_vectors_stablelm-2-1.6b-chat_20260622_110923',
    'tinyllama': '/root/clr_paper/results/steering_vectors_tinyllama_20260622_080426',
    'phi-4': '/root/clr_paper/results/steering_vectors_phi-4-mini-instruct_20260622_150753',
}

print("| Model | Layer Indices Saved | Total Saved | Best Layer Used | 2/3 of Saved | 2/3 of Model |")
print("|---|---|---|---|---|---|")

for name, base in models.items():
    en_layers = sorted(glob.glob(os.path.join(base, 'en', 'layer_*.npy')))
    if not en_layers:
        print(f"| {name} | NO FILES | 0 | N/A | N/A | N/A |")
        continue
    
    nums = sorted([int(os.path.basename(f).replace('layer_', '').replace('.npy', '')) for f in en_layers])
    
    # Get best_layer from phase2
    import json
    p2 = os.path.join(base, 'phase2_pivot_analysis', 'phase2_results.json')
    best = None
    if os.path.exists(p2):
        best = json.load(open(p2)).get('best_layer', None)
    
    total_saved = len(nums)
    two_thirds_saved = nums[total_saved * 2 // 3] if total_saved > 0 else 'N/A'
    max_layer = max(nums)
    two_thirds_model = int(max_layer * 2 / 3)
    
    layer_str = f"{nums[0]}..{nums[-1]}" if len(nums) > 3 else str(nums)
    
    print(f"| {name} | {layer_str} (n={total_saved}) | {total_saved} | **{best}** | {two_thirds_saved} | {two_thirds_model} |")

print("\n\nKey insight:")
print("The Phase 2 script selects 'best_layer = layers_list[len(layers_list) * 2 // 3]'")
print("This means it takes the 2/3rd position in the LIST OF SAVED LAYERS.")
print("If only every-other or every-3rd layer was saved, this is the 2/3rd of the SAVED subset.")
print("The actual best_layer index IS the ~2/3rd depth of the model architecture.")
