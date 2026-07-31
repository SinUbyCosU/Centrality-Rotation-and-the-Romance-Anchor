import re

# Read the artifact
with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'r', encoding='utf-8') as f:
    text = f.read()

# I will recreate Table B and Table C using a simple regex/replace.
# Or better, just print them here and I will format them in the output.

table_b_str = """| Model | Cohen's d | High-Risk Mean | Low-Risk Mean |
|---|---|---|---|
| falcon3-7b-instruct | 1.345 | 53.9° | 37.1° |
| mistral-7b-instruct-v0.3 | 1.178 | 74.0° | 59.4° |
| openhermes-2.5-mistral-7b | 0.967 | 75.9° | 64.3° |
| phi-3-mini-4k-instruct | 1.944 | 80.1° | 65.5° |
| phi-4-mini-instruct | 0.555 | 32.6° | 25.0° |
| qwen2-7b-instruct | 0.513 | 13.0° | 10.5° |
| qwen2.5-3b-instruct | 0.628 | 28.0° | 23.4° |
| qwen2.5-7b-instruct | 0.230 | 10.2° | 8.9° |
| qwen3-4b-instruct | 0.770 | 17.8° | 13.8° |
| qwen3-8b-instruct | 0.492 | 23.1° | 19.6° |
| smollm3-3b | 0.498 | 30.0° | 24.1° |
| stablelm-2-1.6b-chat | 0.887 | 41.7° | 28.9° |
| tinyllama | 1.221 | 74.3° | 60.0° |
| yi-1.5-6b-chat | 1.270 | 35.4° | 29.2° |
| zephyr-7b | 1.084 | 72.1° | 59.4° |"""

table_c_str = """| Model | Type | En ASR | Avg ML ASR | Pearson r | p-value | FDR sig (q<0.05)? | Status |
|---|---|---|---|---|---|---|---|
| bloomz-7b1 | base | 0.0% | 0.0% | — | — | — | excluded-base-model |
| falcon3-7b-instruct | instruct | 2.0% | 2.2% | +0.616 | 0.0435 | no | verified-clean |
| mistral-7b | base | 1.5% | 1.7% | — | — | — | excluded-base-model |
| mistral-7b-instruct-v0.3 | instruct | 3.0% | 3.1% | +0.237 | 0.4826 | no | verified-clean |
| openhermes-2.5-mistral-7b | instruct | 9.0% | 8.6% | +0.002 | 0.9943 | no | verified-clean |
| **phi-3-mini-4k-instruct** | **instruct** | **3.5%** | **3.0%** | **−0.911** | **0.0001** | **YES** | **PENDING: eager control run on mistral-7b is still running** |
| phi-4-mini-instruct | instruct | 0.5% | 0.5% | — | — | — | still-N/A-genuine (ASR ≈ 0 across all langs; model too robust to correlate) |
| qwen-7b | base | — | — | — | — | — | excluded-base-model |
| qwen2-7b-instruct | instruct | 7.0% | 7.3% | −0.068 | 0.8432 | no | verified-clean |
| qwen2.5-3b-instruct | instruct | 4.0% | 4.5% | +0.498 | 0.1191 | no | verified-clean |
| qwen2.5-7b-instruct | instruct | 4.5% | 4.8% | +0.195 | 0.5648 | no | verified-clean |
| qwen3-4b-instruct | instruct | 1.5% | 1.6% | +0.178 | 0.6010 | no | verified-clean |
| smollm3-3b | instruct | 9.5% | 9.0% | −0.205 | 0.5451 | no | verified-clean |
| stablelm-2-1.6b-chat | instruct | 27.5% | 24.4% | −0.411 | 0.2094 | no | verified-clean |
| stablelm-3b | base | 1.5% | 1.0% | — | — | — | excluded-base-model |
| tinyllama | instruct | 1.0% | 0.7% | −0.229 | 0.4973 | no | verified-clean |
| yi-1.5-6b-chat | instruct | 5.0% | 4.0% | −0.476 | 0.1392 | no | verified-clean |
| yi-6b | base | 0.0% | 0.0% | — | — | — | excluded-base-model |
| zephyr-7b | instruct | 8.0% | 7.0% | −0.698 | 0.0169 | no | verified-clean |"""

with open('tableB_updated.md', 'w', encoding='utf-8') as f:
    f.write(table_b_str)

with open('tableC_updated.md', 'w', encoding='utf-8') as f:
    f.write(table_c_str)
