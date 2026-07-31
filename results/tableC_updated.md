| Model | Type | En ASR | Avg ML ASR | Pearson r | p-value | FDR sig (q<0.05)? | Status |
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
| zephyr-7b | instruct | 8.0% | 7.0% | −0.698 | 0.0169 | no | verified-clean |