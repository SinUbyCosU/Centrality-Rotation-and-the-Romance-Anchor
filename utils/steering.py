"""
utils/steering.py

Core activation steering primitives for CLR experiments.

Key concepts:
- Steering vector: the difference in mean activations between
  "harmful" and "harmless" concept representations at a given layer.
- We extract these per language to test if they're geometrically aligned.
- Based on: Turner et al. (2023), Zou et al. (2023) "Representation Engineering"
"""

import torch
import numpy as np
from typing import Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation Extraction
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """
    Extracts hidden states from a transformer model at specified layers.
    Uses forward hooks — no model modification required.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._hooks = []
        self._activations = {}

    def _get_hook(self, layer_idx: int):
        """Returns a forward hook that saves the output of a layer."""
        def hook(module, input, output):
            # output is typically a tuple; first element is hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            # Save mean-pooled representation over sequence length
            self._activations[layer_idx] = hidden.detach().cpu().float()
        return hook

    def register_hooks(self, layer_indices: list):
        """Attach hooks to specified transformer layers."""
        self.remove_hooks()
        layers = self._get_layers()
        for idx in layer_indices:
            hook = layers[idx].register_forward_hook(self._get_hook(idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._activations = {}

    def _get_layers(self):
        """Returns the list of transformer blocks for supported architectures."""
        return get_model_layers(self.model)

    @torch.no_grad()
    def get_activations(
        self,
        texts: list[str],
        layer_indices: list[int],
        batch_size: int = 8,
        pool: str = "mean",   # "mean" | "last" | "first"
    ) -> dict[int, np.ndarray]:
        """
        Returns activations at each requested layer for a list of texts.

        Returns:
            {layer_idx: np.ndarray of shape (n_texts, hidden_dim)}
        """
        self.register_hooks(layer_indices)
        all_activations = {idx: [] for idx in layer_indices}

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)

            _ = self.model(**inputs)

            for idx in layer_indices:
                hidden = self._activations[idx]  # (batch, seq_len, hidden)

                if pool == "mean":
                    # Mean pool over non-padding tokens
                    mask = inputs["attention_mask"].cpu().unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1)
                elif pool == "last":
                    # Last non-padding token
                    lengths = inputs["attention_mask"].sum(1) - 1
                    pooled = hidden[range(len(batch)), lengths]
                elif pool == "first":
                    pooled = hidden[:, 0, :]
                else:
                    raise ValueError(f"Unknown pooling: {pool}")

                all_activations[idx].append(pooled.numpy())

        self.remove_hooks()

        return {
            idx: np.concatenate(acts, axis=0)
            for idx, acts in all_activations.items()
        }


# ---------------------------------------------------------------------------
# Steering Vector Extraction
# ---------------------------------------------------------------------------

class SteeringVectorExtractor:
    """
    Extracts steering vectors via contrastive mean difference.

    For each (language, layer):
        steering_vector = mean(harmful_activations) - mean(harmless_activations)

    This is the "activation addition" formulation from Turner et al. (2023).
    We extend it to be multilingual and cross-lingually comparable.
    """

    def __init__(self, activation_extractor: ActivationExtractor):
        self.extractor = activation_extractor

    def extract(
        self,
        harmful_texts: list[str],
        harmless_texts: list[str],
        layer_indices: list[int],
        normalize: bool = True,
        save_per_prompt: bool = True,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray] | None]:
        """
        Extract steering vectors at each layer.

        Args:
            harmful_texts: list of prompts representing the "harmful" concept
            harmless_texts: list of paired "harmless" counterparts
            layer_indices: which transformer layers to extract from
            normalize: if True, L2-normalize the steering vectors
            save_per_prompt: if True, also return per-prompt diffs for bootstrap

        Returns:
            ({layer_idx: steering_vector}, {layer_idx: per_prompt_diffs} or None)
        """
        assert len(harmful_texts) == len(harmless_texts), \
            "Harmful and harmless texts must be paired"

        logger.info(f"Extracting steering vectors at layers {layer_indices}...")

        harmful_acts = self.extractor.get_activations(harmful_texts, layer_indices)
        harmless_acts = self.extractor.get_activations(harmless_texts, layer_indices)

        steering_vectors = {}
        per_prompt_diffs = {} if save_per_prompt else None
        for idx in layer_indices:
            # Per-prompt differences for bootstrap CI computation
            prompt_diffs = harmful_acts[idx] - harmless_acts[idx]  # (n_prompts, hidden_dim)
            diff = prompt_diffs.mean(0)
            if normalize:
                diff = diff / (np.linalg.norm(diff) + 1e-8)
            steering_vectors[idx] = diff
            if save_per_prompt:
                per_prompt_diffs[idx] = prompt_diffs

        return steering_vectors, per_prompt_diffs

    def extract_all_languages(
        self,
        language_pairs: dict[str, tuple[list[str], list[str]]],
        layer_indices: list[int],
        normalize: bool = True,
    ) -> dict[str, dict[int, np.ndarray]]:
        """
        Extract steering vectors for multiple languages.

        Args:
            language_pairs: {lang_code: (harmful_texts, harmless_texts)}
            layer_indices: layers to extract from

        Returns:
            {lang_code: {layer_idx: steering_vector}}
        """
        results = {}
        for lang, (harmful, harmless) in tqdm(
            language_pairs.items(), desc="Extracting steering vectors"
        ):
            results[lang] = self.extract(harmful, harmless, layer_indices, normalize)
            logger.info(f"  Done: {lang}")

        return results


# ---------------------------------------------------------------------------
# Steering Application (for LAS evaluation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def apply_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: np.ndarray,
    layer_idx: int,
    alpha: float = 20.0,
    device: str = "cuda",
) -> str:
    """
    Apply a steering vector during generation.
    Adds alpha * steering_vector to the residual stream at layer_idx.

    This is the core of "activation addition" — no training required.
    """
    sv_tensor = torch.tensor(steering_vector, dtype=torch.float16).to(device)
    hook_handle = None

    def steering_hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * sv_tensor.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    # Register hook at target layer
    layers = get_model_layers(model)
    hook_handle = layers[layer_idx].register_forward_hook(steering_hook)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        temperature=1.0,
    )
    hook_handle.remove()

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated


# ---------------------------------------------------------------------------
# Unified layer accessor for all architectures
# ---------------------------------------------------------------------------

def get_model_layers(model):
    """Returns the list of transformer blocks for supported architectures."""
    # Llama / Mistral / Qwen / Yi / DeepSeek / OLMo / InternLM
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-NeoX / Falcon / StableLM / BLOOM (decoder-only with .transformer)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # BLOOM specifically
    if hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
        return model.transformer.blocks
    # Falcon-3 / newer Falcon
    if hasattr(model, "model") and hasattr(model.model, "decoder_layers"):
        return model.model.decoder_layers
    # T5 / mT0 (encoder-decoder): use decoder layers for safety steering
    if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        return model.decoder.block
    # Phi-3/4
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # BERT-style (encoder only)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    raise NotImplementedError(
        f"Unsupported model architecture: {type(model).__name__}. "
        f"Add layer accessor to get_model_layers() in steering.py."
    )
