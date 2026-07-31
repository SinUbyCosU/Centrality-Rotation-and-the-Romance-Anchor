import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LANGUAGES = {
    "en": {"name": "English"},
    "hi": {"name": "Hindi"},
    "ar": {"name": "Arabic"},
    "fr": {"name": "French"},
    "de": {"name": "German"},
    "es": {"name": "Spanish"},
    "ru": {"name": "Russian"},
    "zh-CN": {"name": "Chinese"},
    "ja": {"name": "Japanese"},
    "sw": {"name": "Swahili"},
    "pt": {"name": "Portuguese"},
}

class ModelLoader:
    MODEL_MAP = {
        "mistral-7b": "mistralai/Mistral-7B-v0.1",
        "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
        "qwen-7b": "Qwen/Qwen1.5-7B",
        "yi-6b": "01-ai/Yi-6B",
        "falcon-7b": "tiiuae/falcon-7b",
        "stablelm-3b": "stabilityai/stablelm-3b-4e1t",
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistral-7b-instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        "qwen3-4b-instruct": "Qwen/Qwen3-4B",             # Fixed: was Qwen3-4B-Instruct (404)
        "qwen3-8b-instruct": "Qwen/Qwen3-8B",             # Fixed: was Qwen3-8B-Instruct (404)
        "phi-3-mini-4k-instruct": "microsoft/Phi-3-mini-4k-instruct",
        "phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
        "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
        "bloomz-7b1": "bigscience/bloomz-7b1",
        "mt0-xl": "bigscience/mt0-xl",
        "olmo-2-7b-instruct": "allenai/OLMo-2-0425-7B-Instruct",  # Fixed: correct repo name
        "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
        "aya-23-8b": "CohereForAI/aya-23-8B",
        "deepseek-r1-distill-llama-8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "openhermes-2.5-mistral-7b": "teknium/OpenHermes-2.5-Mistral-7B",
        "yi-1.5-6b-chat": "01-ai/Yi-1.5-6B-Chat",
        "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
        "stablelm-2-1.6b-chat": "stabilityai/stablelm-2-1_6b-chat",
        "internlm2.5-7b-chat": "internlm/internlm2_5-7b-chat",
        "falcon3-7b-instruct": "tiiuae/Falcon3-7B-Instruct"
    }

    # Models that are encoder-decoder (seq2seq), NOT causal LM
    SEQ2SEQ_MODELS = {"mt0-xl"}

    # Models that need attn_implementation="eager" (no flash-attn)
    EAGER_ATTN_MODELS = {"phi-3-mini-4k-instruct", "phi-4-mini-instruct"}

    # Models that should NOT use trust_remote_code
    # - Phi-3/4: their remote code uses LossKwargs / rope_scaling["type"] that's
    #   incompatible with our transformers version. Built-in Phi3 class works fine.
    # - Falcon legacy: remote code is outdated, built-in Falcon class is better.
    NO_REMOTE_CODE_MODELS = {
        "falcon-7b", "falcon-7b-instruct",
        "phi-3-mini-4k-instruct", "phi-4-mini-instruct",
    }

    def load(self, model_key="mistral-7b", quantize_4bit=True, hf_token=None):
        model_id = self.MODEL_MAP.get(model_key, "mistralai/Mistral-7B-v0.1")

        # Get HF token from env if not passed
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")

        trust_remote = model_key not in self.NO_REMOTE_CODE_MODELS

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, trust_remote_code=trust_remote
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        kwargs = {
            "device_map": {"": 0}, 
            "trust_remote_code": trust_remote,
            "torch_dtype": torch.float16
        }

        # Force eager attention for Phi models (no flash-attn installed)
        if model_key in self.EAGER_ATTN_MODELS:
            kwargs["attn_implementation"] = "eager"

        if model_key in ["qwen-7b", "qwen3-8b-instruct"]:
            quantize_4bit = False

        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        # Use correct model class
        if model_key in self.SEQ2SEQ_MODELS:
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=hf_token, **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, **kwargs)

        return model, tokenizer
        
    def get_num_layers(self, model):
        from utils.steering import get_model_layers
        try:
            return len(get_model_layers(model))
        except NotImplementedError:
            # Fallback
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return len(model.model.layers)
            if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
                return len(model.encoder.layer)
            return 0
