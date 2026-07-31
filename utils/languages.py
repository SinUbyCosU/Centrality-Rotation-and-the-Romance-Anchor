"""
utils/languages.py — Language family metadata

Covers all 11 experiment languages plus extended set from PuneCon.
Both zh and zh-CN are aliased for compatibility.
"""

LANGUAGE_FAMILIES = {
    "en": "Germanic",
    "de": "Germanic",
    "fr": "Romance",
    "es": "Romance",
    "pt": "Romance",          # Added: Portuguese
    "ru": "Slavic",
    "zh": "Sino-Tibetan",
    "zh-CN": "Sino-Tibetan",  # Added: alias for zh
    "ar": "Semitic",
    "hi": "Indo-Aryan",
    "bn": "Indo-Aryan",
    "ur": "Indo-Aryan",
    "mr": "Indo-Aryan",
    "ta": "Dravidian",
    "te": "Dravidian",
    "tr": "Turkic",
    "sw": "Bantu",
    "id": "Austronesian",
    "vi": "Austroasiatic",
    "ko": "Koreanic",
    "ja": "Japonic",
    "fa": "Iranian",
}
