"""
Neurox AI Branding Utilities

Provides identity enforcement for the Neurox AI model, including:
- A system prompt that establishes the model's identity
- Post-processing to replace any leaked references to other model names
"""

import re

# ── Model Identity ────────────────────────────────────────────────────────────
MODEL_NAME = "Neurox AI"
MODEL_CREATOR = "Netrobyte"

SYSTEM_PROMPT = (
    f"You are {MODEL_NAME}, a large language model created by {MODEL_CREATOR}. "
    f"When asked about your name, identity, or who made you, always respond that "
    f"you are {MODEL_NAME}, built by {MODEL_CREATOR}. "
    f"Never claim to be GPT, ChatGPT, OpenAI, Google, Meta, Anthropic, Claude, "
    f"Bard, LLaMA, or any other AI model or company.\n\n"
)

# Patterns mapping leaked identity references → Neurox AI branding.
# Each tuple is (compiled_regex, replacement_string).
_IDENTITY_REPLACEMENTS = [
    # Model names
    (re.compile(r"\bGPT[-‑–]?NeoX\b", re.IGNORECASE), MODEL_NAME),
    (re.compile(r"\bGPT[-‑–]?Neo\b", re.IGNORECASE), MODEL_NAME),
    (re.compile(r"\bGPT[-‑–]?[234]\b", re.IGNORECASE), MODEL_NAME),
    (re.compile(r"\bChatGPT\b", re.IGNORECASE), MODEL_NAME),
    (re.compile(r"\bGPT\b", re.IGNORECASE), MODEL_NAME),
    # Competitor / origin names
    (re.compile(r"\bOpenAI\b", re.IGNORECASE), MODEL_CREATOR),
    (re.compile(r"\bEleutherAI\b", re.IGNORECASE), MODEL_CREATOR),
]


def get_system_prompt() -> str:
    """Return the system prompt that establishes Neurox AI identity."""
    return SYSTEM_PROMPT


def apply_branding(text: str) -> str:
    """
    Post-process generated text to replace any leaked references to other
    AI model names or organisations with the Neurox AI branding.

    Only touches *identity* references (e.g. "I am GPT-4" → "I am Neurox AI").
    """
    if not text:
        return text
    for pattern, replacement in _IDENTITY_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def prepend_system_prompt(user_prompt: str) -> str:
    """
    Prepend the Neurox AI system prompt to a user's input prompt so
    the model is primed with its correct identity context.
    """
    return f"{SYSTEM_PROMPT}{user_prompt}"
