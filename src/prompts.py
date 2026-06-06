PROMPT_TEMPLATES = {
    "bare": "Summarize the following text:\n\n{content}",
    "polite": "Please summarize the following text:\n\n{content}",
    "overly_polite": (
        "Hey, when you get a chance, could you please summarize the following "
        "text for me? Thank you.\n\n{content}"
    ),
}


def build_prompt(content: str, variant: str) -> str:
    if variant not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {list(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[variant].format(content=content)
