PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "summarize": {
        "bare": "Summarize the following text:\n\n{content}",
        "polite": "Please summarize the following text:\n\n{content}",
        "overly_polite": (
            "Hey, when you get a chance, could you please summarize the following "
            "text for me? Thank you.\n\n{content}"
        ),
    },
    "solve": {
        "bare": "Solve the following problem:\n\n{content}",
        "polite": "Please solve the following problem:\n\n{content}",
        "overly_polite": (
            "Hey, when you get a chance, could you please solve the following "
            "problem for me? I'd really appreciate it, thank you!\n\n{content}"
        ),
    },
}


def build_prompt(content: str, variant: str, style: str = "summarize") -> str:
    if style not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown style: {style!r}. Must be one of {list(PROMPT_TEMPLATES)}")
    templates = PROMPT_TEMPLATES[style]
    if variant not in templates:
        raise ValueError(f"Unknown variant: {variant!r}. Must be one of {list(templates)}")
    return templates[variant].format(content=content)
