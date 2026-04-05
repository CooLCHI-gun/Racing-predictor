from __future__ import annotations

import html
import re


def normalize_name(name: str) -> str:
    """
    Normalize common display artifacts in horse/runner names.

    This helper is intentionally conservative: when normalization is not
    clearly safe, it returns the original input unchanged.
    """
    if not isinstance(name, str):
        return name

    original = name

    # Decode HTML entities and normalize non-breaking/full-width spaces.
    cleaned = html.unescape(name)
    cleaned = cleaned.replace("\xa0", " ").replace("\u3000", " ")

    # Remove control characters except regular whitespace.
    cleaned = "".join(ch for ch in cleaned if ch >= " " or ch in "\n\t")

    # Collapse repeated whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Remove obvious stray question marks at boundaries.
    cleaned = re.sub(r"^\?+\s*", "", cleaned)
    cleaned = re.sub(r"\s*\?+$", "", cleaned)
    cleaned = cleaned.strip()

    # If cleanup produced an empty string, keep original for safety.
    if not cleaned:
        return original

    return cleaned


def normalize_display_name(name: str) -> str:
    """
    Conservative cleanup for display names.

    Removes only clearly technical trailing noise patterns while preserving
    potentially meaningful suffixes when uncertain.
    """
    if not isinstance(name, str):
        return name

    original = name
    cleaned = normalize_name(name)

    # Remove trailing empty brackets: "Name ()" -> "Name"
    cleaned = re.sub(r"\s*\(\s*\)\s*$", "", cleaned)

    # Remove explicit trailing technical slash codes: "Name /A", "Name /B"
    cleaned = re.sub(r"\s+/\s*(A|B)\s*$", "", cleaned)

    # Remove explicit trailing technical role tags: "Name-JR", "Name-TR"
    cleaned = re.sub(r"\s*-(JR|TR)\s*$", "", cleaned)

    cleaned = re.sub(r"\s+$", "", cleaned)

    # If cleanup became empty, keep original to avoid destructive changes.
    if not cleaned:
        return original

    return cleaned
