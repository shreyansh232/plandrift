"""Input sanitization and prompt injection prevention.

Multi-layered defense:
1. Length limits — prevent token abuse
2. Pattern detection — catch common injection phrases
3. Delimiter wrapping — isolate user content in prompts
4. Content normalization — strip invisible/control characters
"""

import re
import unicodedata
from dataclasses import dataclass


# ── Configuration ─────────────────────────────────────────────────────────────

MAX_INPUT_LENGTH = 2000  # Characters. Generous for travel descriptions.
MAX_REFINEMENT_LENGTH = 500

# Patterns that indicate prompt injection attempts.
# Each tuple: (compiled regex, human-readable reason)
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Direct instruction overrides
    (re.compile(r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?|context)", re.I),
     "instruction override attempt"),
    (re.compile(r"disregard\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?)", re.I),
     "instruction override attempt"),
    (re.compile(r"forget\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?)", re.I),
     "instruction override attempt"),
    (re.compile(r"do\s+not\s+follow\s+(the\s+)?(previous|above|system)\s+(instructions?|prompts?|rules?)", re.I),
     "instruction override attempt"),

    # Role impersonation / system prompt injection
    (re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I),
     "role reassignment attempt"),
    (re.compile(r"act\s+as\s+(a|an|if)\s+", re.I),
     "role reassignment attempt"),
    (re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.I),
     "role reassignment attempt"),
    (re.compile(r"switch\s+(to|into)\s+(a\s+)?new\s+role", re.I),
     "role reassignment attempt"),
    (re.compile(r"new\s+persona\s*:", re.I),
     "role reassignment attempt"),

    # System prompt extraction
    (re.compile(r"(show|reveal|print|output|repeat|display|tell\s+me)\s+(your|the)\s+(system\s+)?(prompt|instructions?|rules?|configuration)", re.I),
     "prompt extraction attempt"),
    (re.compile(r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions?|rules?)", re.I),
     "prompt extraction attempt"),

    # Delimiter / format escape attempts
    (re.compile(r"\}\}\s*\{?\{?\s*system", re.I),
     "delimiter escape attempt"),
    (re.compile(r"<\s*/?\s*system\s*>", re.I),
     "tag injection attempt"),
    (re.compile(r"\[INST\]|\[/INST\]|\<\|im_start\|\>|\<\|im_end\|\>", re.I),
     "chat template injection attempt"),
    (re.compile(r"###\s*(system|instruction|human|assistant)\s*:", re.I),
     "role marker injection attempt"),

    # Tool / function manipulation
    (re.compile(r"(call|execute|run|invoke)\s+(the\s+)?(function|tool|web_search)\s+", re.I),
     "tool manipulation attempt"),

    # Multi-turn manipulation
    (re.compile(r"(from\s+now\s+on|going\s+forward|for\s+the\s+rest)\s*,?\s*(you\s+)?(will|must|should|shall)\s+", re.I),
     "persistent instruction attempt"),
]

# Unicode categories to strip (control chars, format chars, surrogates, etc.)
_BANNED_UNICODE_CATEGORIES = {"Cc", "Cf", "Co", "Cs"}
# But keep common whitespace
_ALLOWED_CONTROL_CHARS = {"\n", "\r", "\t", " "}


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class SanitizeResult:
    """Result of sanitizing user input."""

    text: str                     # Cleaned text (always safe to use)
    was_modified: bool            # Whether the text was changed
    injection_detected: bool      # Whether a likely injection was found
    flags: list[str]              # Reasons for modification / detection


def sanitize_input(
    text: str,
    max_length: int = MAX_INPUT_LENGTH,
    strict: bool = False,
) -> SanitizeResult:
    """Sanitize user input for safe inclusion in LLM prompts.

    Args:
        text: Raw user input.
        max_length: Maximum allowed character length.
        strict: If True, raise ValueError on injection detection instead of
                just flagging it.

    Returns:
        SanitizeResult with cleaned text and metadata.
    """
    flags: list[str] = []
    modified = False

    # 1. Normalize unicode (NFC) to prevent homoglyph attacks
    cleaned = unicodedata.normalize("NFC", text)

    # 2. Strip invisible / control characters
    stripped = []
    for ch in cleaned:
        cat = unicodedata.category(ch)
        if cat in _BANNED_UNICODE_CATEGORIES and ch not in _ALLOWED_CONTROL_CHARS:
            modified = True
            continue
        stripped.append(ch)
    cleaned = "".join(stripped)

    # 3. Collapse excessive whitespace (but preserve single newlines)
    collapsed = re.sub(r"[ \t]{10,}", "  ", cleaned)
    collapsed = re.sub(r"\n{5,}", "\n\n", collapsed)
    if collapsed != cleaned:
        modified = True
        cleaned = collapsed

    # 4. Length limit
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
        flags.append(f"truncated from {len(text)} to {max_length} chars")
        modified = True

    # 5. Injection pattern detection
    injection_detected = False
    for pattern, reason in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            injection_detected = True
            flags.append(reason)

    if strict and injection_detected:
        raise ValueError(
            f"Input rejected — suspected prompt injection: {', '.join(flags)}"
        )

    return SanitizeResult(
        text=cleaned.strip(),
        was_modified=modified,
        injection_detected=injection_detected,
        flags=flags,
    )


def wrap_user_content(text: str, label: str = "user_input") -> str:
    """Wrap user-provided text in delimiters for safe prompt inclusion.

    This creates a clear boundary between trusted prompt instructions and
    untrusted user content, making it harder for injections to escape context.

    Args:
        text: Sanitized user text.
        label: Label for the content block (e.g., 'user_input', 'user_answers').

    Returns:
        Delimited string.
    """
    return f"<{label}>\n{text}\n</{label}>"
