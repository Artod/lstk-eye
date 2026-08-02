"""Voice-intent matching for session commands.

The click alphabet is deliberately tiny; rare commands ("back", "repeat",
"cancel") are spoken. A transcript counts as a command only when it is short -
anything longer is a new question and falls through as None.
"""

import re

_MAX_COMMAND_WORDS = 3

# Keyword phrases per intent, English and Russian. Single words match as whole
# words; multi-word phrases match as contiguous word sequences.
_INTENT_PHRASES: dict[str, tuple[str, ...]] = {
    "next": ("next", "forward", "дальше", "далее", "вперед", "вперёд"),
    "prev": ("back", "previous", "назад"),
    "repeat": ("repeat", "again", "повтори", "повторить", "еще раз", "ещё раз"),
    "cancel": ("cancel", "stop", "done", "exit", "quit", "отмена", "стоп", "хватит", "готово"),
}


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    cleaned = re.sub(r"[^\w\s]|_", " ", text.lower())
    return cleaned.split()


def match_intent(text: str) -> str | None:
    """Map a transcript to "next"/"prev"/"cancel"/"repeat", or None.

    Utterances longer than three words after normalization are never
    commands - they are treated as new questions.
    """
    words = _normalize(text)
    if not words or len(words) > _MAX_COMMAND_WORDS:
        return None
    joined = f" {' '.join(words)} "
    for intent, phrases in _INTENT_PHRASES.items():
        for phrase in phrases:
            if f" {phrase} " in joined:
                return intent
    return None


def is_calibration_request(text: str) -> bool:
    """A short utterance asking to (re)calibrate starts the calibration flow.

    Substring match on the stem so "calibrate", "calibration", "калибровка"
    and "откалибруй" all trigger; long sentences do not (the wearer would be
    asking ABOUT calibration, not requesting it).
    """
    words = _normalize(text)
    if not words or len(words) > _MAX_COMMAND_WORDS:
        return False
    return any(w.startswith(("calibrat", "калибр", "откалибр")) for w in words)
