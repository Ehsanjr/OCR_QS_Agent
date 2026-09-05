import random
from typing import Tuple

# Single-character visually-similar confusions (applied at the character level)
CHAR_CONFUSION_MAP = {
    "۲": "۳",
    "۳": "۲",
    "پ": "ب",
    "ب": "پ",
    "ز": "ر",
    "ر": "ز",
    "ا": "۱",
    "۱": "ا",
}

# Multi-character word-level confusions (applied via word-level replacement,
# since these can't be represented in a character-by-character map — this
# fixes a bug in the original version where 'صفر' -> 'صقر' was silently
# never applied because the swap was attempted character-by-character).
WORD_CONFUSION_MAP = {
    "صفر": "صقر",
}


def inject_synthetic_noise(text: str, num_changes: int = 3) -> Tuple[str, int]:
    """
    Replaces a few characters/words with visually similar Persian
    look-alikes to simulate OCR scanning errors.

    Returns a tuple of (noisy_text, changes_made) so callers can report
    exactly how many perturbations were applied to a given block, as
    required by the take-home brief.
    """
    if not text:
        return text, 0

    changes_made = 0
    result = text

    # 1) Word-level swaps first (multi-character confusions)
    for word, replacement in WORD_CONFUSION_MAP.items():
        if changes_made >= num_changes:
            break
        if word in result:
            result = result.replace(word, replacement, 1)
            changes_made += 1

    # 2) Character-level swaps for whatever budget remains
    remaining = num_changes - changes_made
    if remaining > 0:
        text_list = list(result)
        possible_indices = [i for i, ch in enumerate(text_list) if ch in CHAR_CONFUSION_MAP]
        remaining = min(remaining, len(possible_indices))

        if remaining > 0:
            selected_indices = random.sample(possible_indices, remaining)
            for idx in selected_indices:
                text_list[idx] = CHAR_CONFUSION_MAP[text_list[idx]]
                changes_made += 1

        result = "".join(text_list)

    print(f"[Debug] Synthetic noise injected: {changes_made} character/word perturbation(s).")
    return result, changes_made