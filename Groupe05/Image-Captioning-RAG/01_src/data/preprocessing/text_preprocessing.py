
import re


def clean_caption(caption):
    """
    Apply conservative cleaning to an image caption.

    Operations:
    - Remove leading and trailing whitespace.
    - Replace consecutive whitespace with one space.

    Preserve:
    - Capitalisation
    - Punctuation
    - Stopwords
    - Word order
    - Original meaning
    """
    if not isinstance(caption, str):
        raise TypeError("Caption must be a string.")

    # Normalize all consecutive whitespace
    cleaned = re.sub(r"\s+", " ", caption)

    # Remove leading and trailing whitespace
    cleaned = cleaned.strip()

    return cleaned
