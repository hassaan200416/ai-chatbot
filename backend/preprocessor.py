"""
backend/preprocessor.py
-----------------------
NLP text preprocessing pipeline for the AI Chatbot.

Workflow:
  1. Lowercase the raw input text.
  2. Tokenise into individual words using NLTK's word_tokenize.
  3. Remove punctuation-only tokens and English stopwords.
  4. Lemmatise each remaining token using WordNetLemmatizer.
  5. Return the cleaned tokens joined back into a single string.

This module is imported by:
  - ml/train_nb.py   → to clean training data before TF-IDF fitting
  - ml/train_ann.py  → same
  - backend/predictor.py → to clean user input before prediction

Keeping preprocessing in one place guarantees that training and
inference always apply identical transformations.
"""

import string

from nltk.tokenize import word_tokenize  # type: ignore
from nltk.corpus import stopwords  # type: ignore
from nltk.stem import WordNetLemmatizer  # type: ignore

# ---------------------------------------------------------------------------
# Module-level singletons — instantiated once to avoid repeated overhead.
# ---------------------------------------------------------------------------
_lemmatizer = WordNetLemmatizer()
_stop_words: set[str] = set(stopwords.words("english"))  # pyright: ignore[reportUnknownMemberType]


def preprocess(text: str) -> str:
    """
    Clean and normalise a raw text string for NLP feature extraction.

    Steps applied in order:
      1. Lowercase
      2. Tokenise (NLTK punkt tokeniser)
      3. Remove punctuation-only tokens
      4. Remove English stopwords
      5. Lemmatise

    Args:
        text (str): Raw input string (user message or training utterance).

    Returns:
        str: Space-joined string of cleaned, lemmatised tokens.
             Returns an empty string if input is empty or whitespace-only.

    Examples:
        >>> preprocess("I need to cancel my order immediately!")
        'need cancel order immediately'
        >>> preprocess("What are the payment options available?")
        'payment option available'
    """
    if not text or not text.strip():
        return ""

    # Step 1 — Lowercase for case-insensitive matching.
    text = text.lower()

    # Step 2 — Tokenise into a list of word strings.
    tokens = word_tokenize(text)

    # Step 3 — Drop tokens that are entirely punctuation
    #           (e.g. "!", ".", ",") — they carry no intent signal.
    tokens = [t for t in tokens if t not in string.punctuation]

    # Step 4 — Remove stopwords (e.g. "the", "is", "at") to reduce noise.
    tokens = [t for t in tokens if t not in _stop_words]

    # Step 5 — Lemmatise: reduces inflected forms to their base form
    #           (e.g. "cancelling" → "cancel", "orders" → "order").
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)
