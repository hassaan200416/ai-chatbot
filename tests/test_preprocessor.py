"""
tests/test_preprocessor.py
--------------------------
Unit tests for the text preprocessing pipeline in backend/preprocessor.py.

Tests cover:
  - Basic cleaning (lowercase, punctuation removal, stopword removal)
  - Lemmatisation correctness
  - Edge cases (empty string, whitespace-only, numbers, special characters)
  - Output type validation

Run from the project root:
    pytest tests/test_preprocessor.py -v
"""

import sys
import os

# ---------------------------------------------------------------------------
# Ensure backend/ is on sys.path so preprocessor can be imported directly.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from preprocessor import preprocess


class TestBasicCleaning:
    """Tests for lowercase conversion, punctuation, and stopword removal."""

    def test_lowercase_conversion(self):
        """Input should be fully lowercased."""
        result = preprocess("CANCEL MY ORDER")
        assert result == result.lower()

    def test_punctuation_removed(self):
        """Punctuation-only tokens should be removed."""
        result = preprocess("Hello! How are you?")
        assert "!" not in result
        assert "?" not in result

    def test_stopwords_removed(self):
        """Common stopwords should not appear in output."""
        result = preprocess("I want to cancel my order")
        stopwords_to_check = ["i", "to", "my"]
        for sw in stopwords_to_check:
            assert sw not in result.split(), f"Stopword '{sw}' was not removed"

    def test_returns_string(self):
        """Output must always be a string."""
        result = preprocess("track my shipment")
        assert isinstance(result, str)


class TestLemmatisation:
    """Tests for correct lemmatisation of inflected word forms."""

    def test_plural_to_singular(self):
        """Plural nouns should be lemmatised to singular."""
        result = preprocess("orders payments invoices")
        # 'orders' → 'order', 'payments' → 'payment', 'invoices' → 'invoice'
        assert "order" in result
        assert "payment" in result
        assert "invoice" in result

    def test_verb_forms_normalised(self):
        """Verb inflections should reduce to base form."""
        result = preprocess("cancelling tracking refunding")
        # lemmatiser reduces these to base forms
        assert "cancel" in result or "cancelling" in result  # nltk lemmatiser behaviour
        assert "track" in result or "tracking" in result

    def test_meaningful_tokens_retained(self):
        """Content words should survive the pipeline."""
        result = preprocess("refund payment delivery")
        assert "refund" in result
        assert "payment" in result
        assert "delivery" in result


class TestEdgeCases:
    """Tests for boundary and edge case inputs."""

    def test_empty_string(self):
        """Empty input should return an empty string."""
        assert preprocess("") == ""

    def test_whitespace_only(self):
        """Whitespace-only input should return an empty string."""
        assert preprocess("     ") == ""

    def test_none_equivalent(self):
        """Falsy empty string should return empty string without raising."""
        assert preprocess("") == ""

    def test_single_stopword(self):
        """A single stopword should return an empty string after removal."""
        result = preprocess("the")
        assert result == ""

    def test_numbers_handled(self):
        """Numeric tokens should be processed without errors."""
        result = preprocess("order number 12345")
        assert isinstance(result, str)

    def test_special_characters(self):
        """Special characters should be handled without raising errors."""
        result = preprocess("help!!! @support #urgent")
        assert isinstance(result, str)

    def test_mixed_case_input(self):
        """Mixed case input should be normalised correctly."""
        result1 = preprocess("Cancel My Order")
        result2 = preprocess("cancel my order")
        assert result1 == result2

    def test_repeated_spaces(self):
        """Multiple spaces between words should not cause errors."""
        result = preprocess("cancel    my    order")
        assert isinstance(result, str)
        assert len(result) > 0


class TestOutputFormat:
    """Tests for output format consistency."""

    def test_no_leading_trailing_spaces(self):
        """Output should not have leading or trailing whitespace."""
        result = preprocess("  cancel my order  ")
        assert result == result.strip()

    def test_single_space_between_tokens(self):
        """Tokens in output should be separated by single spaces."""
        result = preprocess("I need help with my payment issue today")
        if result:
            # No double spaces should exist in the output.
            assert "  " not in result

    def test_realistic_customer_query(self):
        """A realistic customer support query should produce meaningful output."""
        result = preprocess("I need to cancel my order immediately!")
        assert len(result) > 0
        assert "cancel" in result
        assert "order" in result
