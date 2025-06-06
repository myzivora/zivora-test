"""Emotion scoring utilities for Mercury's AI core."""

from enum import Enum
from string import punctuation
from typing import Dict, Set


class Sentiment(Enum):
    """Represents the supported sentiment states."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class EmotionScorer:
    """Keyword-based sentiment analyzer.

    The class provides a simple, dependency-free way to gauge text
    sentiment. Future sentiment categories can be accommodated by adding
    a keyword bank and extending ``_KEYWORD_BANK``.
    """

    _POSITIVE_KEYWORDS: Set[str] = {
        "happy",
        "joy",
        "love",
        "excited",
        "great",
        "fantastic",
        "good",
        "wonderful",
        "delight",
    }
    _NEGATIVE_KEYWORDS: Set[str] = {
        "sad",
        "angry",
        "hate",
        "terrible",
        "horrible",
        "bad",
        "upset",
        "awful",
        "depressing",
    }

    _KEYWORD_BANK: Dict[Sentiment, Set[str]] = {
        Sentiment.POSITIVE: _POSITIVE_KEYWORDS,
        Sentiment.NEGATIVE: _NEGATIVE_KEYWORDS,
    }

    @classmethod
    def _clean_text(cls, text: str) -> str:
        """Return a lowercase string with punctuation removed."""
        translator = str.maketrans('', '', punctuation)
        return text.lower().translate(translator)

    @classmethod
    def score(cls, text: str) -> Sentiment:
        """Return the sentiment associated with *text*.

        The method searches for keywords from each sentiment's bank and
        determines which sentiment dominates the text. When no keywords
        are found or the text is empty, :class:`Sentiment.NEUTRAL` is
        returned.
        """
        if not text or not text.strip():
            return Sentiment.NEUTRAL

        cleaned = cls._clean_text(text)
        tokens = cleaned.split()
        counts = {
            sentiment: sum(token in keywords for token in tokens)
            for sentiment, keywords in cls._KEYWORD_BANK.items()
        }

        max_sentiment = max(counts, key=counts.get)
        if counts[max_sentiment] == 0:
            return Sentiment.NEUTRAL

        # Check for ties to maintain determinism
        if list(counts.values()).count(counts[max_sentiment]) > 1:
            return Sentiment.NEUTRAL

        return max_sentiment


__all__ = ["EmotionScorer", "Sentiment"]
