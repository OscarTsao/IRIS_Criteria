"""Text chunking utilities for IRIS-based post processing."""

import re
from typing import List


def chunk_by_sentences(
    text: str,
    max_sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> List[str]:
    """
    Split text into overlapping chunks by sentences.

    Args:
        text: Input text to chunk
        max_sentences_per_chunk: Maximum sentences per chunk (default: 3)
        overlap_sentences: Number of sentences to overlap between chunks (default: 1)

    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences using regex
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)

    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    # If very short, return as single chunk
    if len(sentences) <= max_sentences_per_chunk:
        return [text]

    chunks = []
    stride = max_sentences_per_chunk - overlap_sentences

    # Ensure positive stride
    if stride <= 0:
        stride = 1

    for i in range(0, len(sentences), stride):
        chunk_sentences = sentences[i : i + max_sentences_per_chunk]

        if len(chunk_sentences) < 1:
            break

        chunk_text = " ".join(chunk_sentences)
        chunks.append(chunk_text)

        # Break if this chunk includes the last sentence
        if i + max_sentences_per_chunk >= len(sentences):
            break

    return chunks if chunks else [text]


def chunk_by_words(
    text: str,
    max_words: int = 50,
    overlap_words: int = 10,
) -> List[str]:
    """
    Split text into overlapping chunks by words.

    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk (default: 50)
        overlap_words: Number of words to overlap between chunks (default: 10)

    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    # Split into words
    words = text.split()

    if not words:
        return []

    # If short enough, return as single chunk
    if len(words) <= max_words:
        return [text]

    chunks = []
    stride = max_words - overlap_words

    # Ensure positive stride
    if stride <= 0:
        stride = max(1, max_words // 2)

    for i in range(0, len(words), stride):
        chunk_words = words[i : i + max_words]

        if len(chunk_words) < 5:  # Min chunk size
            break

        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)

        # Break if this chunk includes the last word
        if i + max_words >= len(words):
            break

    return chunks if chunks else [text]


def get_optimal_chunking_strategy(
    text: str,
    word_threshold: int = 30,
) -> str:
    """
    Determine optimal chunking strategy based on text length.

    Args:
        text: Input text to analyze
        word_threshold: Word count threshold for chunking decision (default: 30)

    Returns:
        Strategy name: "none", "sentence", or "word"
    """
    text = text.strip()
    if not text:
        return "none"

    word_count = len(text.split())

    # Very short: no chunking
    if word_count <= word_threshold:
        return "none"

    # Check for sentence structure
    sentence_terminators = ['.', '!', '?']
    has_sentences = any(term in text for term in sentence_terminators)

    if has_sentences:
        return "sentence"
    else:
        return "word"


def apply_chunking(
    text: str,
    strategy: str = "auto",
    sentence_params: dict = None,
    word_params: dict = None,
) -> List[str]:
    """
    Apply text chunking with specified or automatic strategy.

    Args:
        text: Input text to chunk
        strategy: Chunking strategy - "auto", "none", "sentence", or "word" (default: "auto")
        sentence_params: Parameters for sentence chunking (default: None)
        word_params: Parameters for word chunking (default: None)

    Returns:
        List of text chunks
    """
    sentence_params = sentence_params or {}
    word_params = word_params or {}

    # Auto-detect strategy if requested
    if strategy == "auto":
        strategy = get_optimal_chunking_strategy(text)

    # Apply chunking
    if strategy == "none":
        return [text] if text.strip() else []
    elif strategy == "sentence":
        return chunk_by_sentences(text, **sentence_params)
    elif strategy == "word":
        return chunk_by_words(text, **word_params)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
