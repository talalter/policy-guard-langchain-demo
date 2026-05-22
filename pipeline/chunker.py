"""
chunker.py - Token-aware document chunking for the RAG pipeline.

Uses tiktoken (cl100k_base, same tokenizer as GPT-4o) to split documents
into overlapping chunks. Token-aware splitting is preferable to character-
based splitting because:
  - It guarantees chunks fit within the model's context window.
  - Overlap_tokens lets adjacent chunks share context, preventing a sentence
    from being split across two chunks with no shared signal.
"""

import logging
from typing import Optional

import tiktoken # type: ignore

logger = logging.getLogger(__name__)

_ENCODING: Optional[tiktoken.Encoding] = None


def _get_encoding() -> tiktoken.Encoding:
    """Return the shared cl100k_base encoding, initializing it once."""
    global _ENCODING
    if _ENCODING is None:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def _sliding_window_chunks(
    tokens: list,
    max_tokens: int,
    step: int,
) -> list[list]:
    """Slide a window over a token list and return sublists."""
    chunks: list[list] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        start += step
    return chunks


def chunk_text(
    text: str,
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into overlapping token-bounded chunks.

    Args:
        text: Raw document text to split.
        max_tokens: Maximum tokens per chunk (default 300).
        overlap_tokens: Tokens shared between adjacent chunks (default 50).

    Returns:
        List of decoded chunk strings. Empty input returns an empty list.
    """
    if not text.strip():
        return []

    enc = _get_encoding()
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    step = max_tokens - overlap_tokens
    chunks = [enc.decode(tc) for tc in _sliding_window_chunks(tokens, max_tokens, step)]
    logger.debug(
        "Chunked %d tokens into %d chunks (max=%d, overlap=%d)",
        len(tokens), len(chunks), max_tokens, overlap_tokens,
    )
    return chunks


def chunk_documents(
    docs: dict[str, str],
    max_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[str]:
    """Chunk all documents in a {name: text} dict and return a flat list.

    Args:
        docs: Mapping of document name to raw text.
        max_tokens: Passed through to chunk_text.
        overlap_tokens: Passed through to chunk_text.

    Returns:
        Flat list of all chunks across all documents.
    """
    all_chunks: list[str] = []
    for name, text in docs.items():
        chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        logger.info("'%s': %d chunks", name, len(chunks))
        all_chunks.extend(chunks)
    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks
