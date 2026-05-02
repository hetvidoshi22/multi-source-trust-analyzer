"""
chunking.py
Content chunking utility.
Splits long text into smaller, overlapping or non-overlapping chunks
suitable for downstream processing (LLMs, search indexing, etc.).
"""

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 600,
    overlap: int = 50,
    min_chunk_length: int = 80,
) -> List[str]:
    """
    Split text into chunks of approximately `chunk_size` characters.

    Strategy:
    1. Try to split on paragraph boundaries (double newlines) first.
    2. If a paragraph is too long, split further on sentence boundaries.
    3. If still too long, split on word boundaries.

    Args:
        text:             Input text to chunk
        chunk_size:       Target maximum characters per chunk
        overlap:          Number of characters to overlap between chunks
        min_chunk_length: Minimum characters for a chunk to be included

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Edge case: text shorter than chunk_size
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk_length else []

    # Step 1: Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If paragraph fits in current chunk, add it
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = (current_chunk + "\n\n" + para).strip()
        else:
            # Save current chunk if non-empty
            if current_chunk and len(current_chunk) >= min_chunk_length:
                chunks.append(current_chunk)

            # If paragraph itself is within chunk_size, start a new chunk
            if len(para) <= chunk_size:
                current_chunk = para
            else:
                # Paragraph is too long → split on sentences
                sentence_chunks = _split_on_sentences(para, chunk_size, min_chunk_length)
                if sentence_chunks:
                    # Add all but last as completed chunks
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = para[:chunk_size]

    # Add the last chunk
    if current_chunk and len(current_chunk) >= min_chunk_length:
        chunks.append(current_chunk)

    # Apply optional overlap
    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    return chunks


def _split_on_sentences(text: str, chunk_size: int, min_chunk_length: int) -> List[str]:
    """Split text on sentence boundaries."""
    # Split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current and len(current) >= min_chunk_length:
                chunks.append(current)
            if len(sentence) <= chunk_size:
                current = sentence
            else:
                # Sentence itself is too long → split on words
                word_chunks = _split_on_words(sentence, chunk_size, min_chunk_length)
                chunks.extend(word_chunks[:-1])
                current = word_chunks[-1] if word_chunks else ""

    if current and len(current) >= min_chunk_length:
        chunks.append(current)

    return chunks


def _split_on_words(text: str, chunk_size: int, min_chunk_length: int) -> List[str]:
    """Last-resort: split on word boundaries."""
    words = text.split()
    chunks = []
    current_words = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 <= chunk_size:
            current_words.append(word)
            current_len += len(word) + 1
        else:
            chunk = " ".join(current_words)
            if len(chunk) >= min_chunk_length:
                chunks.append(chunk)
            current_words = [word]
            current_len = len(word)

    last = " ".join(current_words)
    if len(last) >= min_chunk_length:
        chunks.append(last)

    return chunks if chunks else [text[:chunk_size]]


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """
    Add overlap between consecutive chunks.
    Each chunk (except first) gets the tail of the previous chunk prepended.
    """
    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:].strip()
        overlapped.append(tail + " " + chunks[i] if tail else chunks[i])
    return overlapped


def chunk_transcript(
    transcript_segments: list,
    chunk_size: int = 600,
) -> List[str]:
    """
    Chunk YouTube transcript segments (list of dicts with 'text' and 'start').
    Groups segments by time window rather than character count.

    Args:
        transcript_segments: List of {"text": str, "start": float, "duration": float}
        chunk_size:          Target characters per chunk

    Returns:
        List of text chunks
    """
    if not transcript_segments:
        return []

    chunks = []
    current_text = ""

    for seg in transcript_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if len(current_text) + len(text) + 1 <= chunk_size:
            current_text = (current_text + " " + text).strip()
        else:
            if current_text:
                chunks.append(current_text)
            current_text = text

    if current_text:
        chunks.append(current_text)

    return chunks


if __name__ == "__main__":
    # Test with a long sample text
    sample = """
    Artificial intelligence is transforming every industry. From healthcare to finance,
    AI models are being deployed at scale. Machine learning engineers are in high demand.

    Deep learning, a subset of machine learning, uses neural networks with many layers.
    These networks learn hierarchical representations from raw data. Convolutional neural
    networks (CNNs) are especially effective for image recognition tasks.

    Natural language processing (NLP) deals with the interaction between computers and
    human language. Large language models like GPT and BERT have revolutionized NLP.
    They are pre-trained on massive text corpora and fine-tuned for specific tasks.

    Data scraping is the process of automatically extracting information from websites.
    Web scrapers use HTTP requests and HTML parsers to navigate page structures and
    extract relevant content. Python libraries like BeautifulSoup and Scrapy make this easy.
    """ * 3  # Repeat to make it long

    chunks = chunk_text(sample, chunk_size=300)
    print(f"Generated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk[:120] + "..." if len(chunk) > 120 else chunk)
