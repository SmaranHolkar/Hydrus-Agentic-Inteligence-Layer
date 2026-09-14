"""
Lightweight Claim & Sentence Chunker for GBPS.

Splits context and generated responses into atomic claim candidates without requiring heavy NLP pipelines.
"""

from typing import List
import re


def split_claims(text: str) -> List[str]:
    """
    Splits text into atomic claim units / sentences, stripping markdown boilerplate.
    """
    if not text or not text.strip():
        return []

    # Clean markdown headers and bullet points
    cleaned = re.sub(r'^[#*\-+\d.]+\s+', '', text, flags=re.MULTILINE)

    # Split by standard sentence delimiters (. ! ? \n)
    sentences = re.split(r'(?<=[.!?])\s+|\n+', cleaned)

    claims: List[str] = []
    for s in sentences:
        s_clean = s.strip()
        # Filter out trivial fragments (< 10 chars)
        if len(s_clean) >= 10:
            claims.append(s_clean)

    return claims
