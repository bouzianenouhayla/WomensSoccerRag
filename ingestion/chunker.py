from dataclasses import dataclass
from typing import List

from .base import BaseChunker


@dataclass
class CharacterChunker(BaseChunker):
    """Splits text by character count with overlap."""

    max_chars: int = 800
    overlap_chars: int = 150

    def chunk_text(self, text: str) -> List[str]:
        """Split text into fixed-size character chunks with overlap.

        Args:
            text: Raw input text to split.

        Returns:
            List of chunk strings, each at most max_chars characters.
        """
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        length = len(text)
        step = max(self.max_chars - self.overlap_chars, 1)
        while start < length:
            end = min(start + self.max_chars, length)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == length:
                break
            start += step
        return chunks


# Keep old name as alias so nothing else breaks
Chunker = CharacterChunker
