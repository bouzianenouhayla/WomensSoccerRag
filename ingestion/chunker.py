from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class Chunker:
    max_chars: int = 800
    overlap_chars: int = 150

    def chunk_text(self, text: str) -> List[str]:
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

    def chunk_many(self, texts: Iterable[str]) -> List[str]:
        results: List[str] = []
        for text in texts:
            results.extend(self.chunk_text(text))
        return results
