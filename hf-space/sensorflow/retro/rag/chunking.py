"""Document parsing and configurable chunking.

Documents are markdown-ish text with `## section` headings. The parser splits
into sections (keeping section titles as metadata), and the chunker splits
long sections into overlapping windows measured in words.
"""

from __future__ import annotations

import re
from typing import Dict, List

from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, str]


def parse_sections(text: str) -> List[Dict[str, str]]:
    """Split a document into {section, text} blocks on '## ' headings."""
    sections: List[Dict[str, str]] = []
    current_title = "preamble"
    current_lines: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.*)", line)
        if m:
            if current_lines and "".join(current_lines).strip():
                sections.append({"section": current_title,
                                 "text": "\n".join(current_lines).strip()})
            current_title = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines and "".join(current_lines).strip():
        sections.append({"section": current_title,
                         "text": "\n".join(current_lines).strip()})
    return sections


def chunk_words(text: str, chunk_size: int = 160, overlap: int = 40) -> List[str]:
    """Overlapping word-window chunking. chunk_size/overlap are in words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()] if text.strip() else []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(doc_id: str, text: str, base_metadata: Dict[str, str],
                   chunk_size: int = 160, overlap: int = 40) -> List[Chunk]:
    """Parse into sections then chunk each section, propagating metadata."""
    chunks: List[Chunk] = []
    for si, sec in enumerate(parse_sections(text)):
        for ci, piece in enumerate(chunk_words(sec["text"], chunk_size, overlap)):
            meta = dict(base_metadata)
            meta["section"] = sec["section"]
            chunks.append(Chunk(chunk_id=f"{doc_id}:s{si}:c{ci}", text=piece,
                                metadata=meta))
    return chunks
