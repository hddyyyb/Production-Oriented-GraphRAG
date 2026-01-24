# src/ingest/chunker.py
# Split the input document text into multiple smaller chunks, each with a maximum length of max_len, and with an overlap of overlap characters. You can adjust these two parameters based on the actual situation.

from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str

def simple_chunk(doc_id: str, text: str, section: str = "", max_len: int = 800, overlap: int = 120) -> List[Chunk]:
    chunks = []
    i = 0
    n = len(text)
    k = 0
    while i < n:
        j = min(n, i + max_len)
        chunk_text = text[i:j]
        chunks.append(Chunk(chunk_id=f"{doc_id}_{k}", doc_id=doc_id, text=chunk_text, section=section))
        k += 1
        i = j - overlap
        if i < 0:
            i = 0
        if j == n:
            break
    return chunks
