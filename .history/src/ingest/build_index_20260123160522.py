# src/ingest/build_index.py
import json, os
import numpy as np
import faiss

def build_faiss(chunks_path: str, index_dir: str, embed_fn):
    os.makedirs(index_dir, exist_ok=True)
    chunks = [json.loads(l) for l in open(chunks_path, "r", encoding="utf-8")]
    vecs = []
    for c in chunks:
        v = embed_fn(c["text"])  # 返回1D list/np.array
        vecs.append(v)
    X = np.array(vecs, dtype="float32")
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(X)
    index.add(X)
    faiss.write_index(index, os.path.join(index_dir, "chunks.faiss"))
    with open(os.path.join(index_dir, "chunks.meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
