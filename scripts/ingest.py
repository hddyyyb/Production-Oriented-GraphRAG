# scripts/ingest.py

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:   # Get project root directory + add to sys.path
    sys.path.append(root_path)
import json
from src.ingest.chunker import simple_chunk
from src.ingest.build_index import build_faiss
from sentence_transformers import SentenceTransformer

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # embedding model

raw_docs_path = 'data/raw_docs'     # Path to document folder

# Traverse all txt files in the raw_docs folder, read content and chunk
documents = {}
for filename in os.listdir(raw_docs_path):
    if filename.endswith(".txt"):
        doc_id = filename.replace('.txt', '')
        file_path = os.path.join(raw_docs_path, filename)
        doc_text = read_txt_file(file_path)
        documents[doc_id] = doc_text

# Split document into chunks
all_chunks = []
for doc_name, doc_text in documents.items():
    chunks = simple_chunk(doc_name, doc_text)
    all_chunks.extend(chunks)


embeddings = []
for chunk in all_chunks:
    embeddings.append(model.encode(chunk.text))


# Save chunked data as JSON
chunks_data = [chunk.__dict__ for chunk in all_chunks]
with open('data/processed/chunks.jsonl', 'w', encoding='utf-8') as f:
    for chunk in chunks_data:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

# Call build_faiss function to construct and save FAISS index
build_faiss(embeddings, chunks_data)

from src.graph.build_graph import build_graph

chunks_meta_path = os.path.join("data", "index", "chunks_meta.json")
G = build_graph(chunks_meta_path)

graph_path = os.path.join("data", "index", "graph.gpickle")
import pickle

with open(graph_path, "wb") as f:
    pickle.dump(G, f)

print(f"Graph saved to: {graph_path}")


