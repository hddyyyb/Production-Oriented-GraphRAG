# scripts/demo.py
from src.rag.answer import answer
from src.ingest.build_index import build_faiss
from src.ingest.embedder import embed_text

# 加载chunks和索引
chunks_meta = json.load(open("data/processed/chunks_meta.json"))
build_faiss("data/processed/chunks.jsonl", "data/index", embed_text)

# 问答
question = "What is the approval process for loans?"
result = answer(question, embed_text, chunks_meta, llm_fn=llm_mock_function)

print(result)
