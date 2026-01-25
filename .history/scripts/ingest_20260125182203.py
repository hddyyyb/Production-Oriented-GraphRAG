# scripts/ingest.py

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 获取项目根目录
if root_path not in sys.path:   # 将项目根目录添加到 sys.path 中
    sys.path.append(root_path)
import json
from src.ingest.chunker import simple_chunk    # 处理文档切块
from src.ingest.build_index import build_faiss
from sentence_transformers import SentenceTransformer

# 定义读取文件的函数
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 定义嵌入模型
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 文档文件夹路径
raw_docs_path = 'data/raw_docs'



# 遍历raw_docs文件夹中的所有txt文件，读取内容并切块
documents = {}
for filename in os.listdir(raw_docs_path):
    if filename.endswith(".txt"):
        # 获取文件名不包括后缀
        doc_id = filename.replace('.txt', '')
        
        # 读取文件内容
        file_path = os.path.join(raw_docs_path, filename)
        doc_text = read_txt_file(file_path)
        
        # 将文档内容加入字典
        documents[doc_id] = doc_text

# 将每个文档切割为块
all_chunks = []
for doc_name, doc_text in documents.items():
    chunks = simple_chunk(doc_name, doc_text)
    all_chunks.extend(chunks)


# 提取每个块的文本并进行嵌入（将文本转换为向量）
embeddings = []
for chunk in all_chunks:
    embeddings.append(model.encode(chunk.text))


# 将切割后的数据保存为JSON
chunks_data = [chunk.__dict__ for chunk in all_chunks]  # Convert dataclass to dict
with open('data/processed/chunks.jsonl', 'w', encoding='utf-8') as f:
    for chunk in chunks_data:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

# 调用 build_faiss 函数构建并保存 FAISS 索引
build_faiss(embeddings, chunks_data)

print("文档切块、向量化和 FAISS 索引构建完成！")

import networkx as nx
from src.graph.build_graph import build_graph

chunks_meta_path = os.path.join("data", "index", "chunks_meta.json")
G = build_graph(chunks_meta_path)

graph_path = os.path.join("data", "index", "graph.gpickle")
nx.write_gpickle(G, graph_path)

print(f"Graph saved to: {graph_path}")
