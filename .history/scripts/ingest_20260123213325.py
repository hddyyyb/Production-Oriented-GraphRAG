# scripts/ingest.py

import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 获取项目根目录
if root_path not in sys.path:   # 将项目根目录添加到 sys.path 中
    sys.path.append(root_path)
import json
from src.ingest.chunker import simple_chunk    # 处理文档切块

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 定义读取文件的函数
def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 定义嵌入模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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


# FAISS 索引构建并保存
def build_faiss(embeddings, index_dir='data/index'):
    # 确保索引文件夹存在
    os.makedirs(index_dir, exist_ok=True)
    
    # 转换为 numpy 数组
    embeddings = np.array(embeddings, dtype="float32")
    
    # 创建 FAISS 索引
    dim = embeddings.shape[1]  # 嵌入的维度
    index = faiss.IndexFlatL2(dim)  # 使用 L2 距离度量
    
    # 将嵌入向量添加到索引中
    index.add(embeddings)
    
    # 保存索引到文件
    faiss.write_index(index, os.path.join(index_dir, 'chunks.faiss'))
    
    # 保存元数据文件
    with open(os.path.join(index_dir, 'chunks_meta.json'), 'w', encoding='utf-8') as f:
        json.dump([chunk.__dict__ for chunk in all_chunks], f, ensure_ascii=False)

# 构建并保存 FAISS 索引
build_faiss(embeddings)

print("文档切块、向量化和 FAISS 索引构建完成！")