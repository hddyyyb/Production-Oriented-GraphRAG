import os
import sys
import json

# 获取项目根目录并添加到sys.path中
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # 获取项目根目录
if root_path not in sys.path:
    sys.path.append(root_path)

# 从项目模块中导入所需函数
from src.rag.answer import answer
from src.ingest.build_index import build_faiss
from src.ingest.embedder import embed_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 模拟 LLM 函数，用于生成答案
def llm_mock_function(prompt):
    # 这个函数仅用于模拟问答处理，实际可以调用真正的LLM API
    return "This is a mock answer. Please replace with actual LLM inference."

# 加载切块元数据
with open("data/index/chunks_meta.json", "r", encoding="utf-8") as f:
    chunks_meta = json.load(f)

# 加载FAISS索引
index_path = "data/index/chunks.faiss"
index = faiss.read_index(index_path)

# 向量化函数（用于在demo中检索相似片段）
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def embed_text(text):
    return model.encode(text)

# 问答示例
question = "What is the approval process for loans?"

# 使用FAISS进行向量检索
def faiss_search(query, index, top_k=5):
    query_vec = embed_text(query)  # 获取查询的嵌入向量
    query_vec = np.array([query_vec], dtype='float32')  # 转换为二维数组，符合FAISS要求
    
    # 使用FAISS检索最相似的片段
    distances, indices = index.search(query_vec, top_k)
    return distances, indices

# 执行检索
distances, indices = faiss_search(question, index)

# 获取检索到的片段并格式化
retrieved_chunks = [chunks_meta[i] for i in indices[0]]

# 构建上下文，提供给LLM模型
context = "\n".join([chunk['text'] for chunk in retrieved_chunks])

# 打印检索到的相关片段
print("Retrieved Chunks:\n")
for chunk in retrieved_chunks:
    print(f"Chunk ID: {chunk['chunk_id']}\nText: {chunk['text']}\n")

# 调用答案生成函数（此处模拟调用实际LLM）
result = answer(question, embed_text, chunks_meta, llm_fn=llm_mock_function)

# 打印最终答案
print("\nFinal Answer:")
print(result)
