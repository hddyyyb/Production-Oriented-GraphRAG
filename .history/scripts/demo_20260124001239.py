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

# 示例问题
question = "What is the approval process for loans?"

# 调用 answer 函数生成答案
result = answer(question, embed_text, chunks_meta, llm_fn=llm_mock_function)

# 打印答案和引用
print("Answer:", result["answer"])
print("Citations:", result["citations"])
print("Graph Context:", result["graph_context"])
