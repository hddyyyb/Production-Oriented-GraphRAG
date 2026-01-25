import os
import sys
import json
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
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
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch

# 1) LLM（回答用）
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
llm_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

tokenizer.pad_token = tokenizer.eos_token
llm_model.config.pad_token_id = tokenizer.eos_token_id


# 使用 GPT-Neo 来生成答案
def llm_gpt_neo_function(prompt: str) -> str:
    max_new_tokens = 150
    max_ctx = 2048
    buffer = 8  # 留一点余量防止边界问题
    max_input_tokens = max_ctx - max_new_tokens - buffer

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,   # 硬截断输入
        padding=True
    )

    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 可选：去掉 prompt 回显
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

# 加载切块元数据
with open("data/index/chunks_meta.json", "r", encoding="utf-8") as f:
    chunks_meta = json.load(f)

# 加载FAISS索引
index_path = "data/index/chunks.faiss"
index = faiss.read_index(index_path)

# 向量化函数（用于在demo中检索相似片段）
# 2) Embedder（检索用）
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    return embed_model.encode(text)

# 问答示例
# 领域：贷款（loan）
# 行为：审批流程（approval process）
# 预期答案结构：步骤型 / 流程型
question = "What is the approval process for loans?"

# 使用FAISS进行向量检索
def faiss_search(query, index, top_k=5):
    query_vec = embed_text(query)  # 获取查询的嵌入向量
    query_vec = np.array([query_vec], dtype='float32')  # 转换为二维数组，符合FAISS要求
    
    distances, indices = index.search(query_vec, top_k)  # 使用FAISS检索最相似的片段
    return distances, indices


distances, indices = faiss_search(question, index)  # 执行检索

retrieved_chunks = [chunks_meta[i] for i in indices[0]]  # 获取检索到的片段并格式化

context = "\n".join([chunk['text'] for chunk in retrieved_chunks])  # 构建上下文，提供给LLM模型

# 打印检索到的相关片段
print("Retrieved Chunks:\n")
for chunk in retrieved_chunks:
    print(f"Chunk ID: {chunk['chunk_id']}\nText: {chunk['text']}\n")

# 调用答案生成函数（此处模拟调用实际LLM）
result = answer(question, tokenizer, embed_text, chunks_meta, index, llm_fn=llm_gpt_neo_function)

# 打印最终答案
print("\nFinal Answer:")
print(result["answer"])
print("\nCitations:", result["citations"])
print("\nEvidence Pack:")
for e in result["graph_context"].get("evidence", []):
    print(e)

