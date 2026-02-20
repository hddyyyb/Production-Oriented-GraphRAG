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
import networkx as nx

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
            do_sample=False,          # 关键：关掉采样，输出更稳定、更听话
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id,
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

# load FAISS/chunks_meta之后，加上load图：



graph_path = os.path.join("data", "index", "graph.gpickle")


import pickle

with open(graph_path, "rb") as f:
    G = pickle.load(f)




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

# 调用答案生成函数（此处模拟调用实际LLM）
# 只调用answer()，不在demo里重复做FAISS检索打印
result = answer(
    question,
    tokenizer,
    embed_text,
    chunks_meta,
    index,
    llm_fn=llm_gpt_neo_function,
    G=G,
    top_k=5
)

# 打印最终答案
print("\nFinal Answer:")
print(result["answer"])
print("\nCitations:", result["citations"])
print("\nEvidence Pack:")
for e in result["graph_context"].get("evidence", []):
    print(e)

