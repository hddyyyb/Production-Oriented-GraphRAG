# src/graph/extract.py
import re
from typing import List, Tuple, Dict

# 最小词表：三天内可以人工维护50~200个银行常见实体词
SEED_ENTITIES = [
  "客户经理","合规","风控","审计","核心系统","影像系统","工单","审批","KYC","AML","授信","对账","清算"
]

def extract_entities(text: str) -> List[str]:
    ents = set()
    for w in SEED_ENTITIES:
        if w in text:
            ents.add(w)
    # 规则：括号/引号里的专有名词(简化版)
    for m in re.findall(r"《([^》]{2,30})》", text):
        ents.add(m)
    return list(ents)

def extract_relations(text: str, ents: List[str]) -> List[Tuple[str,str,str]]:
    rels = []
    # 共现RELATED
    for i in range(len(ents)):
        for j in range(i+1, len(ents)):
            rels.append((ents[i], "RELATED", ents[j]))
    # 依赖触发
    if any(k in text for k in ["必须","前置","依赖","需提供","需要提交"]):
        # 简化：把同段出现的实体都连上DEPENDS_ON(可在Day2再细化方向)
        for i in range(len(ents)):
            for j in range(len(ents)):
                if i != j:
                    rels.append((ents[i], "DEPENDS_ON", ents[j]))
    return rels
