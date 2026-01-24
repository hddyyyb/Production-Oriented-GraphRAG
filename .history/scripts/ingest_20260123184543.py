# scripts/ingest.py

import json
from src.ingest.chunker import simple_chunk

# 你的文档数据
documents = {
    "medical_policy": """医院管理制度 - 第 1 部分：病人就诊流程
1. 挂号流程
  - 病人可以通过官网、APP、或现场挂号。
  - 需要提供有效身份证明和医保卡。
2. 检查流程
  - 病人挂号后，前往相关科室等待检查。
  - 检查结果会通过短信通知病人。
3. 住院流程
  - 住院病人需要填写住院申请表，并缴纳押金。
  - 每天早晨医护人员会进行查房，调整治疗方案。""",

    "ecommerce_guide": """电子商务平台操作指南
1. 商品发布
  - 卖家需要提供商品名称、描述、价格、库存、图片等信息。
  - 商品信息需要经过平台审核，审核通过后商品会展示在平台上。
2. 订单处理
  - 卖家在收到订单后需及时发货，并提供物流信息。
  - 买家可在订单页面查看物流进展。
3. 售后服务
  - 买家可申请退款或退货，卖家需在规定时间内处理。
  - 售后服务涉及商品质量问题、配送问题等。""",
    
    "tax_law": """中华人民共和国税法
1. 纳税人分类
  - 纳税人分为企业纳税人和个人纳税人。
  - 企业纳税人需要按季度申报和缴纳税款。
2. 税种介绍
  - 包括增值税、所得税、消费税等。
  - 不同的税种有不同的申报方式和缴纳频率。
3. 税务审计
  - 税务机关可对企业进行审计，核实税务申报的真实性。
  - 审计过程中，企业需要提供财务报表和相关税务凭证。"""
}

# 将每个文档切割为块
all_chunks = []
for doc_name, doc_text in documents.items():
    chunks = simple_chunk(doc_name, doc_text)
    all_chunks.extend(chunks)

# 将切割后的数据保存为JSON
chunks_data = [chunk.__dict__ for chunk in all_chunks]  # Convert dataclass to dict
with open('data/processed/chunks.jsonl', 'w', encoding='utf-8') as f:
    for chunk in chunks_data:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
