# GraphRAG Industrial Upgrade Plan

## 🎯 Project Goal
Upgrade the current GraphRAG into an industrial-grade, deployable GraphRAG System v1.0.
The final deliverables must include:

    A complete layered architecture; 
    Hybrid retrieval; 
    Graph-based expansion; 
    Evidence-constrained generation; 
    An evaluation framework; 
    FastAPI service;
    Dockerized deployment;
    Stress/load testing report; 
    Engineering-level README

## 🏗 当前版本
v0.1 状态说明

## 🗺 Roadmap
- Phase1: ...
- Phase2: ...

## 📅 每周计划
### Week1-2：架构重构（基础打牢）
目标：项目结构工业化
Week1:  

    任务：
        1. 重构目录结构（模块化）
        2. 引入配置系统（yaml）
        3. 定义统一DocumentSchema
        4. 写build_index脚本
    产出：
        1. v0.1
        2. README重写（架构说明）
        3. 最小pipeline跑通
        
Week2:

    任务：
        1. ingestion模块完善（PDF/Markdown支持）
        2. chunking标准化
        3. metadata统一
        4. 写单元测试（最少10个）
    产出：
        1. ingestion稳定版本
        2. docs更新
        3. 测试通过截图

### Week3-4:图构建+Hybrid检索
目标：让它“真的叫GraphRAG”

Week3:

    任务：
        1. Section树构建
        2. 邻接边（next）
        3. 包含边（contains）
        4. 图导出（json/graphml）
    产出：
        1. v0.2
        2. 图可视化脚本
        3. 结构说明文档

Week4:

    任务：
        1. 加入BM25
        2. 向量检索+BM25融合（RRF）
        3. RetrievalRecall@k评测脚本
    产出：
        1. Hybrid检索实验表
        2. 对比报告

### Week5-6：GraphExpansion核心差异化
目标：打出你自己的技术特色

Week5:

    任务：
        1. 实现k-hop扩展
        2. 边权衰减机制
        3. 节点去噪（阈值/topm）
    产出：
        1. graph_expansion.py
        2. 可开关参数化设计

Week6:

    任务：
        1. no-graph vs graph对比实验
        2. recall提升分析
        3. 写技术博客草稿
    产出：
        1. v0.4
        2. 实验报告表格

### Week7-8：Evidence约束生成+评测体系
目标：让系统“可信”

Week7:

    任务：
        1. EvidenceConstrainedPrompt设计
        2. 强制输出：结论+证据chunk_id
        3. 上下文裁剪策略
    产出：
        1. v0.5
        2. prompt设计文档

Week8:

    任务：
        1. 引用校验机制
        2. Faithfulness检测
        3. Eval框架（EM/F1/Latency）
    产出：
        1. eval模块
        2. 自动生成评测报告
        3. 评测json输出


### Week9-10：服务化与部署
目标：工业味道拉满

Week9:

    任务：
        1. FastAPI接口
        2. /ingest
        3. /query
        4. /eval
        5. 日志结构化
    产出：
        1. 可访问API
        2. Swagger页面截图

Week10:

    任务：
        1. Dockerfile
        2. docker-compose
        3. GPU推理支持（可选vLLM）
        4. 一键启动脚本
        
    产出：
        1. v1.0-beta
        2. Quickstart指南


### Week11-12：性能优化+压测+打磨
目标：进入“面试展示级”

Week11:

    任务：
        1. locust压测
        2. p50/p95统计
        3. 缓存机制（embedding/query）
    产出：
        1. 压测报告
        2. 性能对比表

Week12:

    任务：
        1. README全面升级
        2. 架构图（draw.io）
        3. Benchmark表格
        4. 录一个2分钟demo视频
        5. GitHub置顶
    产出：
        1. GraphRAG v1.0 正式版


## 📊 当前指标
RetrievalRecall@k:
Latency:
F1:
...

## 🧠 决策记录
- 为什么选择RRF
- 为什么使用vLLM
...