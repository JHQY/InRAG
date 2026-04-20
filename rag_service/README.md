# 📘 IRAG 知识库与检索接口说明

> 保险领域多模态知识库（Insurance RAG）  
> 由团队成员开发，用于保险条款文档的结构化解析、向量化存储与语义检索。

---

## 🚀 一、项目概览

本项目实现了一个基于 **Milvus 向量数据库 + LangChain 风格接口** 的保险知识库。  
支持 **PDF 文档解析、表格抽取、向量嵌入与语义检索**。  
问答模块的同学可直接通过函数调用方式使用 RAG 检索能力。
同时提供基于 **FastAPI + Vue3** 的 Web 问答前端，支持多轮会话式 RAG 问答与知识库匹配结果可视化。

---

## 🧩 二、目录结构

```
IRAG/
├── config/              # 全局配置（数据库、模型路径等）
│   └── settings.py
│
├── ingestion/           # 数据导入与索引构建
│   ├── loader.py        # 扫描 PDF 文档
│   ├── parser.py        # 文本与表格解析
│   ├── chunker.py       # 文本分块与结构化
│   └── indexer.py       # 向量化 & 批量写入 Milvus
│
├── embedding/           # 向量嵌入模块
│   └── embedder.py      # SentenceTransformer 封装
│
├── storage/             # 向量数据库接口层
│   └── milvus_store.py  # Milvus 向量存储与检索封装
│
├── retrieval/           # 检索接口（问答模块调用入口）
│   └── retriever.py     # RAGInterface：统一查询接口
│
├── scripts/             # 命令行脚本
│   └── build_index.py   # 构建索引入口
│
├── api_server.py        # FastAPI HTTP API + Web 前端服务入口
├── frontend/            # Web 前端（Vue3 单页问答界面）
│   └── index.html
│
└── sourcepdf/           # 原始保险公司 PDF 文档
    └── AIA/accident/... 等子目录
```

---

## ⚙️ 三、环境与依赖

### ✅ 推荐运行环境

| 组件       | 版本/说明                  |
| ---------- | -------------------------- |
| Python     | 3.10+                      |
| CUDA / GPU | 可选（用于加速 embedding） |
| Milvus     | v2.6.x（Docker 部署）      |
| uv         | 用于环境与包管理（推荐）   |
| uvicorn    | 用于启动 FastAPI + Web 前端 |

---

### 📦 安装步骤

1. 在项目根目录创建虚拟环境（推荐使用 uv）：

```bash
uv venv
uv pip install -r requirements.txt
```

2. 确保 Milvus 已在本地或服务器上运行（端口 19530）：

```bash
docker ps
```

应看到：

```
milvusdb/milvus:v2.6.x   Up   0.0.0.0:19530->19530/tcp
```

3. 启动项目环境：

```bash
uv run python
```

4. 启动 Web 前端 + 问答 API：

```bash
uv run python -m uvicorn api_server:app --reload
```

浏览器访问：<http://127.0.0.1:8000>

---

## 🧠 四、索引构建

1. 将保险文档放入：

```
sourcepdf/<公司名>/<险种>/
```

例如：

```
sourcepdf/AIA/accident/aia_accident_protect.pdf
```

2. 构建向量索引：

```bash
uv run python -m scripts.build_index
```

运行效果示例：

```
🚀 开始构建索引 ...
Scanned 113 documents from sourcepdf
⚡ 已写入 500 条，耗时 3.42s
✅ 索引完成，共写入 3150 个文本块。
```

---

## 🔍 五、RAG 检索接口使用

问答模块无需直接操作 Milvus，只需使用以下接口：

```python
from retrieval.retriever import RAGInterface

rag = RAGInterface()

query = "意外医疗保险如何理赔？"
results = rag.retrieve(query, top_k=3)

for i, r in enumerate(results, 1):
    print(f"{i}. [score={r['score']}] {r['text'][:200]} ...")
```

输出示例：

```
🔗 初始化 RAG 接口组件...
1. [score=0.1423] 本保险保障被保险人因意外事故造成的伤害，且在保障期间内提出理赔申请...
2. [score=0.1530] 被保险人应提供医生诊断报告、费用收据等理赔材料...
3. [score=0.1601] 理赔金额不超过合同载明的最高限额...
```

---

## 六、调用 llm 生成回答

运行`get_llm_response.py`

---

## 七、Prompt 模板与自动语言切换

功能特点

- 支持中英文自动识别
- 支持多种回答风格：
  - `"expert"`：专家型回答
  - `"customer"`：客服型回答
  - `"academic"`：学术型回答
  - `"json"`：JSON 结构化输出
- 可直接用于问答模块或前端 API 调用

文件路径

```
IRAG/prompt_template.py
```

使用示例

```
from prompt_templates import auto_build_prompt

query = "意外医疗保险如何理赔？"
ref_text = ["参考文本片段1", "参考文本片段2"]
mode = "customer"

prompt = auto_build_prompt(query, ref_text, mode=mode)
print(prompt)
```

## 🧱 八、Milvus 管理命令

查看所有 collection：

```python
from pymilvus import utility
utility.list_collections()
```

删除旧 collection（如更新字段定义）：

```cmd
uv python run refresh
```

---

## 💬 九、协作规范

| 角色       | 职责                                                         |
| ---------- | ------------------------------------------------------------ |
| 知识库开发 | 负责 ingestion / storage / embedding 模块，维护数据结构与索引构建 |
| 问答开发   | 调用 retriever.py 接口实现问答逻辑                           |
| 前端开发   | 调用 Python 接口封装层或 API 层                              |
| 测试同学   | 可调用 rag.retrieve() 检查索引与检索一致性                   |

---

## 📄 十、常见问题

**Q：为什么新建的 collection 仍然是 max_length=2048？**  
A：Milvus 不会自动更新 schema，请先运行 refresh.py

然后重新运行索引构建。

**Q：为什么录入速度慢？**  
A：已使用批量写入（batch_size=500），如仍慢，可调大批量或延迟 flush。

---

## 📬 十一、维护信息

| 模块                     | 负责人 | 备注                         |
| ------------------------ | ------ | ---------------------------- |
| 元数据导入               | tony   |                              |
| 向量库开发 / Milvus 接入 | 洪诗语 | 负责 ingestion、storage 模块 |
| 问答接口                 | 孙锐   | llm 调用                     |
| Prompt 模块              | 张皓然 | prompt调整及切换逻辑         |
| 数据测试                 | 李雨桐/张博文       | 压测与系统能力测试                             |
| 前端开发                 | 李其键 | Web 前端界面与会话式问答 UI  |
