# LangChain Agentic RAG

一个可直接运行的本地知识库问答项目。它不是固定的“检索后回答”链：LangChain Agent 会根据问题自主决定是否调用检索工具，也可以通过不同关键词重复检索，再基于检索结果给出带来源的回答。

## 功能

- 支持 `data/` 下的 `.md`、`.txt`、`.pdf` 文档入库
- 默认使用本地中文 Embedding 模型 + Chroma 本地持久化向量库
- 使用 LangChain `create_agent` 和 `@tool` 构建检索型 Agent
- 多轮命令行对话；回答会标明使用的资料来源
- 兼容 OpenAI API 与 OpenAI 兼容接口（通过 `OPENAI_BASE_URL` 配置）

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填写 `OPENAI_API_KEY`。如果使用 DeepSeek 等仅提供聊天模型的兼容服务，保持 `EMBEDDING_PROVIDER=local`；首次入库会自动下载本地中文向量模型。只有服务明确支持 `/embeddings` 接口时，才配置 `EMBEDDING_PROVIDER=openai` 及对应的 `OPENAI_EMBEDDING_MODEL`。

把资料放进 `data/`，然后执行：

```powershell
python ingest.py
python chat.py
```

## Web UI

安装依赖后启动：

```powershell
streamlit run app.py
```

浏览器会自动打开。页面支持多轮问答、流式显示回答、展示回答来源，并可在侧栏上传 `.md`、`.txt`、`.pdf` 后重建索引。

## 工作方式

```text
用户问题 → LangChain Agent → 是否需要检索？ → search_knowledge_base 工具
                    ↑                                  ↓
                    └──── 需要更多证据时再次检索 ← Chroma 向量库
                                      ↓
                               带来源的最终回答
```

每次执行 `python ingest.py` 会重建 `chroma_db/`，使索引与 `data/` 当前内容一致。
