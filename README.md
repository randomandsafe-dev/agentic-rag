# LangChain Agentic RAG

一个可直接运行的本地知识库问答项目。它不是固定的"检索后回答"链：LangChain Agent 会根据问题自主决定是否调用检索工具，也可以通过不同关键词重复检索，再基于检索结果给出带来源的回答。

## 功能

- 支持 `data/` 下的 `.md`、`.txt`、`.pdf` 文档入库
- 默认使用本地中文 Embedding 模型 + Chroma 本地持久化向量库
- 使用 LangChain `create_agent` 和 `@tool` 构建检索型 Agent
- 多轮命令行对话；回答会标明使用的资料来源
- 兼容 OpenAI API 与 OpenAI 兼容接口（通过 `OPENAI_BASE_URL` 配置）
- 多会话管理：新建、切换、删除会话，对话历史自动持久化
- 可配置的上下文窗口（`SESSION_WINDOW`），控制每次加载的对话轮数
- **多知识库路由**：根据查询自动选择最相关的知识库（`KeywordRouter` / `LLMRouter`）
- **搜索增强管道**：可选的查询改写 + 相关性判断 + 重试机制（`SearchPipeline`）
- **知识库权限控制**：基于角色的 KB 级访问隔离（`policy.yaml`）

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

```powershell
streamlit run app.py
```

侧栏支持会话管理、知识库上传重建索引、用户身份切换。

## 会话记忆

项目在 `conversations.db`（SQLite）中自动持久化所有对话记录，程序重启后对话历史不会丢失。

**CLI 会话命令：**

| 命令 | 说明 |
|------|------|
| `/new <名称>` | 新建并切换到指定会话 |
| `/list` | 列出所有会话 |
| `/switch <编号>` | 切换到指定会话 |
| `/delete <编号>` | 删除指定会话及其对话记录 |

**Web UI：** 侧栏「📁 会话」区域支持会话切换、新建、删除和清空对话。

**配置项：**

```bash
MEMORY_DB_PATH=conversations.db   # 会话数据库路径
SESSION_WINDOW=20                 # 上下文窗口大小
```

## 多知识库路由

通过 `knowledge_bases.yaml` 定义多个知识库：

```yaml
domains:
  - id: default
    name: "默认知识库"
    description: "通用知识库"
    data_dir: data
    persist_dir: chroma_db
    collection_name: knowledge_base
    default: true
    keywords: []

  - id: tech_docs
    name: "技术文档"
    description: "API、架构、部署相关"
    data_dir: data/tech
    persist_dir: chroma_db/tech
    collection_name: kb_tech
    keywords: [python, api, deploy]
```

路由策略通过 `ROUTER_STRATEGY` 环境变量切换：
- `llm`（默认）：LLM 根据 domain name/description 智能选择
- `keyword`：关键词匹配，零 LLM 调用

```bash
ROUTER_STRATEGY=keyword
```

## 搜索增强管道

通过 `.env` 控制：

```bash
REWRITE_ENABLED=true              # 查询改写
RELEVANCE_JUDGE_ENABLED=true      # 相关性判断
MAX_RETRIES=2                     # 最大重试次数
RELEVANCE_THRESHOLD=2             # 相关性阈值 (0-3)
```

关闭所有增强时，Pipeline 会自动走零开销快速路径（直接检索）。

## 知识库权限控制 (Phase 3)

通过 `config/policy.yaml` 实现 KB 级别的访问隔离。

### policy.yaml 示例

```yaml
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"              # 全部 KB

  developer:
    role: developer
    allowed_kbs:
      - tech_docs
      - api_docs

default:
  role: viewer
  allowed_kbs:
    - default            # 未在 users 中列出的用户默认仅访问 default
```

### 使用方式

**CLI：**

```powershell
python chat.py --user alice --role developer
```

**Web UI：** 侧栏「👤 用户身份」选择角色。

**代码：**

```python
from knowledge.access import UserContext
from knowledge.service import get_knowledge_service

user = UserContext(user_id="alice", role="developer")
docs = get_knowledge_service().search("Python 部署", user=user)
```

不传 `user` 参数时跳过权限过滤（向后兼容）。

## 工作方式

```text
用户问题 → LangChain Agent → search_knowledge_base 工具
                  ↓
         KnowledgeService.search(query, user)
                  ↓
         AccessGuard.filter_domains(user, domains)   ← 权限过滤
                  ↓
         KnowledgeRouter.route(query, domains)        ← KB 路由
                  ↓
         SearchPipeline.retrieve(query, retriever)    ← 搜索增强
                  ↓
         HybridRetriever (BM25 + Vector + RRF + Reranker)
                  ↓
         带来源的最终回答
```

## 项目架构

```
rag_agent.py           — Agent 工具定义 + 构建
knowledge/service.py   — 编排层 (AccessGuard → Router → Pipeline)
knowledge/router.py    — 查询 → KB 路由
knowledge/registry.py  — KB 注册 + Retriever 管理
knowledge/access.py    — 权限过滤 (UserContext / AccessPolicy / AccessGuard)
search_pipeline.py     — 搜索增强管道 (Rewrite → Retrieve → Judge → Retry)
retrieval.py           — BM25 + 向量 混合检索 + RRF + Reranker
llm_factory.py         — 统一 LLM 创建入口
embeddings.py          — Embedding 模型工厂
memory/                — 会话持久化 (SQLite)
```

## Production Configuration (Phase 6)

统一运行时配置入口为 `config/runtime.yaml`：

```yaml
verification:
  enabled: false       # 检索后验证
  max_retry: 2
  min_score: 0.5

self_correction:
  enabled: false       # 自纠正闭环
  max_iterations: 3

metrics:
  enabled: false       # 指标收集
  query_length_limit: 100
```

所有功能默认关闭，与 v0.5.0 行为完全一致。启用后自动接入相应增强能力。

## 配置参考

完整配置见 `.env.example`。

每次执行 `python ingest.py` 会重建对应 KB 的 Chroma 索引，使索引与 `data/` 当前内容一致。可通过 `--domain` 参数指定目标 KB：

```powershell
python ingest.py --domain tech_docs
```
