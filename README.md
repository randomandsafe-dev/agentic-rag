# LangChain Agentic RAG

一个面向企业知识问答场景的 LangChain Agentic RAG 项目。它支持将 PDF、Markdown 与文本资料加载到本地知识库，结合混合检索、查询改写、重排序、答案验证、多知识库路由、权限过滤和可选联网搜索，生成带来源证据的回答。

项目提供命令行对话与 Streamlit Web UI，并将会话记录持久化到 SQLite。

## 技术栈

| 分类 | 技术 / 组件 | 用途 |
| --- | --- | --- |
| 编程语言 | Python | 项目后端、索引构建与应用逻辑。 |
| Agent 框架 | LangChain | 使用 `create_agent` 编排 Agent，并通过 `@tool` 暴露检索工具。 |
| 大语言模型 | OpenAI API 兼容接口 / `langchain-openai` | 生成回答、查询改写、知识库路由和答案验证。 |
| 文档解析 | PyPDFLoader、TextLoader | 加载 PDF、Markdown 与纯文本资料。 |
| 文本处理 | LangChain Text Splitters | 通过递归字符分块将长文档转换为可检索文本块。 |
| Embedding | FastEmbed 或 OpenAI Embeddings | 将文本块和用户问题转换为向量。 |
| 向量数据库 | Chroma / `langchain-chroma` | 本地持久化向量索引与语义检索。 |
| 关键词检索 | BM25 / `rank-bm25` | 精确术语和关键词召回，与向量结果融合。 |
| 重排序 | Sentence Transformers Cross-Encoder | 对混合检索候选和网页候选进行语义重排序。 |
| Web 应用 | Streamlit | 提供上传资料、会话管理、路由选择和问答界面。 |
| 会话存储 | SQLite | 持久化多会话、聊天记录和相关记忆数据。 |
| 联网搜索 | Tavily API | 获取外部网页证据，供独立联网问答链路使用。 |

## 功能概览

| 能力 | 说明 |
| --- | --- |
| PDF / 文本加载 | 支持 `.pdf`、`.md`、`.txt` 文件；PDF 使用 `PyPDFLoader` 解析。 |
| 文本分块 | 使用 `RecursiveCharacterTextSplitter`，默认块大小 800、重叠 120。 |
| Embedding 与向量库 | 支持本地 FastEmbed 或 OpenAI 兼容 Embedding；使用 Chroma 本地持久化。 |
| 检索工具 | 基于 LangChain `@tool` 提供 `search_knowledge_base`，由 `create_agent` 调用。 |
| 引用来源 | 回答和 UI 会显示参与回答的本地文档或网页来源，PDF 保留页码信息。 |
| 查询改写 | 可选地将用户口语化、上下文依赖的问题改写为更适合检索的查询。 |
| 相关性判断与重试 | 检索结果不相关时可判断并改写查询重试，默认最多两次。 |
| 混合检索 | 融合 BM25 关键词检索和 Chroma 向量检索，使用 RRF 合并候选结果。 |
| Reranker | 使用 Cross-Encoder 对候选片段及网页结果重排序；依赖不可用时自动降级。 |
| 答案验证 | 本地回答与联网回答均只针对各自证据单独验证，显示可信度、风险和缺失提示。 |
| 会话记忆 | SQLite 持久化多会话与聊天记录，支持新建、切换、删除和清空会话。 |
| 多知识库路由 | 根据问题选择最相关的知识库，支持 LLM 路由或关键词路由。 |
| 权限过滤 | 按用户/角色在检索前过滤可访问知识库，避免回答后再“遮挡”敏感内容。 |
| 联网搜索 | 可选 Tavily 搜索；本地结果和网页结果分别回答、分别展示和分别验证。 |

## 整体架构

```text
                         ┌──────────── 文档接入 ────────────┐
PDF / Markdown / Text ──>│ 加载 → 分块 → Embedding → Chroma │
                         └─────────────────────────────────┘

用户问题
  │
  ├─ 权限过滤：先确定可访问的知识库
  ├─ 知识库路由：LLM / 关键词选择目标知识库
  ├─ 查询改写：可选
  ├─ 混合检索：BM25 + 向量检索 + RRF
  ├─ Reranker：Cross-Encoder 重排序
  ├─ 相关性判断：不满足阈值时最多重试两次
  ├─ LangChain Agent：调用检索工具并生成带来源回答
  └─ 答案验证：核验回答是否被检索证据支持

可选联网路径：Tavily → 域名可信度 / 时效评分 → Reranker → 独立回答 → 独立验证
```

## 快速开始

### 1. 创建并进入虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### 2. 配置模型

编辑 `.env`，至少填写：

```dotenv
OPENAI_API_KEY=你的_API_Key
OPENAI_MODEL=gpt-4o-mini
```

项目兼容 OpenAI API 格式的模型服务。若使用兼容服务，可额外配置：

```dotenv
OPENAI_BASE_URL=https://你的兼容服务地址/v1
```

默认建议使用本地 Embedding：

```dotenv
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
```

只有模型服务明确提供 Embedding 接口时，才设置 `EMBEDDING_PROVIDER=openai` 并填写 `OPENAI_EMBEDDING_MODEL`。

### 3. 放入资料并构建索引

将 `.pdf`、`.md` 或 `.txt` 文件放入 `data/`：

```text
data/
├── 员工报销制度.pdf
├── 产品说明.md
└── 常见问题.txt
```

执行入库：

```powershell
python ingest.py
```

该命令会重新构建当前知识库的 Chroma 索引，并同步构建 BM25 索引（启用混合检索时）。

### 4. 运行

命令行模式：

```powershell
python chat.py
```

Web UI：

```powershell
streamlit run app.py
```

Web UI 支持上传资料、重建索引、会话管理、用户身份选择及本地/联网路由选择。

> Windows 提示 `WinError 32`（Chroma 文件正在被使用）时，先停止所有 `streamlit run app.py` 和 `python ingest.py` 进程后重启应用。应用在重建前会尝试释放缓存的检索对象。

## 核心能力说明

### 1. 文档加载、分块与引用

`ingest.py` 会递归扫描知识库目录中的 `.pdf`、`.md`、`.txt` 文件：

- PDF：使用 `PyPDFLoader`，保留页码元数据；
- 文本：使用 `TextLoader`，支持 UTF-8 与自动编码识别；
- 分块：`RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)`；
- 元数据：每个文本块记录相对文件路径，因此最终回答可展示来源；
- 存储：文本块 Embedding 后持久化到 `chroma_db/`（或知识库各自的目录）。

### 2. LangChain Agent 与检索工具

`rag_agent.py` 使用 LangChain 的 `create_agent` 构建 Agent，并通过 `@tool` 注册检索能力：

```text
用户问题
  → create_agent
  → search_knowledge_base(query)
  → KnowledgeService.search(query, user)
  → 返回格式化的文档片段与来源
  → Agent 基于证据生成回答
```

Agent 的系统提示要求优先对内部资料调用本地检索工具。联网工具仅在已配置 `TAVILY_API_KEY` 时可用。

### 3. BM25 + 向量混合检索 + Reranker

`HybridRetriever` 组合两种互补检索方式：

- **BM25**：适合制度条款、产品编号、专有名词和精确关键词；
- **向量检索**：适合同义表达、自然语言问法和语义相近内容；
- **RRF 融合**：合并两路候选，降低只依赖单一算法的偏差；
- **Cross-Encoder Reranker**：对候选片段以“问题—文档”相关性重新排序，输出 Top-K。

默认重排序模型为 `BAAI/bge-reranker-v2-m3`。若未安装 `sentence-transformers`、模型下载失败或关闭开关，项目会保留混合检索并降级为无重排序模式。

### 4. 查询改写、相关性判断与最多两次重试

`SearchPipeline` 提供可配置的检索增强闭环：

```text
原始问题
  → 查询改写（可选）
  → 检索与重排序
  → 相关性判断
  → 不相关：改写后重试
  → 最多 MAX_RETRIES 次（默认 2）
```

它适合处理“它的条件是什么”这类依赖上下文的问题，以及第一次查询未召回合适证据的场景。到达最大重试次数仍缺少可靠资料时，系统应明确说明资料不足，而非编造答案。

### 5. 多知识库路由与权限过滤

在 `config/knowledge_bases.yaml`（兼容根目录 `knowledge_bases.yaml`）中可以定义多个知识库：

```yaml
knowledge_bases:
  - id: default
    name: 默认知识库
    description: 通用企业资料
    data_dir: data
    persist_dir: chroma_db
    collection_name: knowledge_base
    default: true
    keywords: []

  - id: tech_docs
    name: 技术文档
    description: API、架构和部署资料
    data_dir: data/tech
    persist_dir: chroma_db/tech
    collection_name: kb_tech
    keywords: [python, api, deploy]
```

路由策略：

- `ROUTER_STRATEGY=llm`：由 LLM 根据知识库名称、描述和问题选择；
- `ROUTER_STRATEGY=keyword`：按关键词匹配，速度快且不额外消耗模型调用。

权限过滤发生在路由和检索之前。`config/policy.yaml` 可配置用户或角色允许访问的 KB：

```yaml
users:
  alice:
    role: developer
    allowed_kbs: [default, tech_docs]

default:
  role: viewer
  allowed_kbs: [default]
```

这意味着不可访问的知识库不会进入候选范围，更不会被模型用作回答上下文。

### 6. 联网搜索：与本地结果严格分开

配置 Tavily Key 后启用：

```dotenv
TAVILY_API_KEY=tvly-你的密钥
WEB_SEARCH_MAX_RESULTS=5
WEB_ROUTE_MODE=auto
```

联网搜索流程：

```text
Tavily 搜索
  → 搜索提供方相关度 + 域名可信度 + 时间新鲜度评分
  → Cross-Encoder 重排序
  → Top-K 网页证据
  → 仅基于网页证据生成“联网回答”
  → 联网答案验证
```

Web UI 可选以下路由：

- `auto`：默认优先本地；遇到“今天、最新、实时、新闻、价格”等时效问题时使用联网搜索；
- `local_only`：仅本地知识库；
- `web_only`：仅联网搜索；
- `both`：本地和联网并行查询。

对密码、Token、身份证等敏感词请求，会强制停留在本地路径。选择 `both` 时，本地证据和网页证据不会混合传给同一次回答生成；系统会分别给出“本地知识库回答”和“联网搜索回答”，各自附原始来源与验证报告。

### 7. 答案验证与引用一致性

每条本地或联网回答生成后，`verify_answer()` 会使用该回答对应的检索证据进行验证，重点检查：

- 回答是否有证据支持；
- 是否可能包含未被证据覆盖的结论；
- 是否存在信息遗漏或低可信度风险。

验证是“回答与当前检索证据的一致性检查”，并不能保证网页本身绝对真实。因此，涉及高风险的财务、医疗、法律或生产决策时，仍应复核原始文档和网页来源。

### 8. 会话记忆

会话数据默认保存至 `conversations.db`。Web UI 支持：

- 新建、切换、删除单个会话；
- 清空当前会话；
- 删除全部会话；
- 设置会话上下文窗口，避免无限累积历史消息。

相关配置：

```dotenv
MEMORY_DB_PATH=conversations.db
SESSION_WINDOW=20
```

## 常用配置

以下配置写入 `.env`，完整清单请参考 `.env.example`。

```dotenv
# 模型
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini

# Embedding / 向量库
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
PERSIST_DIR=chroma_db
COLLECTION_NAME=knowledge_base

# 检索
TOP_K=4
HYBRID_ENABLED=true
RERANKER_ENABLED=true
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANK_TOP_K=10

# 检索增强
REWRITE_ENABLED=true
RELEVANCE_JUDGE_ENABLED=true
RELEVANCE_THRESHOLD=2
MAX_RETRIES=2

# 答案验证
VERIFY_ENABLED=true

# 多知识库路由
ROUTER_STRATEGY=llm
KB_CONFIG_PATH=config/knowledge_bases.yaml

# 联网搜索（可选）
TAVILY_API_KEY=
WEB_SEARCH_MAX_RESULTS=5
WEB_ROUTE_MODE=auto
```

## 项目结构

```text
agentic-rag/
├── app.py                    # Streamlit Web UI
├── chat.py                   # 命令行对话入口
├── ingest.py                 # 文档加载、分块、Embedding 与索引构建
├── rag_agent.py              # LangChain create_agent 与检索工具
├── retrieval.py              # BM25 + 向量 + RRF + Reranker
├── search_pipeline.py        # 查询改写、相关性判断、重试
├── search_router.py          # 本地 / 联网搜索路由
├── web_search.py             # Tavily、网页评分与网页 Reranker
├── verify.py                 # 回答与证据一致性验证
├── embeddings.py             # Embedding 工厂
├── llm_factory.py            # LLM 工厂
├── knowledge/
│   ├── access.py             # 用户上下文与访问控制
│   ├── router.py             # 多知识库路由
│   ├── registry.py           # 知识库注册与 Retriever 管理
│   └── service.py            # 权限 → 路由 → 搜索编排
├── memory/                   # SQLite 会话记忆
├── config/
│   ├── knowledge_bases.yaml  # 多知识库定义
│   └── policy.yaml           # 用户/角色权限策略
├── data/                     # 默认知识库原始资料
├── chroma_db/                # 本地 Chroma 索引（不建议提交）
├── requirements.txt
└── .env.example
```

## 常见问题

### 为什么不上传 `chroma_db`？

`chroma_db` 是从原始文档生成的本地索引，通常体积大、与 Embedding 模型和 Chroma 版本绑定，并且可随时由 `data/` 重建。建议提交文档或其受控存储地址、配置和代码，而不提交索引目录。

### 如何避免重复入库？

当前 `ingest.py` 采用“重建索引”策略：每次入库前替换目标知识库索引，因此不会累积相同文本块。它的优点是索引与当前文件内容一致；对于超大规模知识库，后续可升级为基于文件哈希和文档 ID 的增量入库。

### Reranker 加载失败怎么办？

安装依赖：

```powershell
pip install sentence-transformers torch torchvision
```

未安装或模型不可用时，系统会自动回退到无重排序模式，基础向量/混合检索仍可工作。

### 如何关闭联网搜索？

不配置 `TAVILY_API_KEY`，或者在页面选择“仅本地知识库”。未配置 Key 时系统不会发起联网请求。

## 安全建议

- 不要将 `.env`、API Key、会话数据库或包含敏感资料的 `data/` 直接提交到公开仓库。
- 权限过滤应在服务器端绑定真实身份认证信息；Web UI 中的身份选择仅适合作为演示或内部受控环境。
- 联网内容属于外部不可信输入，不能作为内部制度或敏感决策的唯一依据。
- 建议定期审查来源引用、验证报告、访问日志和知识库权限配置。

## 后续可扩展方向

- SQL Agent：在只读权限、表/列白名单、SQL 解析校验、超时与审计前提下查询结构化业务数据；
- 增量入库：基于文件哈希、文档版本和删除同步减少重建耗时；
- 企业认证：接入 SSO、LDAP 或 OAuth，将真实用户身份传递给权限层；
- 可观测性与评测：增加检索命中率、引用正确率、延迟、成本与人工反馈闭环。
