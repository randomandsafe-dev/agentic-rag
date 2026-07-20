# 查询改写 + 相关性判断 + 重试机制 —— 实现方案文档

## 一、项目现状分析

### 1.1 当前架构

```
chat.py                     # CLI 入口，单轮对话循环
  └── rag_agent.py          # 核心：LangChain Agent + @tool 检索工具
        ├── build_agent()   # 创建 Agent，注入 search_knowledge_base 工具
        ├── search_knowledge_base(query)  # 检索工具：调 retriever → 格式化结果
        └── get_retriever() # Chroma 向量库 retriever
config.py                   # 环境变量配置（Settings dataclass）
ingest.py                   # 文档入库脚本
```

### 1.2 当前检索流程

```
用户问题 → Agent 决定调用 search_knowledge_base(query)
         → Chroma 向量检索 (top_k=4)
         → 格式化返回给 Agent
         → Agent 生成回答
```

### 1.3 关键约束

- **合作项目**：其他人在同一代码库上工作，修改现有文件必须最小化
- **技术栈**：LangChain 1.0+, Chroma, OpenAI 兼容 API, fastembed 本地嵌入

---

## 二、目标功能

| 功能                   | 说明                                                                     |
| ---------------------- | ------------------------------------------------------------------------ |
| **查询改写**     | 检索前用 LLM 将用户原始问题改写为更适合向量检索的关键词/短句             |
| **相关性判断**   | 检索后用 LLM 判断每个文档与原始问题的相关程度                            |
| **最多两次重试** | 若检索结果不相关（所有文档得分均低于阈值），改写 query 后重试，最多 2 次 |

### 2.1 完整流程

```
用户问题
  │
  ▼
┌─────────────────────────────────────────────────┐
│           RetrievalPipeline (新模块)              │
│                                                  │
│  原始 query                                       │
│    │                                             │
│    ▼                                             │
│  [QueryRewriter] ── LLM 改写为检索友好形式        │
│    │                                             │
│    ▼                                             │
│  [Chroma Retriever] ── 向量检索 top_k 个文档      │
│    │                                             │
│    ▼                                             │
│  [RelevanceJudge] ── LLM 逐一判断文档相关性        │
│    │                                             │
│    ├── 有相关文档(score≥阈值) → 返回相关文档        │
│    │                                             │
│    └── 无相关文档 ∧ 重试次数<2                     │
│         │                                        │
│         └── 换个角度改写 query → 回到 [Chroma]     │
│                                                  │
│  超过重试上限 → 返回原始检索结果(降级)              │
└─────────────────────────────────────────────────┘
  │
  ▼
Agent 基于最终结果生成回答
```

---

## 三、模块设计

### 3.1 新增文件

```
agenticRAG/
├── retrieval_pipeline.py    ← 新增：查询改写 + 相关性判断 + 重试
└── prompts.py               ← 新增：提示词模板（与逻辑分离）
```

### 3.2 修改文件（最小化改动）

```
rag_agent.py    ← 仅修改 search_knowledge_base 工具内部，改用 pipeline
config.py       ← 新增 4~5 个配置项
.env.example    ← 新增配置项说明
```

---

## 四、详细设计

### 4.1 `prompts.py` —— 提示词模板

```python
# 查询改写提示词
REWRITE_SYSTEM_PROMPT = """你是一个搜索查询优化专家。
将用户的问题改写为适合向量检索的简短查询（1-2句话）。
- 提取核心关键词和概念
- 去除寒暄和冗余表达
- 保留专业术语
- 可以尝试不同角度表述同一问题

只输出改写后的查询文本，不要加任何解释或前缀。"""

REWRITE_RETRY_PROMPT = """你是一个搜索查询优化专家。
之前的检索结果不够相关。请从**不同角度**重新表述用户的原始问题，尝试不同的关键词或表述方式。

原始问题：{original_query}
上一次改写：{last_rewrite}

只输出新的改写查询文本，不要加任何解释或前缀。"""

# 相关性判断提示词
RELEVANCE_JUDGE_PROMPT = """你是一个文档相关性判断专家。
给定用户问题和检索到的文档，判断该文档是否与问题相关。

评分标准：
- 3: 高度相关 — 文档直接回答了问题或包含了关键信息
- 2: 部分相关 — 文档涉及问题相关领域但不直接回答问题
- 1: 弱相关 — 仅有个别词汇匹配，实质内容不相关
- 0: 无关 — 完全不相关

用户问题：{query}

文档内容：
{document}

请只输出一个数字（0-3）表示相关性分数，不要输出任何其他内容。"""
```

### 4.2 `retrieval_pipeline.py` —— 核心逻辑

采用**策略模式**设计，相关性判断策略可插拔替换：

```python
# ============================================================
# 抽象接口 —— 后续换策略只需实现这个接口
# ============================================================
class RelevanceStrategy(ABC):
    """相关性判断的抽象接口。实现新策略只需继承此类。"""
    @abstractmethod
    def judge(self, query: str, documents: list[Document]) -> list[tuple[Document, int]]:
        """返回 [(doc, score), ...]，score 范围 0-3"""
        ...

    @abstractmethod
    def has_relevant(self, scored: list[tuple[Document, int]], threshold: int) -> bool:
        """判断是否存在相关文档"""
        ...


# ============================================================
# 内置策略实现
# ============================================================
class LLMRelevanceJudge(RelevanceStrategy):
    """策略A（默认）：用 LLM 判断语义相关性 —— 最准确"""
    def judge(query, docs) -> list[tuple[Document, int]]: ...
    def has_relevant(scored, threshold) -> bool:
        return any(score >= threshold for _, score in scored)


class VectorScoreJudge(RelevanceStrategy):
    """策略B：直接用 Chroma 的向量相似度分数 —— 零成本，适合文档类型单一的场景"""
    def judge(query, docs) -> list[tuple[Document, int]]:
        # 用 Chroma.similarity_search_with_score 的 distance 归一化到 0-3
        ...
    def has_relevant(scored, threshold) -> bool:
        return any(score >= threshold for _, score in scored)


class HybridJudge(RelevanceStrategy):
    """策略C：向量分数初筛 + LLM 终判 —— 平衡成本与准确性"""
    def __init__(self, vector_threshold: float = 0.5, llm_judge=None): ...
    def judge(query, docs) -> list[tuple[Document, int]]: ...
    def has_relevant(scored, threshold) -> bool: ...


# ============================================================
# 策略工厂 —— 根据配置自动选择
# ============================================================
def create_relevance_judge(settings) -> RelevanceStrategy:
    """一行配置切换策略，无需改其他代码"""
    strategy = settings.relevance_strategy  # "llm" | "vector" | "hybrid"
    if strategy == "vector":
        return VectorScoreJudge()
    elif strategy == "hybrid":
        return HybridJudge()
    return LLMRelevanceJudge()  # 默认


# ============================================================
# 改写器和编排器
# ============================================================
class QueryRewriter:
    """使用 LLM 改写用户查询，使其更适合向量检索。"""
    def rewrite(original_query: str) -> str
    def rewrite_retry(original_query: str, last_rewrite: str) -> str

class RetrievalPipeline:
    """
    编排完整的检索流程：
    1. 改写查询
    2. 执行检索
    3. 判断相关性（策略可插拔）
    4. 若不相关则换个角度重试（最多2次）
    """
    def __init__(self, retriever, rewriter, judge: RelevanceStrategy, config):
        self.judge = judge  # 注入策略

    def retrieve(query: str) -> list[Document]:
        for attempt in range(self.max_retries + 1):
            rewritten = self.rewriter.rewrite(query) if attempt == 0 \
                        else self.rewriter.rewrite_retry(query, last_rewrite)
            docs = self.retriever.invoke(rewritten)
            scored = self.judge.judge(query, docs)
            if self.judge.has_relevant(scored, self.threshold):
                return [d for d, s in scored if s >= self.threshold]
        return docs  # 降级：返回最后一次检索结果
```

**切换方式** —— 只需改 `.env` 一行：

```bash
# 策略A：LLM 判断（默认，最准）
RELEVANCE_STRATEGY=llm

# 策略B：向量分数（零成本，快速）
RELEVANCE_STRATEGY=vector

# 策略C：混合（先筛后判）
RELEVANCE_STRATEGY=hybrid
```

以后新增策略（比如用 Jina Reranker API、Cross-encoder 本地模型等），只需写一个新类实现 `RelevanceStrategy` 接口，工厂里加一行映射，pipeline 代码完全不动。

### 4.3 `rag_agent.py` 改动

只改 `search_knowledge_base` 函数内部（约10行改动）：

```python
# 原来的代码：
@tool
def search_knowledge_base(query: str) -> str:
    try:
        return format_documents(get_retriever().invoke(query))
    except ...

# 改为：
@tool
def search_knowledge_base(query: str) -> str:
    try:
        pipeline = get_retrieval_pipeline()  # 新增
        documents = pipeline.retrieve(query)  # 替代原来的 retriever.invoke
        return format_documents(documents)
    except ...
```

### 4.4 `config.py` 改动

在 `Settings` dataclass 中新增以下字段：

```python
# 检索增强配置
rewrite_enabled: bool = True              # 是否启用查询改写
relevance_judge_enabled: bool = True      # 是否启用相关性判断
relevance_strategy: str = "llm"           # 判断策略: "llm" | "vector" | "hybrid"
max_retries: int = 2                      # 最大重试次数
relevance_threshold: int = 2              # 相关性阈值（0-3，>=此值视为相关）
rewrite_model: str | None = None          # 改写/判断用的模型（None=用主模型）
```

对应的环境变量：

```bash
REWRITE_ENABLED=true
RELEVANCE_JUDGE_ENABLED=true
RELEVANCE_STRATEGY=llm           # llm(默认) | vector | hybrid
MAX_RETRIES=2
RELEVANCE_THRESHOLD=2
# REWRITE_MODEL=gpt-4o-mini     # 可选：改写/判断专用模型
```

---

## 五、文件改动清单

| 文件                      | 操作           | 改动量 | 说明                                |
| ------------------------- | -------------- | ------ | ----------------------------------- |
| `prompts.py`            | **新增** | ~40行  | 提示词模板                          |
| `retrieval_pipeline.py` | **新增** | ~120行 | 核心逻辑                            |
| `config.py`             | 修改           | +15行  | 新增配置字段                        |
| `rag_agent.py`          | 修改           | ~10行  | search_knowledge_base 改用 pipeline |
| `.env.example`          | 修改           | +6行   | 新配置项说明                        |

**关键点**：现有文件改动量极小，所有新逻辑都在独立的新文件中，方便多人协作时不冲突。

---

## 六、LLM 调用成本估算

每次检索的额外 LLM 调用：

| 阶段       | 调用次数 | 说明                                                    |
| ---------- | -------- | ------------------------------------------------------- |
| 查询改写   | 1次      | 每次检索前改写                                          |
| 相关性判断 | 1次      | 批量判断所有文档（在单个 prompt 中处理 top_k=4 个文档） |
| 重试       | 最多2次  | 每次重试 = 1次改写 + 1次检索 + 1次判断                  |

**最坏情况**（2次重试均触发）：3次改写 + 3次判断 = ~6次额外 LLM 调用
**最好情况**（首次即相关）：1次改写 + 1次判断 = 2次额外 LLM 调用

建议使用 `gpt-4o-mini` 等轻量模型做改写和判断，成本极低。

---

## 七、配置项完整参考

```bash
# .env 新增配置项
REWRITE_ENABLED=true           # 是否启用查询改写（默认 true）
RELEVANCE_JUDGE_ENABLED=true   # 是否启用相关性判断（默认 true）
MAX_RETRIES=2                  # 最大重试次数（默认 2）
RELEVANCE_THRESHOLD=2          # 相关性阈值 0-3（默认 2）
# REWRITE_MODEL=gpt-4o-mini   # 可选：改写/判断专用模型
```

可通过设置 `REWRITE_ENABLED=false` 和 `RELEVANCE_JUDGE_ENABLED=false` 完全回退到原始行为。

---

## 八、测试建议

1. **单元测试**：`QueryRewriter.rewrite()` 输入/输出格式
2. **集成测试**：准备一组已知问题和对应的知识库文档，验证检索质量提升
3. **A/B 对比**：同一问题在启用/禁用 pipeline 下的回答质量对比
4. **边界测试**：
   - 知识库完全无相关内容时的降级行为
   - 知识库为空时的错误处理
   - API 调用失败时的降级策略

---

## 九、实施步骤

1. **创建 `prompts.py`** — 提示词模板，与逻辑分离
2. **创建 `retrieval_pipeline.py`** — QueryRewriter、RelevanceJudge、RetrievalPipeline
3. **修改 `config.py`** — 新增配置字段
4. **修改 `rag_agent.py`** — search_knowledge_base 改用 pipeline
5. **更新 `.env.example`** — 说明新配置
6. **测试验证** — 端到端验证检索质量

预计总代码量：~180行新代码 + ~25行对现有文件的修改。
