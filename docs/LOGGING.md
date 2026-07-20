# 日志记录说明

## 概述

`retrieval_pipeline.py` 使用 Python 标准库 `logging` 模块记录检索增强管道的运行状态。

## 日志级别与含义

| 级别 | 含义 | 触发场景 |
|------|------|----------|
| `INFO` | 正常流程记录 | 每次检索尝试的改写结果、相关性分数、成功/失败的文档数 |
| `WARNING` | 降级行为 | LLM 调用失败降级、JSON 解析失败降级、重试耗尽降级 |

**不记录** `DEBUG` 日志，也没有 `ERROR` 级别（所有异常都有降级策略，不会抛出）。

## 日志示例

正常流程：
```
INFO:__main__:检索第 1 次，改写查询: Agentic RAG 定义 原理
INFO:__main__:第 1 次相关分数: [3, 0, 2]
INFO:__main__:第 1 次检索成功，2/3 个文档相关
```

重试流程：
```
INFO:__main__:检索第 1 次，改写查询: Chroma 向量库 持久化 存储
INFO:__main__:第 1 次相关分数: [1, 0, 0]
INFO:__main__:检索第 2 次，改写查询: ChromaDB 数据存储 本地文件
INFO:__main__:第 2 次相关分数: [3, 1]
INFO:__main__:第 2 次检索成功，1/2 个文档相关
```

降级场景：
```
WARNING:__main__:无法解析相关性判断结果，降级为全部弱相关(1分): 非常抱歉，我无法...
WARNING:__main__:已达最大重试次数(2)，降级返回最后3个文档
```

## 如何开启日志

默认只输出 `WARNING` 及以上。在入口文件（如 `chat.py`）顶部加一行即可看到全部日志：

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
```

或只关注检索管道的日志：

```python
logging.getLogger("retrieval_pipeline").setLevel(logging.INFO)
```

## 日志点一览

`retrieval_pipeline.py` 共 5 个日志埋点：

| 位置 | 级别 | 内容 |
|------|------|------|
| `LLMRelevanceJudge.judge()` | WARNING | LLM 调用失败 → 降级为全 2 分 |
| `LLMRelevanceJudge` 外围 | WARNING | JSON 解析失败 → 降级为全 1 分 |
| `VectorScoreJudge.judge()` | WARNING | 向量分数检索失败 → 降级为全 2 分 |
| `RetrievalPipeline.retrieve()` | INFO | 每次检索尝试的改写查询 |
| `RetrievalPipeline.retrieve()` | INFO | 每次检索尝试的相关性分数 |
| `RetrievalPipeline.retrieve()` | INFO | 检索成功的文档数 |
| `RetrievalPipeline.retrieve()` | WARNING | 重试耗尽后降级 |
| `QueryRewriter.rewrite()` | WARNING | 改写失败 → 使用原始查询 |
| `QueryRewriter.rewrite_retry()` | WARNING | 重试改写失败 → 使用原始查询 |

## 不做的事

- 默认不写日志文件（只在控制台输出）
- 不记录 LLM 的完整 prompt/response（隐私和成本考虑）
- 不记录 embedding 调用

---

需要写日志文件时，只需在入口加一行即可，pipeline 代码不用改：

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    filename="logs/rag.log",
)
```

`logs/` 目录已在 `.gitignore` 中排除，日志文件不会被提交。
