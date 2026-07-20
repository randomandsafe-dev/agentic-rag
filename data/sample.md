# 示例知识库

Agentic RAG 将语言模型、检索工具和推理循环结合起来。模型可以根据问题决定是否检索、用什么关键词检索，以及是否再次检索补足证据。

本项目使用 Chroma 持久化向量索引。向 `data/` 放入 Markdown、纯文本或 PDF 文件后，运行 `python ingest.py` 重建索引。
