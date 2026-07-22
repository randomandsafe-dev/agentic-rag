"""KnowledgeDomain 数据类 —— 描述单个知识库的元数据与存储路径。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class KnowledgeDomain:
    """单个知识库的完整定义。

    每个 domain 对应一个独立的数据目录 + Chroma 持久化目录。
    """

    id: str
    name: str
    description: str
    data_dir: Path
    persist_dir: Path
    collection_name: str
    default: bool = False
    keywords: list[str] = field(default_factory=list)
