"""BM25 + 向量混合检索与 Cross-Encoder Reranker。"""

from __future__ import annotations

import pickle
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings


def tokenize(text: str) -> list[str]:
    """统一的中英文分词函数：优先 jieba，回退为字符级/空格分词。"""
    try:
        import jieba

        return list(jieba.cut(text))
    except ImportError:
        import re

        tokens: list[str] = []
        for part in re.split(r"(\s+)", text):
            if not part.strip():
                tokens.append(part)  # 保留空白用于 BM25 分词对齐
                continue
            if re.search(r"[一-鿿]", part):
                # 中文字符级切分
                tokens.extend(list(part))
            else:
                tokens.extend(part.split())
        return tokens


class HybridRetriever:
    """BM25 + 向量混合检索器，支持 RRF 融合和 Cross-Encoder Reranker。

    检索流程：
    1. 向量检索（语义匹配）  →  BM25 检索（关键词匹配）
    2. Reciprocal Rank Fusion 融合两组排序
    3. Cross-Encoder Reranker 精细重排序
    4. 返回 top_k 条最相关文档
    """

    def __init__(self, vector_store: Chroma) -> None:
        self.vector_store = vector_store
        # 如果需要 rerank，先多取一些候选；否则直接取 top_k
        fetch_k = settings.rerank_top_k if settings.reranker_enabled else settings.top_k
        self.vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": fetch_k}
        )
        self.bm25 = None
        self.corpus_docs: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self.reranker = None

        if settings.hybrid_enabled:
            self._load_bm25()
        if settings.reranker_enabled:
            self._load_reranker()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load_bm25(self) -> None:
        """从磁盘加载 BM25 索引。"""
        bm25_path = settings.persist_dir / "bm25_index.pkl"
        if not bm25_path.exists():
            return
        with open(bm25_path, "rb") as fh:
            data = pickle.load(fh)
        self.bm25 = data["bm25"]
        self.corpus_docs = data["docs"]
        self.tokenized_corpus = data["tokenized_corpus"]

    def _load_reranker(self) -> None:
        """加载 Cross-Encoder 重排序模型（首次下载后缓存）。"""
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(
                settings.reranker_model,
                max_length=512,
            )
        except Exception as exc:
            print(f"⚠ 未能加载 Reranker 模型：{exc}。回退为无重排序模式。")
            self.reranker = None

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[Document]:
        """执行混合检索，返回 top_k 文档。"""
        # 1) 向量检索
        vector_docs: list[Document] = self.vector_retriever.invoke(query)

        # 2) BM25 检索（如果可用）
        bm25_docs: list[Document] = []
        if settings.hybrid_enabled and self.bm25 is not None:
            tokenized_query = tokenize(query)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            k = min(settings.rerank_top_k if settings.reranker_enabled else settings.top_k,
                    len(self.corpus_docs))
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:k]
            bm25_docs = [self.corpus_docs[i] for i in top_indices]

        # 3) RRF 融合
        fused = self._rrf_fusion(vector_docs, bm25_docs)

        # 4) Cross-Encoder 重排序
        if self.reranker and len(fused) > settings.top_k:
            fused = self._rerank(query, fused)

        return fused[: settings.top_k]

    # ------------------------------------------------------------------
    # RRF 融合
    # ------------------------------------------------------------------

    @staticmethod
    def _document_key(doc: Document) -> str:
        """为文档生成去重键（内容 + 来源）。"""
        source = doc.metadata.get("source", "")
        return f"{source}::{doc.page_content}"

    def _rrf_fusion(
        self,
        vector_docs: list[Document],
        bm25_docs: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """Reciprocal Rank Fusion：合并两组排序并返回融合后的有序文档列表。"""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(vector_docs):
            key = self._document_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            key = self._document_key(doc)
            if key not in doc_map:
                doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)

        sorted_keys = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[key] for key in sorted_keys]

    # ------------------------------------------------------------------
    # Cross-Encoder 重排序
    # ------------------------------------------------------------------

    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """使用 Cross-Encoder 对候选文档逐对打分并重新排序。"""
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]


# ------------------------------------------------------------------
# BM25 索引构建（供 ingest.py 调用）
# ------------------------------------------------------------------


def build_bm25_index(documents: list[Document], persist_dir: Path) -> None:
    """构建并持久化 BM25 索引，与 Chroma 向量库一起保存。"""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    bm25_path = persist_dir / "bm25_index.pkl"
    with open(bm25_path, "wb") as fh:
        pickle.dump(
            {
                "bm25": bm25,
                "docs": documents,
                "tokenized_corpus": tokenized_corpus,
            },
            fh,
        )

    print(f"BM25 索引已保存：{len(documents)} 个文档块 → {bm25_path}")
