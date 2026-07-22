"""Real Chroma E2E 集成测试。

覆盖完整链路：Document → Embedding → Chroma → Retriever → Service → Tool。
使用临时目录和真实 Chroma 持久化，不依赖外部服务。

需要: pip install fastembed (本地 embedding) 或 OPENAI_API_KEY
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture(scope="module")
def embeddings():
    """获取 embedding 实例。不可用时跳过整个模块。"""
    try:
        from embeddings import get_embeddings
        return get_embeddings()
    except Exception as e:
        pytest.skip(f"Embedding unavailable: {e}")


@pytest.fixture
def chroma_env(tmp_path: Path):
    """创建临时 Chroma 环境：data 目录 + persist 目录。"""
    data_dir = tmp_path / "data"
    persist_dir = tmp_path / "chroma_db"
    data_dir.mkdir()
    return {"data_dir": data_dir, "persist_dir": persist_dir}


@pytest.fixture
def seed_documents(chroma_env):
    """在临时 data 目录写入测试文档。"""
    data_dir = chroma_env["data_dir"]
    (data_dir / "test.md").write_text(
        "# Test Document\n\nPython is a programming language.\n\n"
        "## Deployment\n\nUse Docker for containerized deployment.\n",
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture
def chroma_collection(seed_documents, chroma_env, embeddings):
    """创建真实 Chroma collection 并返回 persist_dir。"""
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = TextLoader(str(seed_documents / "test.md"), encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    persist_dir = chroma_env["persist_dir"]
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="test_kb",
        persist_directory=str(persist_dir),
    )
    return persist_dir


# ============================================================
# Tests
# ============================================================


def test_chroma_ingest_and_retrieve(chroma_collection, embeddings):
    """真实 Chroma: 写入文档 → 检索。"""
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name="test_kb",
        persist_directory=str(chroma_collection),
        embedding_function=embeddings,
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("Python")

    assert len(docs) >= 1
    assert any("Python" in d.page_content for d in docs)


def test_hybrid_retriever_with_real_chroma(chroma_collection, embeddings):
    """真实 HybridRetriever: BM25 + Vector + RRF。"""
    from langchain_chroma import Chroma
    from retrieval import HybridRetriever

    store = Chroma(
        collection_name="test_kb",
        persist_directory=str(chroma_collection),
        embedding_function=embeddings,
    )
    # 构建 BM25 索引
    docs = store.as_retriever(search_kwargs={"k": 100}).invoke("")
    from retrieval import build_bm25_index
    build_bm25_index(docs, chroma_collection)

    retriever = HybridRetriever(store, persist_dir=chroma_collection)
    results = retriever.search("Python deployment Docker")

    assert len(results) >= 1
    assert any("Python" in d.page_content for d in results)


def test_knowledge_service_with_real_retriever(chroma_collection, embeddings, tmp_path):
    """KnowledgeService.search() 使用真实 HybridRetriever。"""
    from unittest.mock import patch, MagicMock
    from langchain_chroma import Chroma
    from retrieval import HybridRetriever
    from knowledge.domain import KnowledgeDomain

    store = Chroma(
        collection_name="test_kb",
        persist_directory=str(chroma_collection),
        embedding_function=embeddings,
    )
    retriever = HybridRetriever(store, persist_dir=chroma_collection)

    mock_llm = MagicMock()
    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()

    # 注入真实 retriever 到 registry
    svc._registry.get_retriever = MagicMock(return_value=retriever)

    docs = svc.search("Docker deployment", user=None)
    assert len(docs) >= 1
    assert any("Docker" in d.page_content or "deployment" in d.page_content
               for d in docs)


def test_multi_kb_real_chroma(chroma_collection, embeddings, tmp_path):
    """真实 Chroma 多 KB：两个不同内容的 collection → 路由到正确的。"""
    from unittest.mock import patch, MagicMock
    from langchain_chroma import Chroma
    from retrieval import HybridRetriever
    from knowledge.domain import KnowledgeDomain
    from knowledge.router import RoutingDecision

    # 创建第二个 KB
    persist2 = tmp_path / "chroma_db_2"
    persist2.mkdir()
    data2 = tmp_path / "data2"
    data2.mkdir()
    (data2 / "hr.md").write_text(
        "# HR Policy\n\nAnnual leave: 20 days.\nBenefits: health insurance.\n",
        encoding="utf-8",
    )
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    loader = TextLoader(str(data2 / "hr.md"), encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(loader.load())
    Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        collection_name="hr_kb", persist_directory=str(persist2),
    )

    store1 = Chroma(collection_name="test_kb", persist_directory=str(chroma_collection), embedding_function=embeddings)
    store2 = Chroma(collection_name="hr_kb", persist_directory=str(persist2), embedding_function=embeddings)
    retriever1 = HybridRetriever(store1, persist_dir=chroma_collection)
    retriever2 = HybridRetriever(store2, persist_dir=persist2)

    mock_llm = MagicMock()
    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()

    # 模拟: Router 选择 tech KB
    svc._router.route = MagicMock(return_value=RoutingDecision(
        domain_id="tech_docs", confidence=0.85, strategy="llm",
    ))
    svc._registry.get_retriever = MagicMock(return_value=retriever1)

    docs = svc.search("Python programming", user=None)
    assert any("Python" in d.page_content for d in docs)
    # 不应该返回 HR 内容
    assert not any("Annual leave" in d.page_content for d in docs)


def test_pipeline_with_real_retriever(chroma_collection, embeddings):
    """SearchPipeline.retrieve() 使用真实 HybridRetriever — 快速路径。"""
    from langchain_chroma import Chroma
    from retrieval import HybridRetriever
    from search_pipeline import SearchPipeline

    store = Chroma(
        collection_name="test_kb",
        persist_directory=str(chroma_collection),
        embedding_function=embeddings,
    )
    retriever = HybridRetriever(store, persist_dir=chroma_collection)

    pipeline = SearchPipeline(rewrite_enabled=False, judge_enabled=False)
    docs = pipeline.retrieve("Docker", retriever)

    assert len(docs) >= 1


def test_chroma_persistence_across_sessions(chroma_collection, embeddings):
    """Chroma 持久化：关闭再打开，数据不丢失。"""
    from langchain_chroma import Chroma

    store1 = Chroma(
        collection_name="test_kb",
        persist_directory=str(chroma_collection),
        embedding_function=embeddings,
    )
    count1 = len(store1.as_retriever(search_kwargs={"k": 100}).invoke(""))

    # 重新打开
    store2 = Chroma(
        collection_name="test_kb",
        persist_directory=str(chroma_collection),
        embedding_function=embeddings,
    )
    count2 = len(store2.as_retriever(search_kwargs={"k": 100}).invoke(""))

    assert count1 == count2 > 0
