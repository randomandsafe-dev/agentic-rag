"""将 data/ 中的本地文档切分、嵌入并写入 Chroma。"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ROOT_DIR, settings
from rag_agent import get_embeddings
from retrieval import build_bm25_index

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def load_documents(data_dir: Path) -> list[Document]:
    """读取知识库中的 Markdown、文本和 PDF，并保留相对来源路径。"""
    if not data_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在：{data_dir}")

    documents: list[Document] = []
    for path in data_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(
            str(path), encoding="utf-8", autodetect_encoding=True
        )
        for document in loader.load():
            document.metadata["source"] = str(path.resolve().relative_to(ROOT_DIR))
            documents.append(document)
    return documents


def ingest_documents() -> tuple[int, int]:
    """重建本地索引，并返回“文档数、文本块数”。"""
    settings.validate()
    documents = load_documents(settings.data_dir)
    if not documents:
        raise RuntimeError("data/ 中没有可入库的 .md、.txt 或 .pdf 文档。")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)

    # 每次入库重建索引，避免已删除或修改的文档残留在旧索引中。
    if settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_dir),
    )
    if settings.hybrid_enabled:
        build_bm25_index(chunks, settings.persist_dir)
    return len(documents), len(chunks)


def main() -> None:
    document_count, chunk_count = ingest_documents()
    print(f"入库完成：{document_count} 个文档，{chunk_count} 个文本块。")


if __name__ == "__main__":
    main()
