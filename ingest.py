"""将 data/ 中的本地文档切分、嵌入并写入 Chroma。

支持 --domain 参数指定知识库（对应 knowledge_bases.yaml 中的 domain id）。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import ROOT_DIR, settings
from embeddings import get_embeddings
from retrieval import build_bm25_index

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _get_domain_config(domain_id: str | None):
    """Parse domain config, return (data_dir, persist_dir, collection_name).

    If domain_id is None, fall back to settings (backward compatible).
    """
    if domain_id is None:
        return settings.data_dir, settings.persist_dir, settings.collection_name

    from knowledge.registry import KnowledgeBaseRegistry

    registry = KnowledgeBaseRegistry()
    domain = registry.get_domain(domain_id)
    return domain.data_dir, domain.persist_dir, domain.collection_name


def load_documents(data_dir: Path) -> list[Document]:
    """Read Markdown, text and PDF files from the knowledge base directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {data_dir}")

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


def ingest_documents(domain_id: str | None = None) -> tuple[int, int]:
    """Rebuild local index, return (document_count, chunk_count).

    Args:
        domain_id: Optional domain ID from knowledge_bases.yaml.
                   If None, uses settings globals (backward compatible).
    """
    settings.validate()
    data_dir, persist_dir, collection_name = _get_domain_config(domain_id)

    documents = load_documents(data_dir)
    if not documents:
        raise RuntimeError(
            f"No ingestible .md, .txt, or .pdf documents found in {data_dir}."
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(documents)

    # Rebuild index from scratch each time to avoid stale chunks.
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    if settings.hybrid_enabled:
        build_bm25_index(chunks, persist_dir)
    return len(documents), len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into a knowledge base")
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Domain ID (from knowledge_bases.yaml). Uses default config if omitted.",
    )
    args = parser.parse_args()

    document_count, chunk_count = ingest_documents(domain_id=args.domain)
    print(f"Ingest complete: {document_count} documents, {chunk_count} chunks.")


if __name__ == "__main__":
    main()
