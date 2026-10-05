"""Metadata-preserving chunks, local embeddings and hybrid retrieval."""
import hashlib
import json
import os
import re
from typing import List, Sequence, Union

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

load_dotenv()


def tokenize(text: str) -> List[str]:
    """Keep identifiers such as MARIGOLD-42 intact for keyword search."""
    return re.findall(r"\w+(?:[-.]\w+)*", text.casefold())


def get_chunks(documents: Union[str, Sequence[Document]]) -> List[Document]:
    if isinstance(documents, str):
        documents = [Document(page_content=documents, metadata={"source": "Uploaded text", "location": "text"})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(list(documents))
    for chunk in chunks:
        chunk.metadata.setdefault("source", "Uploaded text")
        chunk.metadata.setdefault("location", "text")
        identity = json.dumps([chunk.metadata, chunk.page_content], sort_keys=True)
        chunk.metadata["chunk_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def get_vectorstore(chunks: Sequence[Document]):
    try:
        embeddings = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        return FAISS.from_documents(list(chunks), embeddings)
    except Exception as error:
        st.error(f"Cannot build the document index. Check Ollama and its embedding model: {error}")
        return None


class HybridRetriever(BaseRetriever):
    """Equal-weight reciprocal rank fusion of dense and BM25 results."""
    semantic: BaseRetriever
    keyword: BM25Retriever
    top_k: int = 4

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        dense = self.semantic.invoke(query, config={"callbacks": run_manager.get_child()})
        query_terms = set(tokenize(query))
        # BM25 returns even zero-match documents; only fuse actual term matches.
        lexical = [doc for doc in self.keyword.invoke(query)
                   if query_terms.intersection(tokenize(doc.page_content))]
        scores, candidates = {}, {}
        for results in (dense, lexical):
            seen = set()
            for rank, document in enumerate(results, 1):
                key = document.metadata.get("chunk_id") or hashlib.sha256(document.page_content.encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                candidates[key] = document
                scores[key] = scores.get(key, 0.0) + 0.5 / (60 + rank)
        return [candidates[key] for key in sorted(scores, key=scores.get, reverse=True)[:self.top_k]]


def get_hybrid_retriever(vectorstore, chunks: Sequence[Document]) -> HybridRetriever:
    keyword = BM25Retriever.from_documents(list(chunks), preprocess_func=tokenize, k=8)
    return HybridRetriever(
        semantic=vectorstore.as_retriever(search_kwargs={"k": 8}), keyword=keyword,
    )
