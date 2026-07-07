"""
Modality-aware retrieval for the cross-modal C1 pipeline (Week 9 rework).

Problem (WEEK9_DEFECTS D2/E6): the mm store holds text and caption chunks in
one collection, and a single top-k over the merged pool lets tighter text
chunks out-rank page captions, so the right page's caption almost never
reaches the answer context (measured: right-page caption in top-5 = 2/8 IR8).
C1 therefore degenerated to C0 by construction.

Fix — fixed per-modality budget plus page co-retrieval:

    1. text lane    — hybrid+rerank over text chunks only   -> top ``text_k``
    2. co-retrieval — captions of the pages the text lane found
    3. caption lane — hybrid+rerank over caption chunks only, fills whatever
                      caption slots co-retrieval left open

Total context = ``text_k + caption_k`` (default 3+2 = 5), the same budget as
the C0 baseline top-5, so C0 vs C1 compares *content*, not context size.

The caption lane alone (``retrieve_captions``) is the caption-only arm used
to separate reranker interference from caption content quality.
"""

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_chroma import Chroma

from .retrieval import (
    create_bm25_retriever,
    extract_documents_from_vectorstore,
    rerank_documents,
)
from langchain_classic.retrievers import EnsembleRetriever

TEXT_MODALITY = "text"
CAPTION_MODALITY = "image-derived"


@dataclass
class MMRetrieverConfig:
    text_k: int = 3        # text chunks in the final context
    caption_k: int = 2     # caption chunks in the final context
    first_stage_k: int = 20  # per-lane candidates before rerank (matches C0)


def _page_key(meta: dict) -> tuple[str, str, int]:
    return (meta.get("category", ""), meta.get("complexity", ""), meta.get("page"))


class ModalityAwareRetriever:
    """Two-lane retriever over the mm store with page co-retrieval."""

    def __init__(self, mm_store: Chroma, reranker, config: MMRetrieverConfig | None = None):
        self.config = config or MMRetrieverConfig()
        self.reranker = reranker

        docs = extract_documents_from_vectorstore(mm_store)
        text_docs = [d for d in docs if d.metadata.get("modality") == TEXT_MODALITY]
        caption_docs = [d for d in docs if d.metadata.get("modality") == CAPTION_MODALITY]
        if not text_docs or not caption_docs:
            raise ValueError(
                f"mm store must contain both modalities, got "
                f"{len(text_docs)} text / {len(caption_docs)} caption docs"
            )

        # Page-keyed caption lookup for co-retrieval.
        self.caption_by_page = {_page_key(d.metadata): d for d in caption_docs}

        k = self.config.first_stage_k
        self._text_lane = EnsembleRetriever(
            retrievers=[
                create_bm25_retriever(text_docs, k=k),
                mm_store.as_retriever(
                    search_kwargs={"k": k, "filter": {"modality": TEXT_MODALITY}}
                ),
            ],
            weights=[0.5, 0.5],
        )
        self._caption_lane = EnsembleRetriever(
            retrievers=[
                create_bm25_retriever(caption_docs, k=k),
                mm_store.as_retriever(
                    search_kwargs={"k": k, "filter": {"modality": CAPTION_MODALITY}}
                ),
            ],
            weights=[0.5, 0.5],
        )

    def retrieve_text(self, query: str, k: int | None = None) -> list[Document]:
        """Text lane: hybrid over text chunks -> rerank -> top k."""
        k = k or self.config.text_k
        candidates = self._text_lane.invoke(query)
        return rerank_documents(self.reranker, query, candidates, top_k=k)

    def retrieve_captions(self, query: str, k: int | None = None) -> list[Document]:
        """Caption lane: hybrid over caption chunks -> rerank -> top k.

        Used standalone as the caption-only arm (C1-cap).
        """
        k = k or self.config.caption_k
        candidates = self._caption_lane.invoke(query)
        return rerank_documents(self.reranker, query, candidates, top_k=k)

    def invoke(self, query: str) -> list[Document]:
        """Full C1 retrieval: text top-k + captions (co-retrieved, then lane)."""
        text_hits = self.retrieve_text(query)

        # Co-retrieval: captions of the text-hit pages, in text-rank order.
        captions: list[Document] = []
        seen: set[tuple[str, str, int]] = set()
        for doc in text_hits:
            key = _page_key(doc.metadata)
            cap = self.caption_by_page.get(key)
            if cap is not None and key not in seen:
                captions.append(cap)
                seen.add(key)

        # Caption lane fills remaining slots (skip pages already covered).
        if len(captions) < self.config.caption_k:
            for cap in self.retrieve_captions(query, k=self.config.caption_k + 3):
                key = _page_key(cap.metadata)
                if key not in seen:
                    captions.append(cap)
                    seen.add(key)
                if len(captions) >= self.config.caption_k:
                    break

        return text_hits + captions[: self.config.caption_k]
