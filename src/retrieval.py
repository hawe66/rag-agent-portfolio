"""
Retrieval utilities for hybrid search (BM25 + Dense + RRF) and reranking.

Hybrid search addresses dense-only retrieval limitations:
- Model number exact match failure (e.g., "AS281DAW")
- Explicit category keywords ignored by dense embeddings
- Cross-category term confusion (e.g., "필터" appears in multiple categories)

RRF (Reciprocal Rank Fusion) combines rankings without score normalization:
    score = sum(1 / (k + rank)) where k=60 (typical)

2-Stage Retrieval:
    1. First stage: Hybrid retrieves k=20 candidates
    2. Second stage: Cross-encoder reranks to top-5
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retriever."""
    bm25_k: int = 5  # Number of results from BM25
    dense_k: int = 5  # Number of results from dense
    bm25_weight: float = 0.5  # Weight for BM25 in RRF
    dense_weight: float = 0.5  # Weight for dense in RRF


@dataclass
class RerankConfig:
    """Configuration for reranking."""
    model_name: str = "dragonkue/bge-reranker-v2-m3-ko"  # Korean-specialized
    first_stage_k: int = 20  # Candidates from first stage
    final_k: int = 5  # Final results after reranking


def extract_documents_from_vectorstore(vectorstore: Chroma) -> list[Document]:
    """
    Extract all documents from a Chroma vectorstore for BM25 indexing.

    Args:
        vectorstore: Chroma vectorstore instance

    Returns:
        List of LangChain Document objects
    """
    all_data = vectorstore._collection.get(include=['documents', 'metadatas'])
    texts = all_data['documents']
    metadatas = all_data['metadatas']

    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(texts, metadatas)
    ]


def create_bm25_retriever(
    documents: list[Document],
    k: int = 5,
) -> BM25Retriever:
    """
    Create a BM25 retriever from documents.

    Note on Korean tokenization:
    - Default uses space-based tokenization
    - Limitation: "필터를" vs "필터는" won't match
    - Future improvement: kiwipiepy or konlpy for morpheme-based tokenization

    Args:
        documents: List of LangChain Documents
        k: Number of results to return

    Returns:
        BM25Retriever instance
    """
    return BM25Retriever.from_documents(documents, k=k)


def create_hybrid_retriever(
    vectorstore: Chroma,
    config: HybridRetrieverConfig | None = None,
) -> EnsembleRetriever:
    """
    Create a hybrid retriever combining BM25 and dense search.

    Args:
        vectorstore: Chroma vectorstore for dense retrieval
        config: Configuration for retriever parameters

    Returns:
        EnsembleRetriever with RRF fusion
    """
    config = config or HybridRetrieverConfig()

    # Extract documents for BM25
    documents = extract_documents_from_vectorstore(vectorstore)

    # Create component retrievers
    bm25_retriever = create_bm25_retriever(documents, k=config.bm25_k)
    dense_retriever = vectorstore.as_retriever(search_kwargs={'k': config.dense_k})

    # Combine with RRF
    return EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[config.bm25_weight, config.dense_weight],
    )


def evaluate_retriever_accuracy(
    retriever,
    questions: list,
    k: int = 5,
) -> dict:
    """
    Evaluate top-1 and top-k category accuracy for a retriever.

    Args:
        retriever: LangChain retriever
        questions: List of EvalQuestion objects (must have .question and .category)
        k: Number of results to consider for top-k accuracy

    Returns:
        Dict with top1 and topk accuracy ratios
    """
    top1_correct = 0
    topk_correct = 0

    for q in questions:
        docs = retriever.invoke(q.question)[:k]
        categories = [d.metadata.get('category', 'unknown') for d in docs]

        if categories and categories[0] == q.category:
            top1_correct += 1
        if q.category in categories:
            topk_correct += 1

    n = len(questions)
    return {
        'top1': top1_correct / n if n > 0 else 0,
        f'top{k}': topk_correct / n if n > 0 else 0,
        'top1_correct': top1_correct,
        f'top{k}_correct': topk_correct,
        'total': n,
    }


def create_reranker(model_name: str = "dragonkue/bge-reranker-v2-m3-ko"):
    """
    Create a cross-encoder reranker for 2-stage retrieval.

    Args:
        model_name: HuggingFace model name for cross-encoder

    Returns:
        CrossEncoder instance
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank_documents(
    reranker,
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Rerank documents using cross-encoder.

    Args:
        reranker: CrossEncoder instance
        query: User query
        documents: List of candidate documents
        top_k: Number of results to return

    Returns:
        Reranked list of documents (top_k)
    """
    if not documents:
        return []

    # Score all query-document pairs
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    # Sort by score descending
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    return [doc for _, doc in scored_docs[:top_k]]


class HybridRerankerRetriever:
    """
    2-stage retriever: Hybrid first stage + Cross-encoder reranking.

    This addresses cases where correct documents are in top-20 but not top-5,
    by using a more precise cross-encoder to promote relevant documents.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        hybrid_config: HybridRetrieverConfig | None = None,
        rerank_config: RerankConfig | None = None,
    ):
        self.rerank_config = rerank_config or RerankConfig()

        # Override hybrid_config k values for first stage
        hybrid_config = hybrid_config or HybridRetrieverConfig()
        hybrid_config.bm25_k = self.rerank_config.first_stage_k
        hybrid_config.dense_k = self.rerank_config.first_stage_k

        self.hybrid_retriever = create_hybrid_retriever(vectorstore, hybrid_config)
        self.reranker = create_reranker(self.rerank_config.model_name)

    def invoke(self, query: str) -> list[Document]:
        """Retrieve and rerank documents."""
        # First stage: get candidates
        candidates = self.hybrid_retriever.invoke(query)

        # Second stage: rerank
        return rerank_documents(
            self.reranker,
            query,
            candidates,
            top_k=self.rerank_config.final_k,
        )


if __name__ == "__main__":
    import time
    from src.vectorstore import load_vectorstore
    from src.evaluation import load_eval_questions

    CHROMA_DIR = Path("data/chroma_db_c3")
    EVAL_PATH = Path("docs/eval_questions_v2.json")

    print("Loading vectorstore...")
    vs = load_vectorstore(CHROMA_DIR, collection_name="lg_manuals_c3")
    print(f"Loaded {vs._collection.count()} documents")

    print("\nLoading eval questions...")
    questions = load_eval_questions(EVAL_PATH)
    print(f"Loaded {len(questions)} questions")

    print("\nCreating retrievers...")
    docs = extract_documents_from_vectorstore(vs)
    print(f"Extracted {len(docs)} documents for BM25")

    # Dense only
    dense_retriever = vs.as_retriever(search_kwargs={'k': 5})

    # BM25 only
    bm25_retriever = create_bm25_retriever(docs, k=5)

    # Hybrid
    hybrid_retriever = create_hybrid_retriever(vs)

    # Hybrid + Rerank
    print("Loading reranker model (dragonkue/bge-reranker-v2-m3-ko)...")
    rerank_retriever = HybridRerankerRetriever(vs)

    print("\n=== Evaluation Results ===\n")

    print("Dense Only:")
    t0 = time.time()
    dense_acc = evaluate_retriever_accuracy(dense_retriever, questions)
    dense_time = time.time() - t0
    print(f"  Top-1: {dense_acc['top1']:.1%} ({dense_acc['top1_correct']}/{dense_acc['total']})")
    print(f"  Top-5: {dense_acc['top5']:.1%} ({dense_acc['top5_correct']}/{dense_acc['total']})")
    print(f"  Time: {dense_time:.2f}s")

    print("\nBM25 Only:")
    t0 = time.time()
    bm25_acc = evaluate_retriever_accuracy(bm25_retriever, questions)
    bm25_time = time.time() - t0
    print(f"  Top-1: {bm25_acc['top1']:.1%} ({bm25_acc['top1_correct']}/{bm25_acc['total']})")
    print(f"  Top-5: {bm25_acc['top5']:.1%} ({bm25_acc['top5_correct']}/{bm25_acc['total']})")
    print(f"  Time: {bm25_time:.2f}s")

    print("\nHybrid (BM25 + Dense):")
    t0 = time.time()
    hybrid_acc = evaluate_retriever_accuracy(hybrid_retriever, questions)
    hybrid_time = time.time() - t0
    print(f"  Top-1: {hybrid_acc['top1']:.1%} ({hybrid_acc['top1_correct']}/{hybrid_acc['total']})")
    print(f"  Top-5: {hybrid_acc['top5']:.1%} ({hybrid_acc['top5_correct']}/{hybrid_acc['total']})")
    print(f"  Time: {hybrid_time:.2f}s")

    print("\nHybrid + Rerank:")
    t0 = time.time()
    rerank_acc = evaluate_retriever_accuracy(rerank_retriever, questions)
    rerank_time = time.time() - t0
    print(f"  Top-1: {rerank_acc['top1']:.1%} ({rerank_acc['top1_correct']}/{rerank_acc['total']})")
    print(f"  Top-5: {rerank_acc['top5']:.1%} ({rerank_acc['top5_correct']}/{rerank_acc['total']})")
    print(f"  Time: {rerank_time:.2f}s")

    print("\n=== Summary ===")
    print("| Retriever | Top-1 | Top-5 | Time |")
    print("|-----------|-------|-------|------|")
    print(f"| Dense Only | {dense_acc['top1']:.1%} | {dense_acc['top5']:.1%} | {dense_time:.2f}s |")
    print(f"| BM25 Only | {bm25_acc['top1']:.1%} | {bm25_acc['top5']:.1%} | {bm25_time:.2f}s |")
    print(f"| Hybrid (RRF) | {hybrid_acc['top1']:.1%} | {hybrid_acc['top5']:.1%} | {hybrid_time:.2f}s |")
    print(f"| Hybrid + Rerank | {rerank_acc['top1']:.1%} | {rerank_acc['top5']:.1%} | {rerank_time:.2f}s |")
