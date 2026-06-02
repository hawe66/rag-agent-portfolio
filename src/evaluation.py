"""
RAGAS evaluation utilities for RAG pipeline assessment.

Tested with ragas==0.1.21. API may differ in other versions.
See: https://docs.ragas.io/en/v0.1.21/getstarted/evaluation.html

Metrics used:
- faithfulness: Is the response grounded in retrieved contexts?
- answer_relevancy: Is the response relevant to the question?
- context_precision: Are retrieved contexts relevant? (requires ground_truth)
- context_recall: Can we retrieve all info needed? (requires ground_truth)
"""

import json
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# RAGAS 0.1.x imports - lowercase metric instances
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


@dataclass
class EvalQuestion:
    """Single evaluation question with metadata."""
    id: str
    question: str
    category: str
    question_type: str
    retrieval_bias: str
    expected_section: str
    reference: str | None = None  # Ground truth answer (optional)
    requires_image: bool = False


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    """Load evaluation questions from JSON file."""
    with open(path) as f:
        data = json.load(f)

    questions = []
    for q in data["questions"]:
        questions.append(EvalQuestion(
            id=q["id"],
            question=q["question"],
            category=q["category"],
            question_type=q["question_type"],
            retrieval_bias=q["retrieval_bias"],
            expected_section=q["expected_section"],
            reference=q.get("reference"),
            requires_image=q.get("requires_image", False),
        ))
    return questions


def run_rag_pipeline(
    question: str,
    retriever,
    llm,
    k: int = 5,
) -> tuple[str, list[str]]:
    """
    Run RAG pipeline: retrieve contexts and generate response.

    Returns:
        (response, retrieved_contexts)
    """
    # Retrieve
    docs: list[Document] = retriever.invoke(question)[:k]
    contexts = [doc.page_content for doc in docs]

    # Generate response
    context_str = "\n\n".join(contexts)
    prompt = f"""다음 컨텍스트를 바탕으로 질문에 답하세요.

컨텍스트:
{context_str}

질문: {question}

답변:"""

    response = llm.invoke(prompt)
    return response.content, contexts


def evaluate_single(
    question: str,
    response: str,
    contexts: list[str],
    reference: str | None = None,
) -> dict:
    """
    Evaluate a single RAG response with RAGAS metrics.

    Args:
        question: User question
        response: Generated response
        contexts: Retrieved context strings
        reference: Ground truth answer (required for context_precision/recall)

    Returns:
        Dict with metric scores
    """
    # Build HuggingFace Dataset format for RAGAS 0.1.x
    data = {
        "question": [question],
        "contexts": [contexts],  # List of list of strings
        "answer": [response],
    }
    if reference:
        data["ground_truth"] = [reference]

    dataset = Dataset.from_dict(data)

    # Select metrics based on available data
    metrics = [faithfulness, answer_relevancy]
    if reference:
        metrics.extend([context_precision, context_recall])

    result = evaluate(dataset, metrics=metrics)

    return result.to_pandas().iloc[0].to_dict()


def evaluate_retriever(
    questions: list[EvalQuestion],
    retriever,
    llm: ChatOpenAI | None = None,
    k: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a retriever on a set of questions.

    Args:
        questions: List of EvalQuestion objects
        retriever: LangChain retriever
        llm: LLM for generation and evaluation
        k: Number of contexts to retrieve
        verbose: Print progress

    Returns:
        Dict with aggregated metrics and per-question results
    """
    llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    results = []

    for q in questions:
        if verbose:
            print(f"Evaluating {q.id}: {q.question[:30]}...")

        # Run RAG
        response, contexts = run_rag_pipeline(q.question, retriever, llm, k)

        # Evaluate
        scores = evaluate_single(
            question=q.question,
            response=response,
            contexts=contexts,
            reference=q.reference,
        )

        results.append({
            "id": q.id,
            "question": q.question,
            "category": q.category,
            "question_type": q.question_type,
            "retrieval_bias": q.retrieval_bias,
            "response": response,
            "contexts": contexts,
            **scores,
        })

    # Aggregate
    faithfulness_scores = [r.get("faithfulness", 0) for r in results if r.get("faithfulness") is not None]
    relevancy_scores = [r.get("answer_relevancy", 0) for r in results if r.get("answer_relevancy") is not None]
    precision_scores = [r.get("context_precision", 0) for r in results if r.get("context_precision") is not None]
    recall_scores = [r.get("context_recall", 0) for r in results if r.get("context_recall") is not None]

    aggregated = {
        "faithfulness_mean": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "answer_relevancy_mean": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None,
        "context_precision_mean": sum(precision_scores) / len(precision_scores) if precision_scores else None,
        "context_recall_mean": sum(recall_scores) / len(recall_scores) if recall_scores else None,
        "n_questions": len(questions),
    }

    return {
        "aggregated": aggregated,
        "results": results,
    }


def category_accuracy(
    questions: list[EvalQuestion],
    retriever,
    k: int = 5,
) -> dict:
    """
    Compute category accuracy: does the top retrieved doc match expected category?

    This is a fast sanity check that doesn't require LLM calls.
    """
    correct = 0
    results = []

    for q in questions:
        docs = retriever.invoke(q.question)[:k]
        if not docs:
            results.append({"id": q.id, "correct": False, "reason": "no docs"})
            continue

        top_category = docs[0].metadata.get("category", "unknown")
        is_correct = top_category == q.category
        if is_correct:
            correct += 1

        results.append({
            "id": q.id,
            "expected": q.category,
            "actual": top_category,
            "correct": is_correct,
        })

    return {
        "accuracy": correct / len(questions) if questions else 0,
        "correct": correct,
        "total": len(questions),
        "results": results,
    }


if __name__ == "__main__":
    # Smoke test
    from src.vectorstore import load_vectorstore

    CHROMA_DIR = Path("data/chroma_db_c3")
    EVAL_PATH = Path("docs/eval_questions_v2.json")

    print("Loading vectorstore...")
    vs = load_vectorstore(CHROMA_DIR, collection_name="lg_manuals_c3")
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    print("Loading eval questions...")
    questions = load_eval_questions(EVAL_PATH)
    print(f"Loaded {len(questions)} questions")

    # Quick category accuracy test (no LLM)
    print("\n--- Category Accuracy ---")
    cat_result = category_accuracy(questions[:5], retriever)
    print(f"Accuracy: {cat_result['accuracy']:.1%} ({cat_result['correct']}/{cat_result['total']})")

    # RAGAS smoke test on 2 questions
    print("\n--- RAGAS Smoke Test (2 questions) ---")
    ragas_result = evaluate_retriever(questions[:2], retriever, verbose=True)
    print(f"Faithfulness: {ragas_result['aggregated']['faithfulness_mean']:.3f}")
    print(f"Answer Relevancy: {ragas_result['aggregated']['answer_relevancy_mean']:.3f}")
