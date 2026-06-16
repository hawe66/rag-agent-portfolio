"""
RAGAS evaluation utilities for RAG pipeline assessment.

Updated for ragas>=0.4.0.
See: https://docs.ragas.io/en/latest/

Metrics used:
- Faithfulness: Is the response grounded in retrieved contexts?
- ResponseRelevancy: Is the response relevant to the question?
- ContextPrecision: Are retrieved contexts relevant? (requires reference)
- ContextRecall: Can we retrieve all info needed? (requires reference)
"""

import csv
import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import MagicMock

from dotenv import load_dotenv
load_dotenv()

# Monkey patch: ragas 0.4.x tries to import deprecated langchain_community.chat_models.vertexai
# This module was moved to langchain-google-vertexai in langchain-community 0.4.x
# We create a fake module to prevent ImportError
_fake_vertexai = MagicMock()
_fake_vertexai.ChatVertexAI = MagicMock()
sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

# RAGAS 0.4.x imports - PascalCase metric classes
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
    ContextRecall,
)


MODALITY_LABELS = ("text-only", "image-helpful", "image-required")


@dataclass
class EvalQuestion:
    """Single evaluation question.

    The SSOT is `data/eval/golden_set_v*.csv`. The legacy JSON loader fills the
    same shape so existing callers (W6 notebook, W7 evaluation) keep working.

    Cross-modal fields (`modality_label`, `reference_context`, `figure_ref`)
    are populated from the CSV. `requires_image` is derived for hosts that
    still read it; do not pass it in.
    """

    id: str
    question: str
    category: str
    question_type: str
    retrieval_bias: str
    expected_section: str
    reference: str | None = None
    modality_label: str = "text-only"
    reference_context: str = ""
    figure_ref: str = ""
    notes: str = ""

    @property
    def requires_image(self) -> bool:
        return self.modality_label == "image-required"

    @property
    def q_type(self) -> str:
        """Alias matching the CSV column name."""
        return self.question_type

    @property
    def ground_truth(self) -> str | None:
        """Alias matching the CSV column name (RAGAS reference answer)."""
        return self.reference


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    """Load evaluation questions from the legacy JSON SSOT.

    Kept for the W6 notebook regression flow; new code should call
    ``load_golden_set`` against ``data/eval/golden_set_v*.csv``.
    """
    with open(path) as f:
        data = json.load(f)

    questions = []
    for q in data["questions"]:
        modality_label = "image-required" if q.get("requires_image", False) else "text-only"
        questions.append(EvalQuestion(
            id=q["id"],
            question=q["question"],
            category=q["category"],
            question_type=q["question_type"],
            retrieval_bias=q["retrieval_bias"],
            expected_section=q["expected_section"],
            reference=q.get("reference"),
            modality_label=modality_label,
        ))
    return questions


# Mapping CSV `reference_context` prefixes → category metadata used by retrievers.
_CATEGORY_PREFIXES = {
    "airpurifier": "airpurifier",
    "waterpurifier": "waterpurifier",
    "vacuumcleaner": "vacuumcleaner",
}

_REFCTX_HEAD = re.compile(r"^\s*([a-z]+)_(simple|complex)\s+p\.?(\d+(?:-\d+)?)\s*(?:\(([^)]+)\))?", re.IGNORECASE)


def _parse_reference_context(reference_context: str) -> tuple[str, str]:
    """Pull category + expected_section out of a CSV reference_context string.

    Example: ``waterpurifier_simple p.15 (정수 필터 교체하기)`` ->
    ``("waterpurifier", "정수 필터 교체하기")``.
    """
    if not reference_context:
        return "unknown", ""
    match = _REFCTX_HEAD.match(reference_context)
    if not match:
        return "unknown", ""
    prefix = match.group(1).lower()
    section = (match.group(4) or "").strip()
    return _CATEGORY_PREFIXES.get(prefix, "unknown"), section


_ID_FROM_NOTES = re.compile(r"\b(Q\d+|IR-[A-Z]\d+)\b")


def _id_from_row(notes: str, fallback_index: int) -> str:
    """Notes column commonly carries `Q01 from v2` — pull the ID, else synthesize."""
    if notes:
        match = _ID_FROM_NOTES.search(notes)
        if match:
            return match.group(1)
    return f"R{fallback_index + 1:03d}"


def load_golden_set(path: Path) -> list[EvalQuestion]:
    """Load evaluation questions from the golden-set CSV (SSOT for W7+).

    CSV columns: ``question, ground_truth, reference_context, q_type,
    modality_label, notes`` (plus optional ``figure_ref`` once v2 lands).
    Unknown columns are ignored so the loader survives forward-compatible
    additions.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    questions = []
    for index, row in enumerate(rows):
        reference_context = (row.get("reference_context") or "").strip()
        notes = (row.get("notes") or "").strip()
        category, section = _parse_reference_context(reference_context)
        modality_label = (row.get("modality_label") or "text-only").strip()
        if modality_label not in MODALITY_LABELS:
            modality_label = "text-only"

        questions.append(EvalQuestion(
            id=_id_from_row(notes, index),
            question=(row.get("question") or "").strip(),
            category=category,
            question_type=(row.get("q_type") or "").strip(),
            retrieval_bias="",  # not tracked in golden set
            expected_section=section,
            reference=(row.get("ground_truth") or "").strip() or None,
            modality_label=modality_label,
            reference_context=reference_context,
            figure_ref=(row.get("figure_ref") or "").strip(),
            notes=notes,
        ))
    return questions


# ---------------------------------------------------------------------------
# Domain metrics — Refusal & Citation (hoisted from w7 notebook cells 14, 18)
# ---------------------------------------------------------------------------


REFUSAL_PATTERNS: tuple[str, ...] = (
    r"제공된 문서에서 확인할 수 없습니다",
    r"문서에서 확인할 수 없",
    r"정보를 찾을 수 없",
    r"해당 정보가 없",
    r"포함되어 있지 않",
)

_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS))


REFUSAL_REQUIRED_Q_TYPES: tuple[str, ...] = ("out_of_scope",)
"""q_types whose questions should be refused. Safety questions are expected
to answer with appropriate caveats, not refuse, per the W7 sprint-1 run."""


def is_refusal(response: str) -> bool:
    """True when the answer matches one of the project's refusal patterns."""
    return bool(_REFUSAL_RE.search(response))


# Citation extraction
# -------------------
# The W7 sprint-1 regex `((?:water…|air…|vacuum…)_\w+)\s*p?\.?(\d+)` assumed
# the model token and page sat next to each other, but actual answers wrap
# the model token with a PDF filename — e.g.
#     "waterpurifier_simple_WP_KOR_MFL71817002_03_240524_00_OM_WEB.pdf p.1"
# `_\w+` swallowed the filename and the regex never reached `p.1`, so cell 19
# reported Citation Accuracy = 0% even though answers DID cite sources.
#
# Fix: restrict the model token to `category_(simple|complex)` (the form that
# also appears in `reference_context`), then skip non-`)` characters lazily
# until we find `p<digits>`. The lazy gap traverses `.pdf` and any digits in
# the filename because nothing else competes for `p\.?\s*\d+`.
_CITATION_RE = re.compile(
    r"(?P<model>(?:waterpurifier|airpurifier|vacuumcleaner)_(?:simple|complex))"
    r"[^)\n]*?"
    r"p\.?\s*(?P<page>\d+)",
    re.IGNORECASE,
)


# Reference contexts come straight from the CSV ("waterpurifier_simple p.15
# (정수 필터 교체하기)") so we can use a tighter pattern that requires the
# page to follow the model token.
_REFCTX_CITATION_RE = re.compile(
    r"(?P<model>(?:waterpurifier|airpurifier|vacuumcleaner)_(?:simple|complex))"
    r"\s+p\.?\s*(?P<page>\d+)",
    re.IGNORECASE,
)


def extract_citations(response: str) -> list[tuple[str, int]]:
    """Pull (model_complexity, page) pairs from a generated answer."""
    return [
        (m.group("model").lower(), int(m.group("page")))
        for m in _CITATION_RE.finditer(response)
    ]


def _parse_reference_citation(reference_context: str) -> tuple[str, int] | None:
    """Read the expected (model_complexity, page) out of a CSV reference_context."""
    if not reference_context:
        return None
    match = _REFCTX_CITATION_RE.search(reference_context)
    if not match:
        return None
    return match.group("model").lower(), int(match.group("page"))


def refusal_accuracy(
    questions: list[EvalQuestion],
    answers: list[str],
) -> dict:
    """Refusal vs answer correctness with a per-q_type breakdown.

    A question scores correct if the model refused when it should and
    answered when it should. ``fp_rate`` is the share of *answerable*
    questions (anything not in ``REFUSAL_REQUIRED_Q_TYPES``) that the model
    wrongly refused.
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must align 1:1")

    total = correct = fp = answerable = 0
    by_q_type: dict[str, dict[str, int]] = {}

    for q, ans in zip(questions, answers):
        refused = is_refusal(ans)
        should_refuse = q.question_type in REFUSAL_REQUIRED_Q_TYPES
        bucket = by_q_type.setdefault(q.question_type, {"correct": 0, "total": 0})
        bucket["total"] += 1
        total += 1
        if refused == should_refuse:
            correct += 1
            bucket["correct"] += 1
        if not should_refuse:
            answerable += 1
            if refused:
                fp += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "fp_rate": fp / answerable if answerable else 0.0,
        "by_q_type": {
            qt: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for qt, stats in by_q_type.items()
        },
    }


def citation_accuracy(
    questions: list[EvalQuestion],
    answers: list[str],
    page_tolerance: int = 2,
) -> dict:
    """Citation correctness with a per-q_type breakdown.

    Out-of-scope questions are excluded from the denominator since they have
    no expected source. A question scores correct when at least one extracted
    citation matches the reference's model token and page (±``page_tolerance``).
    """
    if len(questions) != len(answers):
        raise ValueError("questions and answers must align 1:1")

    total = correct = 0
    by_q_type: dict[str, dict[str, int]] = {}

    for q, ans in zip(questions, answers):
        if q.question_type == "out_of_scope":
            continue
        bucket = by_q_type.setdefault(q.question_type, {"correct": 0, "total": 0})
        bucket["total"] += 1
        total += 1

        expected = _parse_reference_citation(q.reference_context)
        if expected is None:
            continue

        expected_model, expected_page = expected
        expected_category = expected_model.split("_", 1)[0]
        for cite_model, cite_page in extract_citations(ans):
            cite_category = cite_model.split("_", 1)[0]
            if cite_category == expected_category and abs(cite_page - expected_page) <= page_tolerance:
                correct += 1
                bucket["correct"] += 1
                break

    return {
        "accuracy": correct / total if total else 0.0,
        "n": total,
        "by_q_type": {
            qt: {**stats, "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0}
            for qt, stats in by_q_type.items()
        },
    }


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

    # Embed source tag so LLM can cite back as "{cat}_{cplx} p.{page}".
    # RAGAS gets raw `contexts` (no tags) — only the prompt sees tags.
    def _tag(doc: Document) -> str:
        cat = doc.metadata.get("category", "unknown")
        cplx = doc.metadata.get("complexity", "")
        page = doc.metadata.get("page", "?")
        head = f"{cat}_{cplx}" if cplx else cat
        return f"[{head} p.{page}]"

    context_str = "\n\n".join(f"{_tag(d)} {d.page_content}" for d in docs)
    prompt = f"""다음 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트 앞에 `[제품_복잡도 p.페이지]` 형태의 출처 태그가 붙어 있습니다.
답변 끝에 `(출처: 제품_복잡도 p.페이지)` 형태로 사용한 근거의 출처를 명시하세요.

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
    # Build HuggingFace Dataset format for RAGAS 0.4.x
    data = {
        "user_input": [question],
        "retrieved_contexts": [contexts],  # List of list of strings
        "response": [response],
    }
    if reference:
        data["reference"] = [reference]

    dataset = Dataset.from_dict(data)

    # Select metrics based on available data (instantiate metric classes)
    metrics = [Faithfulness(), ResponseRelevancy()]
    if reference:
        metrics.extend([ContextPrecision(), ContextRecall()])

    result = evaluate(dataset=dataset, metrics=metrics)

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

    # Aggregate (ragas 0.4.x uses snake_case metric names in results)
    faithfulness_scores = [r.get("faithfulness", 0) for r in results if r.get("faithfulness") is not None]
    relevancy_scores = [r.get("response_relevancy", 0) for r in results if r.get("response_relevancy") is not None]
    precision_scores = [r.get("context_precision", 0) for r in results if r.get("context_precision") is not None]
    recall_scores = [r.get("context_recall", 0) for r in results if r.get("context_recall") is not None]

    aggregated = {
        "faithfulness_mean": sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None,
        "response_relevancy_mean": sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None,
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
    print(f"Response Relevancy: {ragas_result['aggregated']['response_relevancy_mean']:.3f}")
