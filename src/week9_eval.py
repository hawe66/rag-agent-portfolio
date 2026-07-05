"""
C0 / C1 / C2 comparison evaluation (Week 9, WEEK9_TASKS §6).

Compares four configurations on the *same* test set, *same* answer model
(gpt-4o-mini), and *same* LLM-judge so the only variable is how visual
information enters the pipeline:

    C0  text-only RAG               (chroma_db_c3)        — ADR-011 baseline
    C1  caption→text RAG            (chroma_db_mm)        — VLM reasoning done offline
    C2a CLIP retrieval only         (clip_index)          — find, no reading
    C2b CLIP retrieval + VLM read   (clip_index + vision) — late fusion

Two axes (WEEK9_TASKS §6):
- Retrieval page-hit: does top-k contain the reference page (±tol), same manual?
- Answer correctness: LLM-judge (0/1) vs ground_truth.

C2a has no reading step, so by construction it produces no answer
(answer_correct == 0); we still measure its retrieval page-hit.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .evaluation import EvalQuestion, load_golden_set, _parse_reference_citation
from .retrieval import create_hybrid_retriever, HybridRetrieverConfig, create_reranker, rerank_documents
from .multimodal import load_mm_store, _encode_image, _pace
from .clip_index import ClipIndex, clip_retrieve

ANSWER_MODEL = "gpt-4o-mini"
PAGE_TOLERANCE = 2
TOP_K = 5

# IR8 ids (golden_set_v2 image-required). Text-only contrast ids picked below.
IR8_IDS = ("IR-A1", "IR-A2", "IR-A3", "IR-W1", "IR-W2", "IR-V1", "IR-V2", "IR-V3")
TEXT_ONLY_CONTRAST_IDS = ("Q01", "Q08", "Q15", "Q22")  # one per category + extra


@dataclass
class RefTarget:
    category: str
    complexity: str
    page: int


def reference_target(q: EvalQuestion) -> RefTarget | None:
    """Pull (category, complexity, page) the question's figure/text lives on."""
    parsed = _parse_reference_citation(q.reference_context)
    if parsed is None:
        return None
    model, page = parsed  # model == "airpurifier_complex"
    category, complexity = model.split("_", 1)
    return RefTarget(category=category, complexity=complexity, page=page)


def page_hit(retrieved: list[tuple[str, str, int]], ref: RefTarget, tol: int = PAGE_TOLERANCE) -> bool:
    """True if any retrieved (cat, cplx, page) matches the same manual within ±tol."""
    for category, complexity, page in retrieved:
        if (category == ref.category and complexity == ref.complexity
                and page is not None and abs(page - ref.page) <= tol):
            return True
    return False


# ---------------------------------------------------------------------------
# Retrievers (C0 / C1 share the hybrid+rerank pipeline over different stores)
# ---------------------------------------------------------------------------


def make_text_retriever(vectorstore, reranker, first_stage_k: int = 20, top_k: int = TOP_K):
    """Hybrid (BM25+Dense, RRF) → cross-encoder rerank, returns top_k Documents."""
    hybrid = create_hybrid_retriever(
        vectorstore,
        HybridRetrieverConfig(bm25_k=first_stage_k, dense_k=first_stage_k),
    )

    def retrieve(query: str) -> list[Document]:
        candidates = hybrid.invoke(query)
        return rerank_documents(reranker, query, candidates, top_k=top_k)

    return retrieve


def docs_to_pages(docs: list[Document]) -> list[tuple[str, str, int]]:
    return [
        (d.metadata.get("category", "unknown"),
         d.metadata.get("complexity", ""),
         d.metadata.get("page"))
        for d in docs
    ]


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


ANSWER_PROMPT = """당신은 LG 가전제품 매뉴얼 기반으로 답하는 도우미입니다.
제공된 컨텍스트에 있는 정보만 사용하세요. 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.

## 질문
{question}

## 참고 문서
{context}

## 답변"""

VISION_ANSWER_PROMPT = """당신은 LG 가전제품 매뉴얼 기반으로 답하는 도우미입니다.
아래 매뉴얼 페이지 이미지를 직접 보고 질문에 답하세요.
이미지에서 보이는 정보만 사용하고, 보이지 않으면 "이미지에서 확인할 수 없습니다."라고 답하세요.

질문: {question}"""


def _tag(doc: Document) -> str:
    cat = doc.metadata.get("category", "unknown")
    cplx = doc.metadata.get("complexity", "")
    page = doc.metadata.get("page", "?")
    head = f"{cat}_{cplx}" if cplx else cat
    return f"[{head} p.{page}]"


def answer_from_text(question: str, docs: list[Document], llm: ChatOpenAI) -> str:
    """C0/C1 answer: generate from retrieved text/caption chunks."""
    context = "\n\n".join(f"{_tag(d)} {d.page_content}" for d in docs)
    return llm.invoke(ANSWER_PROMPT.format(question=question, context=context)).content


def answer_from_images(question: str, image_paths: list[str], client: OpenAI, max_images: int = 1) -> str:
    """C2b answer: gpt-4o-mini reads the CLIP-retrieved page image(s) (late fusion).

    Defaults to the CLIP top-1 page: each high-detail page is ~26k tokens, so we
    pace the call (shared limiter with captioning) to stay under the 200k TPM tier.
    """
    content: list[dict] = [{"type": "text", "text": VISION_ANSWER_PROMPT.format(question=question)}]
    for path in image_paths[:max_images]:
        b64 = _encode_image(Path(path))
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })
    _pace()
    response = client.chat.completions.create(
        model=ANSWER_MODEL, temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


JUDGE_PROMPT = """당신은 답변 채점자입니다. 모범답안의 핵심 정보가 모델답변에 담겼는지 0/1로 채점하세요.

질문: {question}
모범답안: {ground_truth}
모델답변: {answer}

채점 기준:
- 모범답안의 핵심 사실(위치/방향/모양/수치 등)이 모델답변에 맞게 담겨 있으면 1.
- 핵심이 빠졌거나 틀렸거나, 답을 못 했으면 0.

JSON으로만 답하세요: {{"score": 0 또는 1, "reason": "한 줄 근거"}}"""


def judge_answer(question: str, ground_truth: str, answer: str, llm: ChatOpenAI) -> tuple[int, str]:
    raw = llm.invoke(JUDGE_PROMPT.format(
        question=question, ground_truth=ground_truth, answer=answer,
    )).content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        return int(parsed.get("score", 0)), str(parsed.get("reason", ""))
    except (json.JSONDecodeError, ValueError):
        return 0, f"judge parse error: {raw[:80]}"


# ---------------------------------------------------------------------------
# Per-config runners
# ---------------------------------------------------------------------------


def _run_text_config(questions, retrieve_fn, answer_llm, judge_llm) -> list[dict]:
    """C0 or C1: retrieve text/caption chunks → answer → judge."""
    rows = []
    for q in questions:
        ref = reference_target(q)
        docs = retrieve_fn(q.question)
        pages = docs_to_pages(docs)
        answer = answer_from_text(q.question, docs, answer_llm)
        score, reason = judge_answer(q.question, q.reference or "", answer, judge_llm)
        rows.append({
            "id": q.id,
            "retrieved": [[c, x, p] for c, x, p in pages],
            "page_hit": page_hit(pages, ref) if ref else None,
            "answer": answer,
            "judge": score,
            "judge_reason": reason,
        })
    return rows


def _run_clip_configs(questions, clip_index, vision_client, judge_llm) -> tuple[list[dict], list[dict]]:
    """C2a (retrieval only) and C2b (retrieval + VLM read) in one CLIP pass."""
    c2a_rows, c2b_rows = [], []
    for q in questions:
        ref = reference_target(q)
        hits = clip_retrieve(clip_index, q.question, k=TOP_K)
        pages = [(h["category"], h["complexity"], h["page"]) for h in hits]
        hit = page_hit(pages, ref) if ref else None

        # C2a: retrieval only — no reading step, so it cannot answer.
        c2a_rows.append({
            "id": q.id,
            "retrieved": [[c, x, p] for c, x, p in pages],
            "page_hit": hit,
            "answer": "(검색만 수행 — 읽기/추론 단계 없음)",
            "judge": 0,
            "judge_reason": "retrieval-only config has no reasoning step",
        })

        # C2b: feed retrieved page images to the VLM (late fusion).
        answer = answer_from_images(q.question, [h["image_path"] for h in hits], vision_client)
        score, reason = judge_answer(q.question, q.reference or "", answer, judge_llm)
        c2b_rows.append({
            "id": q.id,
            "retrieved": [[c, x, p] for c, x, p in pages],
            "page_hit": hit,
            "answer": answer,
            "judge": score,
            "judge_reason": reason,
        })
    return c2a_rows, c2b_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _summarize(rows: list[dict], ir8_ids: set[str]) -> dict:
    def split(predicate):
        sel = [r for r in rows if predicate(r["id"])]
        n = len(sel)
        ph = [r for r in sel if r["page_hit"] is not None]
        return {
            "n": n,
            "page_hit": sum(1 for r in ph if r["page_hit"]) / len(ph) if ph else None,
            "answer_correct": sum(r["judge"] for r in sel) / n if n else None,
        }
    return {
        "ir8": split(lambda i: i in ir8_ids),
        "text_only": split(lambda i: i not in ir8_ids),
        "per_q": rows,
    }


def run_comparison(
    golden_path: Path = Path("data/eval/golden_set_v2.csv"),
    mm_dir: Path = Path("data/chroma_db_mm"),
    c0_dir: Path = Path("data/chroma_db_c3"),
    clip_dir: Path = Path("data/clip_index"),
    out_path: Path = Path("data/week9_results.json"),
) -> dict:
    """Run C0/C1/C2a/C2b over IR8 + text-only contrast and persist results."""
    from .vectorstore import load_vectorstore

    all_q = load_golden_set(golden_path)
    by_id = {q.id: q for q in all_q}
    selected_ids = list(IR8_IDS) + [i for i in TEXT_ONLY_CONTRAST_IDS if i in by_id]
    questions = [by_id[i] for i in selected_ids if i in by_id]
    ir8_set = set(IR8_IDS)
    print(f"Test set: {len(questions)} questions ({len(ir8_set & set(selected_ids))} IR8 + "
          f"{len(questions) - len(ir8_set & set(selected_ids))} text-only)")

    answer_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
    judge_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
    vision_client = OpenAI(max_retries=8)  # C2b sends ~26k-token high-detail images

    print("Loading reranker (shared by C0/C1)...")
    reranker = create_reranker()

    print("C0: text-only RAG (chroma_db_c3)...")
    c0_store = load_vectorstore(c0_dir, collection_name="lg_manuals_c3")
    c0_rows = _run_text_config(questions, make_text_retriever(c0_store, reranker), answer_llm, judge_llm)

    print("C1: caption→text RAG (chroma_db_mm)...")
    c1_store = load_mm_store(mm_dir)
    c1_rows = _run_text_config(questions, make_text_retriever(c1_store, reranker), answer_llm, judge_llm)

    print("C2a/C2b: CLIP retrieval (+VLM read)...")
    clip_index = ClipIndex.load(clip_dir)
    c2a_rows, c2b_rows = _run_clip_configs(questions, clip_index, vision_client, judge_llm)

    results = {
        "configs": ["C0", "C1", "C2a", "C2b"],
        "page_tolerance": PAGE_TOLERANCE,
        "top_k": TOP_K,
        "questions": [{
            "id": q.id,
            "question": q.question,
            "modality": q.modality_label,
            "ground_truth": q.reference,
            "reference_context": q.reference_context,
        } for q in questions],
        "results": {
            "C0": _summarize(c0_rows, ir8_set),
            "C1": _summarize(c1_rows, ir8_set),
            "C2a": _summarize(c2a_rows, ir8_set),
            "C2b": _summarize(c2b_rows, ir8_set),
        },
    }
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    return results


def print_summary(results: dict) -> None:
    print("\n| Config | IR8 page-hit | IR8 answer | text-only page-hit | text-only answer |")
    print("|--------|-------------|------------|--------------------|------------------|")
    for cfg in results["configs"]:
        r = results["results"][cfg]
        def fmt(v):
            return f"{v:.0%}" if isinstance(v, (int, float)) else "n/a"
        print(f"| {cfg} | {fmt(r['ir8']['page_hit'])} | {fmt(r['ir8']['answer_correct'])} | "
              f"{fmt(r['text_only']['page_hit'])} | {fmt(r['text_only']['answer_correct'])} |")


if __name__ == "__main__":
    results = run_comparison()
    print_summary(results)
