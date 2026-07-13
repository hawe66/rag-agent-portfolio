"""
C0 / C1 / C2 comparison evaluation (Week 9 rework — WEEK9_DEFECTS §A/§E).

Configurations, same test set / answer model / judge across all:

    C0    text-only RAG            (chroma_db_c3, flat top-5)   — baseline
    C1    modality-aware mm RAG    (chroma_db_mm, text 3 + caption 2)
    C1cap caption-only RAG         (chroma_db_mm, caption lane top-5)
    C2a   CLIP retrieval only      (clip_index)  — no reading step
    C2b   CLIP retrieval + VLM read(clip_index + vision, top-1 image)

Metric layers (kept separate — see WEEK9_DEFECTS E1):
- retrieval_page_hit@5 : does top-k contain the reference page (±tol)?
  This is a RETRIEVAL-layer metric. Do NOT compare it against ADR-011's
  25%/50% anchor, which is a CITATION-layer metric.
- caption_hit@5        : does top-k contain the reference page's CAPTION?
  (only meaningful for stores that hold captions: C1/C1cap)
- citation (model/page): Week 7 definition — reuses evaluation.py's
  extract_citations/_parse_reference_citation on the generated answer.
  C0's IR8 numbers here must reproduce the ADR-011 anchor (~50%/~25%).
- answer_correct       : strict LLM-judge (gpt-4o, partial match = 0),
  a different model from the answerer to avoid self-preference (E4).

C2a has no reading step: its answer_correct is None (n/a by design), never 0
(E3). C2b answers are additionally summarized on the page-hit subset only,
since off-page images test nothing about late fusion (D7).
"""

import json
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .evaluation import (
    EvalQuestion,
    load_golden_set,
    extract_citations,
    _parse_reference_citation,
)
from .retrieval import create_hybrid_retriever, HybridRetrieverConfig, create_reranker, rerank_documents
from .multimodal import load_mm_store, _encode_image, _pace
from .mm_retrieval import ModalityAwareRetriever
from .clip_index import ClipIndex, clip_retrieve

ANSWER_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o"  # different from ANSWER_MODEL — E4 self-preference
PAGE_TOLERANCE = 2
TOP_K = 5

IR8_IDS = ("IR-A1", "IR-A2", "IR-A3", "IR-W1", "IR-W2", "IR-V1", "IR-V2", "IR-V3")
# Pure text-only contrast (E2): 2 per category, factual + one multi_hop.
# Q08 stays in deliberately — it is the D6 investigation target, and dropping
# a question because the baseline fails it would bias the contrast set.
TEXT_ONLY_CONTRAST_IDS = ("Q01", "Q06", "Q08", "Q11", "Q18", "Q19")


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


# ---------------------------------------------------------------------------
# Metric layer 1: retrieval
# ---------------------------------------------------------------------------


def docs_to_pages(docs: list[Document]) -> list[tuple[str, str, int, str]]:
    """Flatten retrieved docs to (category, complexity, page, modality)."""
    return [
        (d.metadata.get("category", "unknown"),
         d.metadata.get("complexity", ""),
         d.metadata.get("page"),
         d.metadata.get("modality", "text"))
        for d in docs
    ]


def retrieval_page_hit(
    retrieved: list[tuple[str, str, int, str]], ref: RefTarget, tol: int,
) -> bool:
    """True if any retrieved doc is in the same manual within ±tol pages."""
    return any(
        cat == ref.category and cplx == ref.complexity
        and page is not None and abs(page - ref.page) <= tol
        for cat, cplx, page, _ in retrieved
    )


def caption_hit(
    retrieved: list[tuple[str, str, int, str]], ref: RefTarget, tol: int,
) -> bool:
    """True if the reference page's CAPTION chunk is in the retrieved set."""
    return any(
        modality == "image-derived"
        and cat == ref.category and cplx == ref.complexity
        and page is not None and abs(page - ref.page) <= tol
        for cat, cplx, page, modality in retrieved
    )


# ---------------------------------------------------------------------------
# Metric layer 2: citation (Week 7 definition — ADR-011 anchor)
# ---------------------------------------------------------------------------


def citation_match(q: EvalQuestion, answer: str, tol: int = PAGE_TOLERANCE) -> dict | None:
    """Per-question citation check, same semantics as evaluation.citation_accuracy:
    model = cited category matches; page = cited page within ±tol of reference."""
    expected = _parse_reference_citation(q.reference_context)
    if expected is None:
        return None
    expected_model, expected_page = expected
    expected_category = expected_model.split("_", 1)[0]
    model_ok = page_ok = False
    for cite_model, cite_page in extract_citations(answer):
        if cite_model.split("_", 1)[0] == expected_category:
            model_ok = True
            if abs(cite_page - expected_page) <= tol:
                page_ok = True
    return {"model": model_ok, "page": page_ok}


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


def make_text_retriever(vectorstore, reranker, first_stage_k: int = 20, top_k: int = TOP_K):
    """C0: Hybrid (BM25+Dense, RRF) → cross-encoder rerank, flat top_k."""
    hybrid = create_hybrid_retriever(
        vectorstore,
        HybridRetrieverConfig(bm25_k=first_stage_k, dense_k=first_stage_k),
    )

    def retrieve(query: str) -> list[Document]:
        candidates = hybrid.invoke(query)
        return rerank_documents(reranker, query, candidates, top_k=top_k)

    return retrieve


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def _tag(doc: Document) -> str:
    cat = doc.metadata.get("category", "unknown")
    cplx = doc.metadata.get("complexity", "")
    page = doc.metadata.get("page", "?")
    head = f"{cat}_{cplx}" if cplx else cat
    return f"[{head} p.{page}]"


# Mirrors Week 7's run_rag_pipeline prompt (src/evaluation.py) so the
# citation layer (extract_citations) sees the same answer format that
# produced the ADR-011 anchor numbers.
ANSWER_PROMPT = """다음 컨텍스트를 바탕으로 질문에 답하세요.
각 컨텍스트 앞에 `[제품_복잡도 p.페이지]` 형태의 출처 태그가 붙어 있습니다.
컨텍스트에 있는 정보만 사용하세요. 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.
답변 끝에 `(출처: 제품_복잡도 p.페이지)` 형태로 사용한 근거의 출처를 명시하세요.

컨텍스트:
{context}

질문: {question}

답변:"""

VISION_ANSWER_PROMPT = """당신은 LG 가전제품 매뉴얼 기반으로 답하는 도우미입니다.
아래 매뉴얼 페이지 이미지를 직접 보고 질문에 답하세요.
이미지에서 보이는 정보만 사용하고, 보이지 않으면 "이미지에서 확인할 수 없습니다."라고 답하세요.

질문: {question}"""


def answer_from_text(question: str, docs: list[Document], llm: ChatOpenAI) -> str:
    """C0/C1/C1cap answer: generate from retrieved text/caption chunks."""
    context = "\n\n".join(f"{_tag(d)} {d.page_content}" for d in docs)
    return llm.invoke(ANSWER_PROMPT.format(question=question, context=context)).content


def answer_from_images(question: str, image_paths: list[str], client: OpenAI, max_images: int = 1) -> str:
    """C2b answer: gpt-4o-mini reads the CLIP-retrieved page image(s).

    max_images=1 (TPM constraint; recorded in results meta): each high-detail
    page is ~26k tokens on gpt-4o-mini, so we pace and send only top-1.
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
# Metric layer 3: strict LLM judge (E4/D4)
# ---------------------------------------------------------------------------


JUDGE_PROMPT = """당신은 엄격한 답변 채점자입니다. 모범답안의 핵심 사실이 **모두 정확히** 모델답변에 담겨 있어야만 1점입니다.

질문: {question}
모범답안: {ground_truth}
모델답변: {answer}

채점 기준 (부분 일치 = 0점):
- 핵심 사실 = 위치(수식어 포함 — 모범답안이 "앞면 상단"이면 "상단"까지 있어야 함), 방향, 모양(빗금·사선 등 세부 포함), 수치, 부품명.
- 핵심 사실 중 하나라도 빠지거나 틀리면 0.
- 모델답변이 모범답안보다 덜 구체적이면 0.
- 답변을 거부·회피했으면 0.

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


def _base_row(q: EvalQuestion, pages: list[tuple[str, str, int, str]], has_captions: bool) -> dict:
    ref = reference_target(q)
    row = {
        "id": q.id,
        "retrieved": [[c, x, p, m] for c, x, p, m in pages],
        "retrieval_page_hit": retrieval_page_hit(pages, ref, PAGE_TOLERANCE) if ref else None,
        "retrieval_page_hit_strict": retrieval_page_hit(pages, ref, 0) if ref else None,
    }
    if has_captions:
        row["caption_hit"] = caption_hit(pages, ref, PAGE_TOLERANCE) if ref else None
        row["caption_hit_strict"] = caption_hit(pages, ref, 0) if ref else None
    else:
        row["caption_hit"] = None
        row["caption_hit_strict"] = None
    return row


def _run_text_config(questions, retrieve_fn, answer_llm, judge_llm, has_captions: bool,
                     ckpt: Path | None = None) -> list[dict]:
    """C0 / C1 / C1cap: retrieve chunks → answer → citation → judge.

    ``ckpt`` (jsonl) is appended after every question and consulted on start,
    so a killed process resumes instead of redoing paid work.
    """
    done: dict[str, dict] = {}
    if ckpt is not None and ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
        if done:
            print(f"    resuming {len(done)} rows from {ckpt}", flush=True)

    rows = []
    for q in questions:
        if q.id in done:
            rows.append(done[q.id])
            continue
        docs = retrieve_fn(q.question)
        pages = docs_to_pages(docs)
        answer = answer_from_text(q.question, docs, answer_llm)
        score, reason = judge_answer(q.question, q.reference or "", answer, judge_llm)
        row = _base_row(q, pages, has_captions)
        row.update({
            "answer": answer,
            "citation": citation_match(q, answer),
            "judge": score,
            "judge_reason": reason,
        })
        rows.append(row)
        if ckpt is not None:
            with ckpt.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"    {q.id} done", flush=True)
    return rows


def _run_clip_configs(questions, clip_index, vision_client, judge_llm) -> tuple[list[dict], list[dict]]:
    """C2a (retrieval only) and C2b (retrieval + VLM read) in one CLIP pass."""
    c2a_rows, c2b_rows = [], []
    for q in questions:
        hits = clip_retrieve(clip_index, q.question, k=TOP_K)
        pages = [(h["category"], h["complexity"], h["page"], "page-image") for h in hits]

        # C2a: retrieval only. answer_correct is n/a BY DESIGN (E3) — there is
        # no reading step, so we do not report a fake 0.
        row_a = _base_row(q, pages, has_captions=False)
        row_a.update({"answer": None, "citation": None, "judge": None,
                      "judge_reason": "n/a — retrieval-only config has no reading step"})
        c2a_rows.append(row_a)

        # C2b: VLM reads the top-1 retrieved page image (late fusion).
        answer = answer_from_images(q.question, [h["image_path"] for h in hits], vision_client)
        score, reason = judge_answer(q.question, q.reference or "", answer, judge_llm)
        row_b = _base_row(q, pages, has_captions=False)
        row_b.update({"answer": answer, "citation": None, "judge": score, "judge_reason": reason})
        c2b_rows.append(row_b)
        print(f"    {q.id} done", flush=True)
    return c2a_rows, c2b_rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _ratio(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return sum(1 for v in vals if v) / len(vals) if vals else None


def _summarize(rows: list[dict], ir8_ids: set[str]) -> dict:
    def split(predicate):
        sel = [r for r in rows if predicate(r["id"])]
        judged = [r for r in sel if r["judge"] is not None]
        cited = [r for r in sel if r.get("citation") is not None]
        hit_subset = [r for r in judged if r["retrieval_page_hit"]]
        return {
            "n": len(sel),
            "retrieval_page_hit": _ratio(sel, "retrieval_page_hit"),
            "retrieval_page_hit_strict": _ratio(sel, "retrieval_page_hit_strict"),
            "caption_hit": _ratio(sel, "caption_hit"),
            "caption_hit_strict": _ratio(sel, "caption_hit_strict"),
            "citation_model": (sum(1 for r in cited if r["citation"]["model"]) / len(cited)) if cited else None,
            "citation_page": (sum(1 for r in cited if r["citation"]["page"]) / len(cited)) if cited else None,
            "answer_correct": (sum(r["judge"] for r in judged) / len(judged)) if judged else None,
            # D7: late-fusion effect is only interpretable where retrieval found
            # the right page — off-page images test nothing.
            "answer_correct_on_pagehit": (
                sum(r["judge"] for r in hit_subset) / len(hit_subset)
            ) if hit_subset else None,
        }
    return {
        "ir8": split(lambda i: i in ir8_ids),
        "text_only": split(lambda i: i not in ir8_ids),
        "per_q": rows,
    }


CONFIG_NAMES = ("C0", "C1", "C1cap", "C2a", "C2b")


def _rows_path(cfg: str) -> Path:
    return Path(f"data/week9_rows_{cfg}.json")


def _load_questions(golden_path: Path = Path("data/eval/golden_set_v2.csv")) -> list[EvalQuestion]:
    all_q = load_golden_set(golden_path)
    by_id = {q.id: q for q in all_q}

    # E2: a requested id that is missing from the golden set is a test-set
    # bug (this silently shrank the run-1 contrast set to n=3) — fail loudly.
    requested = list(IR8_IDS) + list(TEXT_ONLY_CONTRAST_IDS)
    missing = [i for i in requested if i not in by_id]
    if missing:
        raise ValueError(f"golden set {golden_path} is missing requested ids: {missing}")
    return [by_id[i] for i in requested]


def run_stage(
    stage: str,
    golden_path: Path = Path("data/eval/golden_set_v2.csv"),
    mm_dir: Path = Path("data/chroma_db_mm"),
    c0_dir: Path = Path("data/chroma_db_c3"),
    clip_dir: Path = Path("data/clip_index"),
) -> None:
    """Run one config family and checkpoint its rows to data/week9_rows_*.json.

    One stage per PROCESS: the previous single-process run (reranker + two
    Chroma stores + two BM25 lanes resident at once) was killed ~1h in with
    no traceback — a memory-pressure signature on a 16GB host. Staging keeps
    peak memory low and a crash never loses finished configs.
    """
    questions = _load_questions(golden_path)
    print(f"Test set: {len(questions)} questions ({len(IR8_IDS)} IR8 + "
          f"{len(TEXT_ONLY_CONTRAST_IDS)} text-only contrast)", flush=True)

    answer_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
    judge_llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
    out: dict[str, list[dict]] = {}

    if stage == "C0":
        from .vectorstore import load_vectorstore
        print("Loading reranker...", flush=True)
        reranker = create_reranker()
        print("C0: text-only RAG (chroma_db_c3)...", flush=True)
        store = load_vectorstore(c0_dir, collection_name="lg_manuals_c3")
        out["C0"] = _run_text_config(
            questions, make_text_retriever(store, reranker), answer_llm, judge_llm,
            has_captions=False, ckpt=Path("data/week9_rows_C0.jsonl.part"),
        )
    elif stage in ("C1", "C1cap"):
        print("Loading reranker...", flush=True)
        reranker = create_reranker()
        mm_store = load_mm_store(mm_dir)
        mm_retriever = ModalityAwareRetriever(mm_store, reranker)
        if stage == "C1":
            print("C1: modality-aware mm RAG (text 3 + caption 2)...", flush=True)
            retrieve_fn = mm_retriever.invoke
        else:
            print("C1cap: caption lane top-5...", flush=True)
            retrieve_fn = lambda q: mm_retriever.retrieve_captions(q, k=TOP_K)
        out[stage] = _run_text_config(
            questions, retrieve_fn, answer_llm, judge_llm, has_captions=True,
            ckpt=Path(f"data/week9_rows_{stage}.jsonl.part"),
        )
    elif stage == "C2":
        print("C2a/C2b: CLIP retrieval (+VLM read)...", flush=True)
        vision_client = OpenAI(max_retries=8)
        clip_index = ClipIndex.load(clip_dir)
        out["C2a"], out["C2b"] = _run_clip_configs(questions, clip_index, vision_client, judge_llm)
    else:
        raise ValueError(f"unknown stage {stage!r} (use C0 | C1 | C1cap | C2)")

    for name, rows in out.items():
        path = _rows_path(name)
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"wrote {path}", flush=True)


def merge_results(
    golden_path: Path = Path("data/eval/golden_set_v2.csv"),
    out_path: Path = Path("data/week9_results.json"),
) -> dict:
    """Assemble data/week9_results.json from the per-config row checkpoints."""
    questions = _load_questions(golden_path)
    ir8_set = set(IR8_IDS)

    results = {
        "configs": list(CONFIG_NAMES),
        "page_tolerance": PAGE_TOLERANCE,
        "top_k": TOP_K,
        "answer_model": ANSWER_MODEL,
        "judge_model": JUDGE_MODEL,
        "c2b_max_images": 1,
        "metric_layers_note": (
            "retrieval_page_hit is a RETRIEVAL-layer metric; citation_model/"
            "citation_page are the Week7 CITATION-layer metrics (ADR-011 anchor). "
            "Do not compare across layers."
        ),
        "questions": [{
            "id": q.id,
            "question": q.question,
            "modality": q.modality_label,
            "ground_truth": q.reference,
            "reference_context": q.reference_context,
        } for q in questions],
        "results": {},
    }
    for cfg in CONFIG_NAMES:
        rows = json.loads(_rows_path(cfg).read_text())
        results["results"][cfg] = _summarize(rows, ir8_set)

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")
    return results


def run_comparison(**kwargs) -> dict:
    """Legacy one-call entry (notebook): all stages in-process, then merge.

    Prefer the staged CLI on memory-constrained hosts:
        python -m src.week9_eval C0 && python -m src.week9_eval C1 \
            && python -m src.week9_eval C2 && python -m src.week9_eval merge
    """
    for stage in ("C0", "C1", "C1cap", "C2"):
        run_stage(stage)
    return merge_results()


def print_summary(results: dict) -> None:
    def fmt(v):
        return f"{v:.0%}" if isinstance(v, (int, float)) else "n/a"
    print("\n| Config | IR8 ret-hit | IR8 cap-hit | IR8 cite p | IR8 answer | IR8 ans@hit | TO ret-hit | TO answer |")
    print("|--------|------------|-------------|-----------|------------|-------------|-----------|-----------|")
    for cfg in results["configs"]:
        r = results["results"][cfg]
        i, t = r["ir8"], r["text_only"]
        print(f"| {cfg} | {fmt(i['retrieval_page_hit'])} | {fmt(i['caption_hit'])} | "
              f"{fmt(i['citation_page'])} | {fmt(i['answer_correct'])} | {fmt(i['answer_correct_on_pagehit'])} | "
              f"{fmt(t['retrieval_page_hit'])} | {fmt(t['answer_correct'])} |")


if __name__ == "__main__":
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "merge":
        print_summary(merge_results())
    elif stage == "all":
        print_summary(run_comparison())
    else:
        run_stage(stage)
