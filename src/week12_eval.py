"""Week 12 three-way cross-modal comparison (WEEK12_TASKS §4).

Arms — same question set, same answer model, same context budget (5 chunks):

    C0        text-only RAG          (chroma_db_c3,     flat top-5)
    C1page    text + PAGE captions   (chroma_db_mm_w12, text 3 + caption 2)
    C1region  text + REGION captions (chroma_db_mm_w12, text 3 + caption 2)

C1page and C1region are two `caption_scope` filters over ONE store, so the
arms differ only in which caption family they may draw from — the text lane is
byte-identical between them.

Deliberately NO automatic answer metric. Week 9 showed a strict LLM judge
scoring correct answers as 0 because of golden-set defects (D6), and §4 of the
task doc rules out leaning on automatic metrics. This module produces answers
plus the retrieval-layer lenses, then emits a grading sheet; L3/L4/L5 are
filled in by a human reading the answer against the manual page, and merged
back via `data/week12_scores.json`.

Six lenses (§4.1):
    L1 검색 도달   page_hit / manual_hit      — was it a retrieval failure?
    L2 근거 도달   caption_hit                — did figure info reach the context?
    L3 답변 정확도 manual, 4-level            — from week12_scores.json
    L4 거절 품질   refused + manual verdict   — from week12_scores.json
    L5 실패 원인   F-RET/CAP/CROP/GEN/GT      — from week12_scores.json
    L6 비용·지연   retrieve_s/answer_s/calls  — measured here

Run:
    .venv/bin/python -m src.week12_eval C0
    .venv/bin/python -m src.week12_eval C1page
    .venv/bin/python -m src.week12_eval C1region
    .venv/bin/python -m src.week12_eval sheet     # grading sheet for L3-L5
    .venv/bin/python -m src.week12_eval merge     # -> data/week12_results.json
"""

import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from .evaluation import EvalQuestion, load_golden_set
from .mm_eval_v1 import (
    ANSWER_MODEL,
    PAGE_TOLERANCE,
    TOP_K,
    answer_from_text,
    caption_hit,
    docs_to_pages,
    make_text_retriever,
    reference_target,
    retrieval_page_hit,
)

GOLDEN = Path("data/eval/golden_set_v4.csv")
MM_DIR = Path("data/chroma_db_mm_w12")
C0_DIR = Path("data/chroma_db_c3")
RESULTS = Path("data/week12_results.json")
SHEET = Path("data/week12_grading_sheet.md")
SCORES = Path("data/week12_scores.json")

CONFIGS = ("C0", "C1page", "C1region")

# Text-only contrast (§4): the same 6 ids Week 9 used, so a regression in the
# text lane is visible. Q08 stays in deliberately — it is a known golden-set
# defect and dropping it would flatter every arm equally.
CONTRAST_IDS = ("Q01", "Q06", "Q08", "Q11", "Q18", "Q19")

REFUSAL_MARKERS = ("확인할 수 없습니다", "확인할 수 없", "제공된 문서에서")


def _rows_path(config: str) -> Path:
    return Path(f"data/week12_rows_{config}.json")


def load_questions(golden: Path = GOLDEN) -> tuple[list[EvalQuestion], dict[str, str]]:
    """Image-required rows + the text-only contrast set, plus id -> ir_type."""
    questions = load_golden_set(golden)
    by_id = {q.id: q for q in questions}

    missing = [i for i in CONTRAST_IDS if i not in by_id]
    if missing:
        raise ValueError(f"{golden} is missing contrast ids: {missing}")

    with open(golden, newline="", encoding="utf-8") as f:
        ir_types = {}
        for index, row in enumerate(csv.DictReader(f)):
            ir_types[questions[index].id] = (row.get("ir_type") or "").strip()

    selected = [q for q in questions if q.modality_label == "image-required"]
    selected += [by_id[i] for i in CONTRAST_IDS]
    return selected, ir_types


def is_refusal(answer: str) -> bool:
    return any(marker in answer for marker in REFUSAL_MARKERS)


def manual_hit(retrieved: list[tuple[str, str, int, str]], ref) -> bool:
    """Did anything at all come from the right manual (category+complexity)?

    Separated from page_hit because the simple/complex pairs (IR-W1 vs IR-W6)
    fail differently: wrong manual entirely vs right manual, wrong page.
    """
    return any(
        category == ref.category and complexity == ref.complexity
        for category, complexity, _page, _modality in retrieved
    )


def run_config(config: str) -> list[dict]:
    """Retrieve + answer every question for one arm, checkpointing as it goes.

    One arm per PROCESS (same reason as Week 9's staged runner: reranker plus
    a resident Chroma store plus BM25 lanes is what got the single-process run
    killed on a 16GB host).
    """
    from .retrieval import create_reranker

    questions, ir_types = load_questions()
    print(f"Test set: {len(questions)} questions "
          f"({len(questions) - len(CONTRAST_IDS)} image-required "
          f"+ {len(CONTRAST_IDS)} text-only contrast)", flush=True)

    print("Loading reranker...", flush=True)
    reranker = create_reranker()

    if config == "C0":
        from .vectorstore import load_vectorstore

        store = load_vectorstore(C0_DIR, collection_name="lg_manuals_c3")
        retrieve = make_text_retriever(store, reranker)
        has_captions = False
    elif config in ("C1page", "C1region"):
        from .mm_retrieval import MMRetrieverConfig, ModalityAwareRetriever
        from .multimodal import load_mm_store

        scope = "page" if config == "C1page" else "region"
        retriever = ModalityAwareRetriever(
            load_mm_store(MM_DIR), reranker, MMRetrieverConfig(caption_scope=scope)
        )
        retrieve = retriever.invoke
        has_captions = True
    else:
        raise ValueError(f"unknown config {config!r} (use {' | '.join(CONFIGS)})")

    answer_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)

    checkpoint = Path(f"data/week12_rows_{config}.jsonl.part")
    done: dict[str, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
        print(f"  resuming {len(done)} rows from {checkpoint}", flush=True)

    rows: list[dict] = []
    for question in questions:
        if question.id in done:
            rows.append(done[question.id])
            continue

        started = time.monotonic()
        docs: list[Document] = retrieve(question.question)
        retrieved_at = time.monotonic()
        answer = answer_from_text(question.question, docs, answer_llm)
        finished = time.monotonic()

        pages = docs_to_pages(docs)
        ref = reference_target(question)
        row = {
            "id": question.id,
            "ir_type": ir_types.get(question.id, ""),
            "modality": question.modality_label,
            "question": question.question,
            "ground_truth": question.reference,
            "reference_context": question.reference_context,
            "retrieved": [list(p) for p in pages],
            "manual_hit": manual_hit(pages, ref) if ref else None,
            "page_hit": retrieval_page_hit(pages, ref, PAGE_TOLERANCE) if ref else None,
            "page_hit_strict": retrieval_page_hit(pages, ref, 0) if ref else None,
            "caption_hit": (caption_hit(pages, ref, PAGE_TOLERANCE)
                            if (ref and has_captions) else None),
            "caption_hit_strict": (caption_hit(pages, ref, 0)
                                   if (ref and has_captions) else None),
            "context": [
                {"tag": f"{d.metadata.get('category')}_{d.metadata.get('complexity')} "
                        f"p.{d.metadata.get('page')}",
                 "modality": d.metadata.get("modality"),
                 "caption_scope": d.metadata.get("caption_scope"),
                 "text": d.page_content}
                for d in docs
            ],
            "answer": answer,
            "refused": is_refusal(answer),
            "retrieve_s": round(retrieved_at - started, 2),
            "answer_s": round(finished - retrieved_at, 2),
            "llm_calls": 1,  # 1 generation; embedding + rerank are not LLM calls
        }
        rows.append(row)
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"    {question.id} done ({row['retrieve_s']}s + {row['answer_s']}s)", flush=True)

    path = _rows_path(config)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"wrote {path} ({len(rows)} rows)")
    return rows


def write_grading_sheet(path: Path = SHEET) -> None:
    """Emit every question's three answers side by side for manual L3-L5.

    Side by side rather than one arm at a time: scoring a whole arm in a block
    invites the standard drift between blocks.
    """
    per_config = {c: {r["id"]: r for r in json.loads(_rows_path(c).read_text())}
                  for c in CONFIGS}
    questions, _ = load_questions()

    lines = [
        "# Week 12 grading sheet (L3 답변 정확도 / L4 거절 품질 / L5 실패 원인)",
        "",
        "채점 기준: GT 문자열 일치가 아니라 **매뉴얼 충실성**. 4단계 = 정답 / 부분 / 오답 / 거절.",
        "L5 코드: F-RET(검색실패) F-CAP(페이지캡션에 정보없음) F-CROP(영역캡션에도 없음) "
        "F-GEN(근거는 있었으나 생성실패) F-GT(정답지 결함)",
        "",
    ]
    for question in questions:
        first = per_config["C0"][question.id]
        lines += [
            f"## {question.id} · {first['ir_type'] or first['modality']}",
            f"- **Q**: {question.question}",
            f"- **GT**: {question.reference}",
            f"- **ref**: {question.reference_context}",
            "",
        ]
        for config in CONFIGS:
            row = per_config[config][question.id]
            flags = (f"manual_hit={row['manual_hit']} page_hit={row['page_hit']} "
                     f"caption_hit={row['caption_hit']} refused={row['refused']}")
            lines += [f"### {config} ({flags})", "", "근거 컨텍스트:"]
            lines += [
                f"  - `{c['tag']}` [{c['modality']}/{c['caption_scope']}] "
                f"{c['text'][:160].replace(chr(10), ' ')}"
                for c in row["context"]
            ]
            lines += ["", f"답변: {row['answer']}", ""]
        lines.append("---\n")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path} ({len(questions)} questions x {len(CONFIGS)} arms)")


def _rate(rows: list[dict], key: str) -> dict | None:
    """{'n': int, 'k': int, 'rate': float} — n is always carried so every claim
    can be written with its denominator (§4.4)."""
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    hits = sum(1 for v in values if v)
    return {"n": len(values), "k": hits, "rate": round(hits / len(values), 3)}


def _score_block(rows: list[dict], scores: dict, config: str) -> dict:
    """L3/L4/L5 aggregation for one slice from the manual scores file."""
    graded = [(r, scores[r["id"]][config]) for r in rows if r["id"] in scores]
    if not graded:
        return {"n": 0}
    tally: dict[str, int] = {}
    causes: dict[str, int] = {}
    for _row, entry in graded:
        tally[entry["score"]] = tally.get(entry["score"], 0) + 1
        if entry.get("cause"):
            causes[entry["cause"]] = causes.get(entry["cause"], 0) + 1
    n = len(graded)
    return {
        "n": n,
        "정답": tally.get("정답", 0),
        "부분": tally.get("부분", 0),
        "오답": tally.get("오답", 0),
        "거절": tally.get("거절", 0),
        "정답_rate": round(tally.get("정답", 0) / n, 3),
        "정답+부분_rate": round((tally.get("정답", 0) + tally.get("부분", 0)) / n, 3),
        "refusal_appropriate": sum(1 for _r, e in graded if e.get("refusal_ok") is True),
        "refusal_inappropriate": sum(1 for _r, e in graded if e.get("refusal_ok") is False),
        "failure_causes": dict(sorted(causes.items())),
    }


def _slice(rows: list[dict], scores: dict, config: str) -> dict:
    return {
        "n": len(rows),
        "L1_manual_hit": _rate(rows, "manual_hit"),
        "L1_page_hit": _rate(rows, "page_hit"),
        "L1_page_hit_strict": _rate(rows, "page_hit_strict"),
        "L2_caption_hit": _rate(rows, "caption_hit"),
        "L2_caption_hit_strict": _rate(rows, "caption_hit_strict"),
        "L3_L4_L5": _score_block(rows, scores, config),
        "L6_latency_s": (
            round(sum(r["retrieve_s"] + r["answer_s"] for r in rows) / len(rows), 2)
            if rows else None
        ),
        "L6_retrieve_s": (
            round(sum(r["retrieve_s"] for r in rows) / len(rows), 2) if rows else None
        ),
        "L6_llm_calls_per_q": (
            round(sum(r["llm_calls"] for r in rows) / len(rows), 2) if rows else None
        ),
    }


def merge_results(out: Path = RESULTS) -> dict:
    """Assemble data/week12_results.json from the per-arm rows + manual scores."""
    scores = json.loads(SCORES.read_text()) if SCORES.exists() else {}
    if not scores:
        print(f"WARNING: {SCORES} missing — L3/L4/L5 will be empty.")

    results: dict = {
        "configs": list(CONFIGS),
        "golden_set": str(GOLDEN),
        "mm_store": str(MM_DIR),
        "answer_model": ANSWER_MODEL,
        "top_k": TOP_K,
        "page_tolerance": PAGE_TOLERANCE,
        "grading": "manual (4-level 정답/부분/오답/거절); no LLM judge, no RAGAS",
        "results": {},
    }
    for config in CONFIGS:
        rows = json.loads(_rows_path(config).read_text())
        image_required = [r for r in rows if r["modality"] == "image-required"]
        results["results"][config] = {
            "image_required": _slice(image_required, scores, config),
            "text_only_contrast": _slice(
                [r for r in rows if r["modality"] != "image-required"], scores, config
            ),
            "by_ir_type": {
                ir_type: _slice(
                    [r for r in image_required if r["ir_type"] == ir_type], scores, config
                )
                for ir_type in sorted({r["ir_type"] for r in image_required if r["ir_type"]})
            },
            "by_complexity": {
                complexity: _slice(
                    [r for r in image_required
                     if r["reference_context"].split(" ")[0].endswith(complexity)],
                    scores, config,
                )
                for complexity in ("simple", "complex")
            },
            "per_q": rows,
        }

    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return results


def print_summary(results: dict) -> None:
    def pct(block):
        return f"{block['rate']:.0%} ({block['k']}/{block['n']})" if block else "n/a"

    print("\n### image-required")
    print("| arm | L1 manual | L1 page | L2 caption | L3 정답 | L3 정답+부분 | L6 s/q |")
    print("|---|---|---|---|---|---|---|")
    for config in results["configs"]:
        s = results["results"][config]["image_required"]
        g = s["L3_L4_L5"]
        correct = f"{g['정답']}/{g['n']}" if g.get("n") else "n/a"
        partial = f"{g['정답'] + g['부분']}/{g['n']}" if g.get("n") else "n/a"
        print(f"| {config} | {pct(s['L1_manual_hit'])} | {pct(s['L1_page_hit'])} | "
              f"{pct(s['L2_caption_hit'])} | {correct} | {partial} | {s['L6_latency_s']} |")


if __name__ == "__main__":
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if stage == "sheet":
        write_grading_sheet()
    elif stage == "merge":
        print_summary(merge_results())
    else:
        run_config(stage)
