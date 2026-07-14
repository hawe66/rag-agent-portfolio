"""
Week 11 multimodal agent (WEEK11_TASKS §3).

LangGraph agent that composes the four tools in `src/agent_tools.py` and
answers "with fallbacks" instead of failing silently:

    entry ─(input_type)─┬─ image/pdf → ocr ──┬─(conf ok)──────→ rag
                        │                    └─(F1: low conf)→ vision_input → rag
                        └─ text ────────────────────────────→ rag
    rag ──(ok)→ grade ──┬─(sufficient)──────────────────────→ generate
        └─(fail)→ refuse└─(F2: insufficient, first time)───→ vision_escalate → generate
                         └─(insufficient, exhausted)────────→ refuse (F3)
    generate ──┬─(grounded)──────────────────────────────────→ END
               ├─(F2: ungrounded, not yet escalated)────────→ vision_escalate → generate
               └─(ungrounded, escalation exhausted)─────────→ refuse (F3)

Fallbacks (§3.3):
    F1  OCR failed / low confidence  → query-side Vision read of the input image
        (domain truth: vector line-art defeats OCR — LIM-002).
    F2  Evidence insufficient → Vision reads the *retrieved* page image
        directly. This is the query-side bypass of the Week 9 bottleneck
        (full-page captions can't carry icon shapes / positions). Two
        triggers: grade(not_relevant), OR an ungrounded generate — the first
        scenario run showed the grader alone is too lenient (it passed
        caption context that the generator then refused on; same grader
        limitation Week 9 §6 documented), so the generator's refusal is the
        robust trigger.
    F3  Still ungrounded after escalation → explicit refusal, never a
        fabricated answer.

Every node appends to ``route_history``; every fallback appends to
``fallback_history`` — those two lists are the evidence for the §4 scenario
table and §5 architecture doc.

Reuses Week 6's ``GRADE_PROMPT`` / ``REFUSAL_MESSAGE`` (`src/agent.py`) so the
grounding contract stays identical across the two agents.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, List, Optional

from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from .agent import GRADE_PROMPT, REFUSAL_MESSAGE

OCR_CONFIDENCE_THRESHOLD = 0.5  # below this, F1 escalates OCR → Vision
# Confidence alone is a weak trigger: near-blank diagram pages OCR at conf
# ~0.9 with 2 recognized chars (measured across all 252 pages). Require a
# minimum of extracted text as well.
MIN_OCR_CHARS = 30
IMAGE_ROOT = Path("data/sample_images_150")  # 150dpi pages (Week 9 v2 캡션과 동일)
QUERY_SNIPPET_CHARS = 300  # OCR/vision text appended to the retrieval query


class MMAgentState(TypedDict, total=False):
    """Shared state (WEEK11_TASKS §3.1)."""

    question: str
    input_type: str  # "text" | "image" | "pdf"
    image_path: Optional[str]
    # tool intermediate results
    ocr_text: str
    image_summary: str
    docs: List[Any]
    scores: List[float]
    caption_hit: bool
    # grounding
    evidence: str
    is_grounded: bool
    confidence: float
    grade_result: str  # "relevant" | "not_relevant"
    vision_escalated: bool
    # output
    answer: str
    refused: bool
    # flow records (basis of the §4 scenario table)
    route_history: List[str]
    fallback_history: List[str]
    # internal router flags (routers must not mutate state, so nodes stash
    # tool ok-ness here; stripped from the public result by run_mm_agent)
    _ocr_ok: bool
    _rag_ok: bool


ToolFn = Callable[..., dict]


def _push(state: MMAgentState, key: str, label: str) -> List[str]:
    history = list(state.get(key, []))
    history.append(label)
    return history


def page_image_path(meta: dict, image_root: Path = IMAGE_ROOT) -> Optional[Path]:
    """Map a retrieved chunk's metadata to its rasterized page PNG."""
    category = meta.get("category")
    complexity = meta.get("complexity")
    page = meta.get("page")
    if not (category and complexity and page):
        return None
    path = image_root / f"{category}_{complexity}" / f"p{page:03d}.png"
    return path if path.exists() else None


def _pdf_first_page_png(pdf_path: str) -> str:
    """Render page 1 of a PDF to a sibling PNG so ocr_tool can read it."""
    import pymupdf

    out_path = Path(pdf_path).with_suffix(".p001.png")
    doc = pymupdf.open(pdf_path)
    try:
        doc[0].get_pixmap(dpi=150).save(str(out_path))
    finally:
        doc.close()
    return str(out_path)


def _tag(doc) -> str:
    cat = doc.metadata.get("category", "unknown")
    cplx = doc.metadata.get("complexity", "")
    page = doc.metadata.get("page", "?")
    head = f"{cat}_{cplx}" if cplx else cat
    return f"[{head} p.{page}]"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _build_ocr_node(ocr_tool: ToolFn):
    def ocr_node(state: MMAgentState) -> MMAgentState:
        path = state.get("image_path") or ""
        if state.get("input_type") == "pdf" and path.lower().endswith(".pdf"):
            path = _pdf_first_page_png(path)
        result = ocr_tool(path)
        return {
            **state,
            "image_path": path,
            "ocr_text": result["text"],
            "confidence": result["confidence"],
            "route_history": _push(
                state, "route_history",
                f"ocr(ok={result['ok']}, conf={result['confidence']:.2f}, chars={len(result['text'])})",
            ),
            # stash ok-ness for the router (routers must not mutate state)
            "_ocr_ok": result["ok"],
        }

    return ocr_node


def _ocr_usable(state: MMAgentState) -> bool:
    return (
        bool(state.get("_ocr_ok"))
        and state.get("confidence", 0.0) >= OCR_CONFIDENCE_THRESHOLD
        and len(state.get("ocr_text", "")) >= MIN_OCR_CHARS
    )


def _route_after_ocr(state: MMAgentState) -> str:
    return "rag" if _ocr_usable(state) else "vision_input"


def _build_vision_input_node(image_analysis_tool: ToolFn):
    """F1 target: OCR was unusable → gpt-4o reads the input image directly."""

    def vision_input_node(state: MMAgentState) -> MMAgentState:
        trigger = (
            f"F1: ocr unusable (conf={state.get('confidence', 0.0):.2f}, "
            f"chars={len(state.get('ocr_text', ''))}) → vision_input"
        )
        result = image_analysis_tool(state.get("image_path") or "", state["question"])
        return {
            **state,
            "image_summary": result["image_summary"],
            "confidence": result["confidence"],
            "route_history": _push(
                state, "route_history",
                f"vision_input(ok={result['ok']}, conf={result['confidence']:.2f})",
            ),
            "fallback_history": _push(state, "fallback_history", trigger),
        }

    return vision_input_node


def _build_rag_node(rag_search_tool: ToolFn):
    def rag_node(state: MMAgentState) -> MMAgentState:
        query = state["question"]
        # Image-derived context sharpens retrieval for image/pdf inputs.
        # image_summary is only set when F1 fired (OCR unusable) — prefer it,
        # so leftover junk OCR text never reaches the query.
        if state.get("image_summary"):
            query += f"\n[이미지 분석 요약] {state['image_summary'][:QUERY_SNIPPET_CHARS]}"
        elif len(state.get("ocr_text", "")) >= MIN_OCR_CHARS:
            query += f"\n[이미지 OCR 텍스트] {state['ocr_text'][:QUERY_SNIPPET_CHARS]}"
        result = rag_search_tool(query)
        return {
            **state,
            "docs": result["docs"],
            "scores": result["scores"],
            "caption_hit": result["caption_hit"],
            "route_history": _push(
                state, "route_history",
                f"rag(ok={result['ok']}, k={len(result['docs'])}, caption_hit={result['caption_hit']})",
            ),
            "_rag_ok": result["ok"],
        }

    return rag_node


def _route_after_rag(state: MMAgentState) -> str:
    return "grade" if state.get("_rag_ok") else "refuse"


def _build_grade_node(llm):
    """Evidence sufficiency check — same contract as Week 6's grade node,
    extended to include the vision summary as evidence."""

    def grade_node(state: MMAgentState) -> MMAgentState:
        docs = state.get("docs", [])
        parts = [
            f"[문서 {i + 1}] (category: {doc.metadata.get('category', 'unknown')})\n"
            f"{doc.page_content[:500]}..."
            for i, doc in enumerate(docs[:5])
        ]
        if state.get("image_summary"):
            parts.append(f"[이미지 분석] {state['image_summary']}")
        prompt = GRADE_PROMPT.format(question=state["question"], documents="\n\n".join(parts))
        grade = llm.invoke(prompt).content.strip().lower()
        grade_result = "relevant" if grade == "yes" else "not_relevant"
        return {
            **state,
            "grade_result": grade_result,
            "route_history": _push(state, "route_history", f"grade({grade_result})"),
        }

    return grade_node


def _route_after_grade(state: MMAgentState) -> str:
    if state.get("grade_result") == "relevant":
        return "generate"
    if not state.get("vision_escalated"):
        # Only escalate if we can actually find a page image to read.
        for doc in _escalation_candidates(state):
            if page_image_path(doc.metadata) is not None:
                return "vision_escalate"
    return "refuse"


def _escalation_candidates(state: MMAgentState) -> list:
    """Retrieved docs in escalation-priority order: caption chunks first
    (their page is where the figure lives), then text chunks by rank."""
    docs = state.get("docs", [])
    captions = [d for d in docs if d.metadata.get("modality") == "image-derived"]
    texts = [d for d in docs if d.metadata.get("modality") != "image-derived"]
    return captions + texts


def _build_vision_escalate_node(image_analysis_tool: ToolFn):
    """F2 target: caption RAG evidence insufficient → gpt-4o reads the
    retrieved page image at query time (Week 9 bottleneck bypass)."""

    def vision_escalate_node(state: MMAgentState) -> MMAgentState:
        target_doc = next(
            (d for d in _escalation_candidates(state) if page_image_path(d.metadata)),
            None,
        )
        path = page_image_path(target_doc.metadata) if target_doc else None
        label = _tag(target_doc) if target_doc else "no-page"
        route = state.get("route_history", [])
        came_from_generate = bool(route) and route[-1].startswith("generate")
        cause = (
            "generate(ungrounded)" if came_from_generate
            else f"grade(not_relevant)"
        )
        trigger = (
            f"F2: {cause} (caption_hit={state.get('caption_hit')}) "
            f"→ vision_escalate({label})"
        )
        result = image_analysis_tool(str(path), state["question"]) if path else {
            "image_summary": "", "confidence": 0.0, "ok": False, "error": "no page image",
        }
        merged = "\n".join(
            s for s in (state.get("image_summary", ""), result["image_summary"]) if s
        )
        return {
            **state,
            "image_summary": merged,
            "confidence": result["confidence"],
            "vision_escalated": True,
            "route_history": _push(
                state, "route_history",
                f"vision_escalate({label}, ok={result['ok']}, conf={result['confidence']:.2f})",
            ),
            "fallback_history": _push(state, "fallback_history", trigger),
        }

    return vision_escalate_node


def _build_generate_node(answer_generation_tool: ToolFn):
    def generate_node(state: MMAgentState) -> MMAgentState:
        context = "\n\n".join(f"{_tag(d)} {d.page_content}" for d in state.get("docs", []))
        if state.get("image_summary"):
            context += f"\n\n[이미지 분석 — 검색된 페이지 도면을 직접 읽은 결과] {state['image_summary']}"
        result = answer_generation_tool(context, state["question"])
        # Refusal is decided by the router (a first ungrounded answer may
        # still be rescued by vision escalation) — not here.
        return {
            **state,
            "evidence": context,
            "answer": result["answer"] or REFUSAL_MESSAGE,
            "is_grounded": bool(result["ok"] and result["is_grounded"]),
            "route_history": _push(
                state, "route_history",
                f"generate(ok={result['ok']}, grounded={result['is_grounded']})",
            ),
        }

    return generate_node


def _route_after_generate(state: MMAgentState) -> str:
    if state.get("is_grounded"):
        return "end"
    if not state.get("vision_escalated"):
        for doc in _escalation_candidates(state):
            if page_image_path(doc.metadata) is not None:
                return "vision_escalate"
    return "refuse"


def _refuse_node(state: MMAgentState) -> MMAgentState:
    return {
        **state,
        "answer": REFUSAL_MESSAGE,
        "is_grounded": False,
        "refused": True,
        "route_history": _push(state, "route_history", "refuse"),
        "fallback_history": _push(
            state, "fallback_history",
            "F3: ungrounded / insufficient evidence (escalation exhausted) → refusal",
        ),
    }


def _route_entry(state: MMAgentState) -> str:
    return "ocr" if state.get("input_type") in ("image", "pdf") else "rag"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_mm_agent(
    ocr_tool: ToolFn,
    image_analysis_tool: ToolFn,
    rag_search_tool: ToolFn,
    answer_generation_tool: ToolFn,
    llm,
):
    """Compile the multimodal agent graph.

    ``llm`` powers only the grade (evidence-sufficiency) node; answering goes
    through ``answer_generation_tool`` so the refusal contract stays in one place.
    """
    workflow = StateGraph(MMAgentState)

    workflow.add_node("ocr", _build_ocr_node(ocr_tool))
    workflow.add_node("vision_input", _build_vision_input_node(image_analysis_tool))
    workflow.add_node("rag", _build_rag_node(rag_search_tool))
    workflow.add_node("grade", _build_grade_node(llm))
    workflow.add_node("vision_escalate", _build_vision_escalate_node(image_analysis_tool))
    workflow.add_node("generate", _build_generate_node(answer_generation_tool))
    workflow.add_node("refuse", _refuse_node)

    workflow.set_conditional_entry_point(_route_entry, {"ocr": "ocr", "rag": "rag"})
    workflow.add_conditional_edges(
        "ocr", _route_after_ocr, {"rag": "rag", "vision_input": "vision_input"}
    )
    workflow.add_edge("vision_input", "rag")
    workflow.add_conditional_edges(
        "rag", _route_after_rag, {"grade": "grade", "refuse": "refuse"}
    )
    workflow.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"generate": "generate", "vision_escalate": "vision_escalate", "refuse": "refuse"},
    )
    workflow.add_edge("vision_escalate", "generate")
    workflow.add_conditional_edges(
        "generate",
        _route_after_generate,
        {"end": END, "vision_escalate": "vision_escalate", "refuse": "refuse"},
    )
    workflow.add_edge("refuse", END)

    return workflow.compile()


def run_mm_agent(
    graph,
    question: str,
    input_type: str = "text",
    image_path: str | None = None,
) -> dict:
    """Run one query through the agent; returns final state + total_latency."""
    initial: MMAgentState = {
        "question": question,
        "input_type": input_type,
        "image_path": image_path,
        "ocr_text": "",
        "image_summary": "",
        "docs": [],
        "scores": [],
        "caption_hit": False,
        "evidence": "",
        "is_grounded": False,
        "confidence": 0.0,
        "grade_result": "",
        "vision_escalated": False,
        "answer": "",
        "refused": False,
        "route_history": [],
        "fallback_history": [],
    }
    start = time.time()
    result = dict(graph.invoke(initial))
    result["total_latency"] = time.time() - start
    # internal router flags — not part of the public state contract
    result.pop("_ocr_ok", None)
    result.pop("_rag_ok", None)
    return result
