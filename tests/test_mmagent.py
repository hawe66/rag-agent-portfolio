"""
Week 11 offline verification (WEEK11_TASKS §9).

Covers what can be checked without network access:
- tool output schemas and explicit failure signals (``ok=False``, never raise),
- agent graph routing: happy paths, F1/F2/F3 fallback triggers, refusal.

API-dependent behavior (gpt-4o vision content, live retrieval quality) is
exercised by ``python -m w11.week11_scenarios`` and recorded in
``data/week11_scenarios.json``.

Run:  .venv/bin/pytest tests/test_week11_agent.py -q
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.agent_tools import (
    image_analysis_tool,
    make_answer_generation_tool,
    make_rag_search_tool,
    ocr_tool,
)
from src.mm_agent import build_mm_agent, run_mm_agent

OCR_DEAD_PAGE = Path("data/sample_images_150/waterpurifier_simple/p008.png")


# ---------------------------------------------------------------------------
# Tool schemas + failure signals
# ---------------------------------------------------------------------------


def test_ocr_tool_returns_ok_false_on_missing_file():
    result = ocr_tool("data/does_not_exist.png")
    assert result["ok"] is False
    assert result["text"] == ""
    assert result["confidence"] == 0.0
    assert "error" in result


@pytest.mark.skipif(not OCR_DEAD_PAGE.exists(), reason="rasterized pages not present")
def test_ocr_tool_reports_empty_page_as_result_not_error():
    result = ocr_tool(str(OCR_DEAD_PAGE))
    assert result["ok"] is True  # OCR ran; finding nothing is a result
    assert result["confidence"] == 0.0
    assert result["text"] == ""


def test_image_analysis_tool_fails_before_api_call_on_missing_file():
    result = image_analysis_tool("data/does_not_exist.png", "질문")
    assert result["ok"] is False
    assert result["image_summary"] == ""
    assert "error" in result


def test_rag_search_tool_signals_retriever_failure():
    class BrokenRetriever:
        def invoke(self, query):
            raise RuntimeError("store unavailable")

    result = make_rag_search_tool(BrokenRetriever(), reranker=None)("query")
    assert result == {
        "docs": [], "scores": [], "caption_hit": False,
        "ok": False, "error": "store unavailable",
    }


def test_answer_generation_tool_signals_llm_failure_as_ungrounded():
    class BrokenLLM:
        def invoke(self, prompt):
            raise RuntimeError("api down")

    result = make_answer_generation_tool(BrokenLLM())("컨텍스트", "질문")
    assert result["ok"] is False
    assert result["is_grounded"] is False
    assert result["answer"] == ""


def test_answer_generation_tool_flags_refusal_as_ungrounded():
    class RefusingLLM:
        def invoke(self, prompt):
            class Response:
                content = "제공된 문서에서 확인할 수 없습니다."
            return Response()

    result = make_answer_generation_tool(RefusingLLM())("컨텍스트", "질문")
    assert result["ok"] is True
    assert result["is_grounded"] is False


# ---------------------------------------------------------------------------
# Graph routing + fallback triggers (stub tools, no network)
# ---------------------------------------------------------------------------


REAL_PAGE_DOC = Document(
    page_content="caption chunk",
    metadata={"category": "airpurifier", "complexity": "complex",
              "page": 18, "modality": "image-derived"},
)


def _make_graph(ocr_conf, ocr_text, grade, grounded, rag_ok=True):
    def ocr(path):
        return {"text": ocr_text, "confidence": ocr_conf, "ok": True}

    def vision(path, question, **kwargs):
        return {"image_summary": "vision-summary", "confidence": 0.8, "ok": True}

    def rag(query):
        docs = [REAL_PAGE_DOC] if rag_ok else []
        return {"docs": docs, "scores": [0.9] * len(docs),
                "caption_hit": rag_ok, "ok": rag_ok}

    def answer(context, question):
        if grounded:
            return {"answer": "답변", "is_grounded": True, "ok": True}
        return {"answer": "제공된 문서에서 확인할 수 없습니다.",
                "is_grounded": False, "ok": True}

    class StubLLM:
        def invoke(self, prompt):
            class Response:
                content = grade
            return Response()

    return build_mm_agent(ocr, vision, rag, answer, StubLLM())


def _nodes(state):
    return [step.split("(")[0] for step in state["route_history"]]


def test_text_question_happy_path_has_no_fallbacks():
    state = run_mm_agent(_make_graph(0.9, "x" * 100, "yes", True), "질문")
    assert _nodes(state) == ["rag", "grade", "generate"]
    assert state["fallback_history"] == []
    assert state["refused"] is False


def test_image_input_with_usable_ocr_skips_vision():
    state = run_mm_agent(
        _make_graph(0.9, "x" * 100, "yes", True), "질문", "image", "img.png"
    )
    assert _nodes(state) == ["ocr", "rag", "grade", "generate"]
    assert state["fallback_history"] == []


def test_dead_ocr_fires_f1_then_f2_then_f3_refusal():
    state = run_mm_agent(
        _make_graph(0.0, "", "no", False), "질문", "image", "img.png"
    )
    assert _nodes(state) == [
        "ocr", "vision_input", "rag", "grade", "vision_escalate", "generate", "refuse",
    ]
    assert len(state["fallback_history"]) == 3
    assert state["refused"] is True


def test_ungrounded_generate_triggers_f2_despite_lenient_grade():
    # measured gap (scenario run 1): grade passed caption context that the
    # generator then refused on — the ungrounded generate must escalate.
    state = run_mm_agent(_make_graph(0.9, "x" * 100, "yes", False), "질문")
    assert _nodes(state) == [
        "rag", "grade", "generate", "vision_escalate", "generate", "refuse",
    ]
    assert state["fallback_history"][0].startswith("F2: generate(ungrounded)")
    assert state["refused"] is True


def test_post_generate_escalation_recovers_when_second_answer_grounded():
    calls = {"n": 0}

    def answer(context, question):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"answer": "제공된 문서에서 확인할 수 없습니다.",
                    "is_grounded": False, "ok": True}
        return {"answer": "우측 상단입니다.", "is_grounded": True, "ok": True}

    def ocr(path):
        return {"text": "", "confidence": 0.0, "ok": True}

    def vision(path, question, **kwargs):
        return {"image_summary": "vision-summary", "confidence": 0.8, "ok": True}

    def rag(query):
        return {"docs": [REAL_PAGE_DOC], "scores": [0.9],
                "caption_hit": True, "ok": True}

    class StubLLM:
        def invoke(self, prompt):
            class Response:
                content = "yes"
            return Response()

    graph = build_mm_agent(ocr, vision, rag, answer, StubLLM())
    state = run_mm_agent(graph, "질문")
    assert _nodes(state) == ["rag", "grade", "generate", "vision_escalate", "generate"]
    assert state["refused"] is False
    assert state["is_grounded"] is True
    assert len(state["fallback_history"]) == 1  # F2 only — recovered, no F3


def test_near_blank_page_high_conf_low_text_still_fires_f1():
    # measured trap: near-blank pages OCR at conf ~0.9 with 2 chars
    state = run_mm_agent(
        _make_graph(0.92, "aq", "yes", True), "질문", "image", "img.png"
    )
    assert _nodes(state)[1] == "vision_input"
    assert state["fallback_history"][0].startswith("F1")


def test_insufficient_evidence_escalates_to_vision_once_then_generates():
    state = run_mm_agent(_make_graph(0.9, "x" * 100, "no", True), "질문")
    assert _nodes(state) == ["rag", "grade", "vision_escalate", "generate"]
    assert len(state["fallback_history"]) == 1
    assert state["fallback_history"][0].startswith("F2")


def test_rag_hard_failure_routes_to_refusal():
    state = run_mm_agent(
        _make_graph(0.9, "x" * 100, "yes", True, rag_ok=False), "질문"
    )
    assert _nodes(state) == ["rag", "refuse"]
    assert state["refused"] is True
    assert state["answer"]  # refusal message, never empty
