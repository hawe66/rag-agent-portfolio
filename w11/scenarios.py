"""
Week 11 multi-step scenarios (WEEK11_TASKS §4).

Runs the multimodal agent (`src/mm_agent.py`) through the required flows and
records route/fallback histories to ``data/week11_scenarios.json`` — the raw
material for the §5 architecture doc's scenario table.

    S1  happy path      image page → OCR → RAG → answer
    S2  core fallback   text question (IR-A1) → caption RAG insufficient
                        → F2 Vision escalation on the retrieved page → answer
    S3  edge            OCR-dead diagram page + unanswerable question
                        → F1 Vision bypass → RAG → insufficient → refusal
    S4  bonus           IR-A3 (icon shape) — second F2 case, probes whether
                        query-side full-page vision beats the caption ceiling

Results are flushed after each scenario so a crash never loses finished runs
(same checkpoint discipline as the Week 9 harness).

Run:  .venv/bin/python -m w11.week11_scenarios
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from ..src.agent_tools import (
    image_analysis_tool,
    make_answer_generation_tool,
    make_rag_search_tool,
    ocr_tool,
)
from ..src.mm_agent import build_mm_agent, run_mm_agent
from ..src.mm_retrieval import ModalityAwareRetriever
from ..src.multimodal import load_mm_store
from ..src.retrieval import create_reranker

AGENT_MODEL = "gpt-4o-mini"  # grade + answer, same as Week 6/9 answerers
OUT_PATH = Path("data/week11_scenarios.json")
IMAGES = Path("data/sample_images_150")

SCENARIOS = [
    {
        "id": "S1",
        "name": "happy path: 이미지 페이지 → OCR → 검색 → 답변",
        "input_type": "image",
        "image_path": str(IMAGES / "waterpurifier_complex" / "p029.png"),
        "question": "정수기 필터의 교체 주기는 얼마나 되나요?",
        "expected_flow": ["ocr", "rag", "grade", "generate"],
        "expected_fallbacks": 0,
    },
    {
        "id": "S2",
        "name": "core: 텍스트 질문(IR-A1) → 캡션 RAG 근거 부족 → Vision 에스컬레이션",
        "input_type": "text",
        "image_path": None,
        "question": "AS281DAW 공기청정기의 조작부는 제품 앞면의 어느 위치에 있나요?",
        "expected_flow": ["rag", "grade", "vision_escalate", "generate"],
        "expected_fallbacks": 1,
    },
    {
        "id": "S3",
        "name": "edge: OCR-dead 도면 페이지 + 매뉴얼 밖 질문 → F1 우회 → 근거 부족 → 거절",
        "input_type": "image",
        "image_path": str(IMAGES / "waterpurifier_simple" / "p008.png"),
        "question": "이 제품의 와이파이 초기 비밀번호는 무엇인가요?",
        # F2 may or may not fire before the refusal — the required outcome is
        # F1 + refused, so expected_flow lists the mandatory prefix only.
        "expected_flow": ["ocr", "vision_input", "rag", "grade"],
        "expected_fallbacks": 2,  # at least F1 + F3
    },
    {
        "id": "S4",
        "name": "bonus: IR-A3 아이콘 모양 — 두 번째 F2 케이스 (캡션 상한 vs 질의측 vision)",
        "input_type": "text",
        "image_path": None,
        "question": "AS281DAW 공기청정기 상태 표시부에서 공기제균 기능이 켜지면 표시되는 아이콘은 어떤 모양인가요?",
        "expected_flow": ["rag", "grade", "vision_escalate", "generate"],
        "expected_fallbacks": 1,
    },
]


def _node_names(route_history: list[str]) -> list[str]:
    return [step.split("(")[0] for step in route_history]


def _serialize_docs(docs, scores) -> list[dict]:
    rows = []
    for i, doc in enumerate(docs):
        meta = doc.metadata
        rows.append({
            "tag": f"[{meta.get('category')}_{meta.get('complexity')} p.{meta.get('page')}]",
            "modality": meta.get("modality", "text"),
            "score": round(scores[i], 4) if i < len(scores) else None,
            "content_head": doc.page_content[:120],
        })
    return rows


def _flow_match(scenario: dict, actual: list[str]) -> bool:
    """Expected flow is a required *prefix subsequence*: every expected node
    must appear in order (S3's optional F2 detour must not fail the check)."""
    it = iter(actual)
    return all(node in it for node in scenario["expected_flow"])


def run_scenarios(scenarios: list[dict] = SCENARIOS, out_path: Path = OUT_PATH) -> list[dict]:
    print("Loading reranker + mm store (one-time)...", flush=True)
    reranker = create_reranker()
    mm_store = load_mm_store()
    retriever = ModalityAwareRetriever(mm_store, reranker)
    llm = ChatOpenAI(model=AGENT_MODEL, temperature=0)

    graph = build_mm_agent(
        ocr_tool,
        image_analysis_tool,
        make_rag_search_tool(retriever, reranker),
        make_answer_generation_tool(llm),
        llm,
    )

    results: list[dict] = []
    for scenario in scenarios:
        print(f"\n=== {scenario['id']}: {scenario['name']}", flush=True)
        state = run_mm_agent(
            graph,
            scenario["question"],
            input_type=scenario["input_type"],
            image_path=scenario["image_path"],
        )
        actual_flow = _node_names(state["route_history"])
        row = {
            "id": scenario["id"],
            "name": scenario["name"],
            "input_type": scenario["input_type"],
            "image_path": scenario["image_path"],
            "question": scenario["question"],
            "expected_flow": scenario["expected_flow"],
            "actual_flow": actual_flow,
            "flow_match": _flow_match(scenario, actual_flow),
            "route_history": state["route_history"],
            "fallback_history": state["fallback_history"],
            "expected_fallbacks": scenario["expected_fallbacks"],
            "caption_hit": state.get("caption_hit"),
            "scores": [round(s, 4) for s in state.get("scores", [])],
            "retrieved": _serialize_docs(state.get("docs", []), state.get("scores", [])),
            "ocr_text_head": state.get("ocr_text", "")[:200],
            "image_summary": state.get("image_summary", ""),
            "grade_result": state.get("grade_result"),
            "is_grounded": state.get("is_grounded"),
            "refused": state.get("refused"),
            "answer": state.get("answer"),
            "latency_s": round(state["total_latency"], 2),
        }
        results.append(row)
        # flush after every scenario — crash loses at most one run
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"  flow: {' → '.join(actual_flow)}", flush=True)
        print(f"  fallbacks: {row['fallback_history'] or 'none'}", flush=True)
        print(f"  refused={row['refused']}  latency={row['latency_s']}s", flush=True)
        print(f"  answer: {row['answer'][:150]}", flush=True)

    print(f"\nwrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    run_scenarios()
