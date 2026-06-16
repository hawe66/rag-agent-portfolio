"""
Agentic RAG with LangGraph (Week 6).

4-node graph: retrieve → grade_documents → rewrite_query → generate.
On `not_relevant` grade, loops back through rewrite_query (Self-Query on first
retry, free-form rewrite on second). After `max_retries`, routes to a refusal.

Extracted from `w6/week6_agentic_rag.ipynb` without behavior change so the
notebook can call this module and 7-week evaluation can reuse it.

Public surface:
- ``GraphState`` — TypedDict carrying retrieval/grade/answer state.
- ``make_filterable_hybrid_retriever`` — wraps the notebook's hybrid + filter +
  rerank logic into a callable ``(query, metadata_filter) -> list[Document]``.
- ``build_agent_graph(retriever, llm, max_retries=2)`` — returns a compiled graph.
- ``run_agent(graph, question)`` — single-question helper returning the result dict
  used by the 7-week evaluation harness.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, List, Optional

from typing_extensions import TypedDict

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.retrieval import rerank_documents


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class GraphState(TypedDict):
    """LangGraph state for the agentic RAG loop."""

    question: str
    rewritten_question: str
    metadata_filter: Optional[dict]
    documents: List[Any]
    answer: str
    grade_result: str  # "relevant" | "not_relevant"
    retry_count: int
    route_history: List[str]
    latency_breakdown: dict
    # --- Phase 2 자리 (이번엔 미사용) ---
    # image_context: Optional[list]  # cross-modal 도입 시 retrieve 노드가 채움


RetrieverFn = Callable[[str, Optional[dict]], List[Document]]


# ---------------------------------------------------------------------------
# Retriever factory — hybrid + metadata filter + rerank
# ---------------------------------------------------------------------------


def make_filterable_hybrid_retriever(
    vectorstore,
    all_docs: List[Document],
    reranker,
    first_stage_k: int = 20,
    top_k: int = 5,
) -> RetrieverFn:
    """Build the notebook's retriever as a single callable.

    The notebook's `retrieve_node` re-builds a BM25 index per call when a
    metadata filter is present, because `BM25Retriever` does not natively
    support metadata filtering. Encapsulating that here keeps `build_agent_graph`
    free of retrieval plumbing and gives Phase 2 a single seam to swap
    cross-modal retrieval into.
    """

    def retrieve(query: str, metadata_filter: Optional[dict] = None) -> List[Document]:
        # Dense retriever with optional Chroma filter
        search_kwargs: dict = {"k": first_stage_k}
        if metadata_filter:
            search_kwargs["filter"] = metadata_filter
        dense_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        dense_docs = dense_retriever.invoke(query)

        # BM25 — filter manually since BM25Retriever has no native filter
        if metadata_filter:
            filtered_docs = [
                doc
                for doc in all_docs
                if all(doc.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
            bm25_retriever = BM25Retriever.from_documents(filtered_docs, k=first_stage_k)
        else:
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=first_stage_k)
        bm25_docs = bm25_retriever.invoke(query)

        # Simple dedupe by leading content (matches notebook behavior)
        seen: set = set()
        merged: List[Document] = []
        for doc in dense_docs + bm25_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                merged.append(doc)

        return rerank_documents(reranker, query, merged, top_k=top_k)

    return retrieve


# ---------------------------------------------------------------------------
# Prompts (verbatim from notebook)
# ---------------------------------------------------------------------------


GRADE_PROMPT = """당신은 검색 결과의 관련성을 판단하는 평가자입니다.

질문: {question}

검색된 문서들:
{documents}

위 문서들이 질문에 답하는 데 충분히 관련 있습니까?
- 문서에 질문의 답이 포함되어 있거나, 답을 도출하기에 충분한 정보가 있으면 "yes"
- 문서가 질문과 관련이 없거나, 답을 도출하기에 정보가 부족하면 "no"

한 단어로만 답하세요 (yes 또는 no):"""


SELF_QUERY_PROMPT = """당신은 사용자 질문에서 제품 카테고리와 모델명을 추출하는 도우미입니다.

질문: {question}

아래 규칙에 따라 카테고리를 추출하세요:
- "공기청정기", "에어컨", "청정기" 언급 → airpurifier
- "정수기", "물" 언급 → waterpurifier
- "청소기", "흡입" 언급 → vacuumcleaner
- 모델명 패턴: AS로 시작 → airpurifier, WD로 시작 → waterpurifier, A9/K8/O9로 시작 → vacuumcleaner
- 위 규칙에 해당하지 않으면 null

JSON 형식으로만 답하세요 (다른 텍스트 없이):
{{"category": "airpurifier" | "waterpurifier" | "vacuumcleaner" | null, "model_name": "모델명" | null}}"""


REWRITE_PROMPT = """당신은 검색 품질을 높이기 위해 질문을 재작성하는 도우미입니다.

원본 질문: {question}

검색에 더 효과적인 질문으로 재작성하세요:
- 도메인 키워드(LG, 가전, 매뉴얼)를 추가
- 동의어나 더 구체적인 표현 사용
- 질문 의도를 명확히 유지

재작성된 질문만 출력하세요:"""


GENERATE_PROMPT = """당신은 LG 가전제품 매뉴얼을 기반으로 사용자 질문에 답변하는 도우미입니다.

## 규칙
1. 제공된 문서에 있는 정보만 사용하여 답변하세요.
2. 문서에 없는 정보는 추측하지 마세요.
3. 근거가 부족하면 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.
4. 각 참고 문서 앞에는 `[제품_복잡도 p.페이지]` 형태의 출처 태그가 붙어 있습니다.
5. 답변 끝에 `(출처: 제품_복잡도 p.페이지)` 형태로 사용한 근거의 출처를 명시하세요.

## 질문
{question}

## 참고 문서
{context}

## 답변"""


REFUSAL_MESSAGE = "죄송합니다. 제공된 매뉴얼에서 해당 질문에 대한 정보를 찾을 수 없습니다."


# ---------------------------------------------------------------------------
# Node builders — return closures bound to retriever/llm
# ---------------------------------------------------------------------------


def _bump_latency(state: GraphState, key: str, elapsed: float) -> dict:
    latency = state.get("latency_breakdown", {}).copy()
    latency[key] = latency.get(key, 0) + elapsed
    return latency


def _append_history(state: GraphState, label: str) -> List[str]:
    history = state.get("route_history", []).copy()
    history.append(label)
    return history


def _build_retrieve_node(retriever: RetrieverFn):
    def retrieve_node(state: GraphState) -> GraphState:
        start = time.time()
        query = state.get("rewritten_question") or state["question"]
        metadata_filter = state.get("metadata_filter")

        docs = retriever(query, metadata_filter)

        elapsed = time.time() - start
        return {
            **state,
            "documents": docs,
            "latency_breakdown": _bump_latency(state, "retrieve", elapsed),
            "route_history": _append_history(state, f"retrieve(filter={metadata_filter})"),
        }

    return retrieve_node


def _build_grade_node(llm):
    def grade_documents_node(state: GraphState) -> GraphState:
        start = time.time()
        question = state["question"]
        documents = state["documents"]

        docs_text = "\n\n".join(
            [
                f"[문서 {i+1}] (category: {doc.metadata.get('category', 'unknown')})\n{doc.page_content[:500]}..."
                for i, doc in enumerate(documents[:3])
            ]
        )

        prompt = GRADE_PROMPT.format(question=question, documents=docs_text)
        response = llm.invoke(prompt)

        grade = response.content.strip().lower()
        grade_result = "relevant" if grade == "yes" else "not_relevant"

        elapsed = time.time() - start
        return {
            **state,
            "grade_result": grade_result,
            "latency_breakdown": _bump_latency(state, "grade", elapsed),
            "route_history": _append_history(state, f"grade({grade_result})"),
        }

    return grade_documents_node


def _build_rewrite_node(llm):
    def rewrite_query_node(state: GraphState) -> GraphState:
        start = time.time()
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        current_filter = state.get("metadata_filter")

        new_filter = current_filter
        new_query = state.get("rewritten_question") or question
        action = "none"

        if retry_count == 0 and current_filter is None:
            # First retry: Self-Query (metadata extraction)
            prompt = SELF_QUERY_PROMPT.format(question=question)
            response = llm.invoke(prompt)
            try:
                content = response.content.strip()
                content = re.sub(r"^```json\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
                extracted = json.loads(content)
                if extracted.get("category"):
                    new_filter = {"category": extracted["category"]}
                    action = f"self_query(category={extracted['category']})"
                else:
                    action = "self_query(no_category_found)"
            except (json.JSONDecodeError, KeyError):
                action = "self_query(parse_error)"
        else:
            # Second retry: free-form query rewrite
            prompt = REWRITE_PROMPT.format(question=question)
            response = llm.invoke(prompt)
            new_query = response.content.strip()
            action = f"rewrite('{new_query[:30]}...')"

        elapsed = time.time() - start
        return {
            **state,
            "rewritten_question": new_query,
            "metadata_filter": new_filter,
            "retry_count": retry_count + 1,
            "latency_breakdown": _bump_latency(state, "rewrite", elapsed),
            "route_history": _append_history(state, action),
        }

    return rewrite_query_node


def _build_generate_node(llm):
    def generate_node(state: GraphState) -> GraphState:
        start = time.time()
        question = state["question"]
        documents = state["documents"]

        def _tag(doc):
            cat = doc.metadata.get("category", "unknown")
            cplx = doc.metadata.get("complexity", "")
            page = doc.metadata.get("page", "?")
            head = f"{cat}_{cplx}" if cplx else cat
            return f"[{head} p.{page}]"

        context = "\n\n".join(f"{_tag(d)} {d.page_content}" for d in documents)

        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = llm.invoke(prompt)

        elapsed = time.time() - start
        return {
            **state,
            "answer": response.content,
            "latency_breakdown": {**state.get("latency_breakdown", {}), "generate": elapsed},
            "route_history": _append_history(state, "generate"),
        }

    return generate_node


def _cannot_answer_node(state: GraphState) -> GraphState:
    return {
        **state,
        "answer": REFUSAL_MESSAGE,
        "route_history": _append_history(state, "cannot_answer"),
    }


def _build_route_after_grade(max_retries: int):
    def route_after_grade(state: GraphState) -> str:
        grade_result = state.get("grade_result", "not_relevant")
        retry_count = state.get("retry_count", 0)
        if grade_result == "relevant":
            return "generate"
        if retry_count < max_retries:
            return "rewrite_query"
        return "cannot_answer"

    return route_after_grade


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_agent_graph(retriever: RetrieverFn, llm, max_retries: int = 2):
    """Compile the 4-node agentic RAG graph.

    Args:
        retriever: Callable ``(query, metadata_filter) -> list[Document]``.
            Use ``make_filterable_hybrid_retriever`` to wire the notebook's
            hybrid+filter+rerank pipeline; any callable with that signature
            works (Phase 2 cross-modal swaps here).
        llm: LangChain chat model used by grade/rewrite/generate nodes.
        max_retries: Loop ceiling before the graph routes to the refusal node.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", _build_retrieve_node(retriever))
    workflow.add_node("grade_documents", _build_grade_node(llm))
    workflow.add_node("rewrite_query", _build_rewrite_node(llm))
    workflow.add_node("generate", _build_generate_node(llm))
    workflow.add_node("cannot_answer", _cannot_answer_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        _build_route_after_grade(max_retries),
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "cannot_answer": "cannot_answer",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("cannot_answer", END)

    return workflow.compile()


def run_agent(graph, question: str) -> dict:
    """Run a single question through the compiled graph.

    Returns the final state plus ``total_latency`` (wall-clock seconds).
    Used by both the W6 evaluation loop and the W7 evaluation harness so
    the two scripts measure the same object.
    """
    initial_state: GraphState = {
        "question": question,
        "rewritten_question": "",
        "metadata_filter": None,
        "documents": [],
        "answer": "",
        "grade_result": "",
        "retry_count": 0,
        "route_history": [],
        "latency_breakdown": {},
    }

    start = time.time()
    result = graph.invoke(initial_state)
    result["total_latency"] = time.time() - start
    return result
