# WEEK 6 REVISED — Agentic RAG 정직한 마무리 + 모듈화 (Phase 1 Closeout, 1단계)

> **이 문서는 `WEEK6_TASKS.md`를 대체(supersede)한다.** 기존 명세는 "Agentic 첫 구현"용이었고, 1회 sprint로 구현·실행까지 됐으나 **회고가 비어 있고(§2~4 TBD) 코드가 노트북에만 있다.** 이 문서는 그 미완을 닫고, Phase 2(cross-modal) 재사용이 가능하도록 모듈화하는 것이 목표다.
> 시작 전 `PROJECT_CONTEXT.md`, `docs/week5_retrospective.md`, `docs/week6_retrospective.md`(§1만 차 있음), `docs/week7_retrospective.md`(§1.1에 Agentic 실측치 있음)를 읽는다.
> **경로 주의:** 실제 repo는 `w6/`·`w7/`·`adr/ADR-xxx-*.md`·`docs/weekN_retrospective.md` 구조다. (기존 명세의 `notebooks/`·`docs/adr/`는 오기.)

---

## 0. 현재 상태 진단 (이 sprint의 출발점)

이미 된 것:
- `w6/week6_agentic_rag.ipynb` — LangGraph 4노드(retrieve → grade_documents → rewrite_query(Self-Query 통합) → generate, retry 2회 후 거절) 구현 및 실행 완료.
- `data/week6_agentic_rag_results.json` — 실행 결과 존재.
- 실측치(`week7_retrospective.md §1.1`): **Top-1 95.7%(22/23), 22q 기준 100%, latency 34.46s, retry 분포 retry0=19/retry1=1/retry2=3, 유일 실패 Q17.**

안 된 것 (이번에 닫는다):
1. `week6_retrospective.md` §2~4가 전부 **TBD** — Baseline vs Agentic 비교표, 운영 회고, Q20 해결 여부, regression, 불필요 retry 통계가 비어 있음.
2. **LangGraph가 노트북에만 존재** — `src/`에 모듈 없음. 7주차 평가(Baseline vs Agentic 동시 측정)와 Phase 2 재사용 모두 불가.
3. **ADR-009 파일 부재** — `adr/`에 008·010만 있고 009 없음.

---

## 1. 이 sprint의 핵심 원칙

- **숫자를 다시 만든다, 베껴오지 않는다.** §1.1의 실측치는 노트북 1회 실행 결과다. 모듈화 후 재실행하여 같은 숫자가 나오는지(재현성) 확인하고, 회고는 재실행 산출물 기준으로 채운다.
- **모듈화는 리팩토링이지 재설계가 아니다.** 노트북의 노드 로직을 그대로 `src/agent.py`로 옮긴다. 동작이 바뀌면 안 된다(regression = Top-1 95.7% 유지).
- **Phase 2 자리만 비워둔다, 구현은 안 한다.** `GraphState`에 이미지 컨텍스트가 들어올 자리를 주석/Optional 필드로 남기되 이번엔 텍스트만 흐른다 (CLAUDE.md 단순성 원칙 — 지금 안 쓰는 코드는 만들지 않는다).

---

## 2. 우선순위 트랙

| 우선순위 | 작업 | 산출물 | 비고 |
|---|---|---|---|
| **P0** | LangGraph를 `src/agent.py`로 모듈화 | `src/agent.py` | §3 |
| **P0** | 모듈로 재실행 → regression 확인 (Top-1 95.7% 유지) | `data/week6_agentic_rag_results.json` 갱신 | §4 |
| **P0** | `week6_retrospective.md` §2~4 실측치로 채움 | retrospective | §5 |
| **P0** | ADR-009 작성 + `PROJECT_CONTEXT.md` 등재 | `adr/ADR-009-agentic-rag.md` | §6 |
| **P1** | 불필요 retry / Q20 / Q17 routing 심화 분석 | retrospective §4.x | §5.3 |

> P0가 안 닫히면 7주차(Baseline vs Agentic 동시 평가)가 시작 불가다. `src/agent.py`가 7주차의 의존성.

---

## 3. `src/agent.py` 모듈화 (P0)

> 산출물: `src/agent.py`. 노트북 `w6/week6_agentic_rag.ipynb`의 로직을 함수/클래스로 추출.

### 3.1 노출할 공개 인터페이스

```python
# src/agent.py
from typing_extensions import TypedDict
from typing import Any, Optional

class GraphState(TypedDict):
    question: str
    rewritten_question: str
    metadata_filter: Optional[dict]
    documents: list[Any]
    answer: str
    grade_result: str          # "relevant" | "not_relevant"
    retry_count: int
    route_history: list[str]
    latency_breakdown: dict
    # --- Phase 2 자리 (이번엔 미사용) ---
    # image_context: Optional[list] = None  # cross-modal 도입 시 retrieve가 채움

def build_agent_graph(retriever, llm, max_retries: int = 2):
    """노트북의 4노드 그래프를 그대로 빌드해 compiled graph를 반환."""
    ...

def run_agent(graph, question: str) -> dict:
    """단일 질문 실행 → {answer, documents, route_history, latency_breakdown, retry_count} 반환.
    7주차 평가 하네스가 이 함수를 호출한다."""
    ...
```

- 노드 4개(retrieve / grade_documents / rewrite_query / generate)와 조건부 엣지는 `WEEK6_TASKS.md` §4~5의 설계 그대로. **로직 변경 금지.**
- `retriever`는 5주차 최종 Hybrid+Rerank(`src/retrieval.py`)를 주입받는다. agent 모듈이 retriever를 직접 만들지 않는다(테스트·교체 용이).
- 노트북은 `from src.agent import build_agent_graph, run_agent`로 갈아끼워 셀을 슬림화한다.

### 3.2 검증 기준

- `src/agent.py` import 후 골든셋(또는 기존 23문항)으로 재실행 → **Top-1 95.7%(22/23) 재현.** 어긋나면 모듈화 과정의 누락이므로 노트북과 diff 비교.
- `run_agent` 반환 구조가 7주차 평가 하네스(§WEEK7_REVISED §3)가 기대하는 키를 포함하는지 확인.

---

## 4. 모듈 재실행 & regression (P0)

> 산출물: `data/week6_agentic_rag_results.json` 갱신(모듈 기반 재생성).

- `src/agent.py`로 23문항(Q17 포함) 재실행, `latency_breakdown` 수집.
- **regression 체크:** 5주차에서 풀린 21문항(22q 기준)이 모듈 Agentic에서도 풀리는가. 못 풀면 모듈화 결함.
- 숫자가 §1.1과 다르면 **다르다는 사실 자체를 회고에 기록**하고 원인(비결정성? 모듈화 누락?) 분석. temperature=0 고정으로 비결정성 최소화.

---

## 5. `week6_retrospective.md` §2~4 채우기 (P0)

> 기존 파일의 빈 표/TBD를 **재실행 실측치로** 채운다.

### 5.1 §2 비교표 (22q / 23q 둘 다)

| 구성 | Top-1(22q) | Top-1(23q) | Faithfulness | Answer Rel. | Context Prec. | 평균 Latency(s) |
|---|---|---|---|---|---|---|
| Baseline: Hybrid+Rerank | (5주차) | 91.3% | (채움) | (채움) | (채움) | 6.73 |
| Agentic RAG | 100% | 95.7% | (채움) | (채움) | (채움) | 34.46 |

- RAGAS 행은 7주차에서 동일 하네스로 재측정하므로, 여기선 Top-1·latency만 확정하고 RAGAS는 "7주차 §4에서 측정"으로 명시 링크해도 됨(중복 측정 방지).

### 5.2 §3 운영 회고 (필수 4항목)

- 재검색이 도움 된 사례 1건(id + 첫 검색 → rewrite 후 query/filter → 두 번째 검색).
- 재검색이 불필요/악영향 1건(grade 오판으로 들어간 retry 등).
- Baseline 대비 latency 변화(절댓값 6.73→34.46s, 배수 ≈5.1x).
- "Agentic이 본 도메인에 필요한가" 본인 판단(질문 유형 조건부 발동 관점).

### 5.3 §4 도메인 특화 분석 (P1 심화 가능)

- Q20(공기청정기 소음): Self-Query metadata filter가 실제 작동해 해결됐는지.
- Q17(와이파이): routing 동작 기록(retry 횟수, 거절 여부). 분석 통계에서는 제외.
- regression: 5주차 21문항 유지 여부.
- 불필요 retry 통계: 첫 검색이 옳았는데 grade가 not_relevant로 오판해 retry된 비율(retry1=1, retry2=3건의 내역 분해).

---

## 6. ADR-009 작성 (P0)

> 산출물: `adr/ADR-009-agentic-rag.md` (`adr/ADR-000-template.md` 형식). 작성 후 `PROJECT_CONTEXT.md` §2 ADR 목록과 §7 변경이력에 등재.

내용은 `WEEK6_TASKS.md` §9의 Decision/Context/Alternatives/Trade-off/Consequence를 따르되, **이번 실측치(95.7%, 5.1x latency)와 retry 분포를 근거로 채운다.** 특히:
- Trade-off에 latency 5.1x를 정량 기록.
- Consequence를 7주차(평가)와 Phase 2(cross-modal)로 연결 — "agent 모듈이 cross-modal retrieve로 확장될 자리"를 한 줄 명시.

---

## 7. 산출물 체크리스트

- [ ] `src/agent.py` (§3)
- [ ] `data/week6_agentic_rag_results.json` 모듈 기반 갱신 (§4)
- [ ] `docs/week6_retrospective.md` §2~4 실측치로 채움 (§5)
- [ ] `adr/ADR-009-agentic-rag.md` + `PROJECT_CONTEXT.md` 등재 (§6)
- [ ] (P1) 불필요 retry / Q20 / Q17 심화 분석 (§5.3)

---

## 8. 작업 순서

1. 진단 문서 읽기(§0 대상 파일들) + 노트북 현 상태 확인.
2. §3 `src/agent.py` 추출 → 노트북을 모듈 호출로 교체.
3. §4 재실행 + regression 확인(Top-1 95.7% 재현).
4. §5 retrospective §2~4 채움.
5. §6 ADR-009 작성 + PROJECT_CONTEXT 등재.
6. (P1) §5.3 심화.

---

## 9. WEEK7_REVISED로 연결

- `src/agent.py`의 `run_agent`가 7주차 평가 하네스의 입력. **이 인터페이스가 7주차의 선결 의존성.**
- 6주차에서 확정한 latency·routing 통계는 7주차 q_type별 분석의 입력.

---

## 10. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-06-14 | `WEEK6_TASKS.md` supersede. 1회 sprint 후 미완(회고 §2~4, 모듈화, ADR-009) 마감 + Phase 2 재사용 위한 `src/agent.py` 모듈화로 재정의. 실제 repo 경로(w6/, adr/ADR-xxx) 반영. |
