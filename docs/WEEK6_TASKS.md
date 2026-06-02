# WEEK 6 — Agentic RAG (LangGraph)

> **이 작업은 `PROJECT_CONTEXT.md` Phase 1 (Advanced RAG)의 네 번째이자 마지막 단계다.**
> 시작 전 `PROJECT_CONTEXT.md`, `docs/week5_retrospective.md`, `docs/adr/week5_retrieval_strategy.md`를 반드시 읽는다.
> 이번 주차 발표자는 **아님**. 발표 자료 산출물 없음. 대신 ADR-009 작성이 필수.

---

## 0. 이 주차의 핵심 원칙 (5주차 결과에 비춰)

- **점수 향상이 핵심 산출물이 아니다.** 5주차에서 이미 91.3% Top-1을 달성. 23문항 중 잔여 2건뿐이라 headroom은 8.7%p에 불과. **6주차의 진짜 가치는 routing 판단 / 재검색 효과 / latency 비용 / 답변 거절 기준의 운영적 통찰**.
- **단순 파이프라인 → 의사결정 시스템의 전환 경험**. 화려한 구조보다 4개 노드를 제대로.
- **변수 통제**: chunking(C3), embedding, retriever(Hybrid+Rerank)는 5주차 최종 그대로. 바뀌는 것은 **Agentic routing 구조**뿐.
- **23문항 한계 인지**: 5주차에서 1문항 = 4.3%p. 점수 변화가 통계적 신호인지 노이즈인지 구분 어려움. **7주차 평가셋 확대의 의제로 명시적 기록**.

---

## 1. 5주차 회고 정리 (선결)

> 산출물: `docs/week6_retrospective.md` 의 첫 섹션

스터디 과제 1번 항목. 다음 정리:

1. **5주차 최종 retrieval 전략 = Hybrid (BM25+Dense, RRF) + bge-reranker-v2-m3-ko**
   - 6주차 Agentic RAG의 `retrieve` 노드에 그대로 사용
   - Top-1 91.3%, Top-5 100%, latency ~6.7s

2. **5주차 이후 잔여 실패 케이스**
   - Q17 (와이파이): category-agnostic 질문, ground truth 의심 — **본 분석에서 제외, 단 코드 실행에는 포함**
   - Q20 (공기청정기 소음): explicit category keyword가 reranker에서도 안 먹힘 — **Self-Query/metadata filter 후보**
   - **추가 케이스 (6주차 진행 중 발견 시)**: 23문항 중 Top-5 진입에 실패한 0건 외에도, top-1이 흔들리는 borderline 케이스가 있을 수 있음

3. **"왜 단일 RAG 파이프라인이 아니라 Agentic 구조가 필요한가"** (한 단락)
   - 5주차 Hybrid+Rerank는 retrieval 자체는 100% Top-5에 정답을 회수한다. 하지만 Top-1 정확도가 91.3%에서 막혀 있고, 잔여 케이스의 본질이 "검색 알고리즘의 한계"가 아니라 **"질문 유형에 맞는 다른 처리가 필요"**한 케이스(category-agnostic, explicit keyword 무시 등)다. 즉 문제가 **단일 알고리즘 튜닝이 아니라 시스템 구조의 문제**로 옮겨졌다. Agentic 구조는 검색 결과를 보고 다음 행동을 조건부로 선택할 수 있게 해서, 같은 retriever로도 질문 유형별 다른 전략을 적용할 수 있다.

---

## 2. 평가셋 처리 방침

- **23문항 그대로 유지** (확대 보류 — 7주차 의제로 이월)
- **Q17은 코드 실행에는 포함, 분석에서만 제외**
  - 이유: ground truth가 의심스러우나(category-agnostic), 코드 일관성 위해 빼지 않음
  - Q17의 routing 동작은 기록하되, ablation 통계 산출에서는 별도 표기
- **22문항 기준 통계 + 23문항 기준 통계 둘 다 표에 표기** (투명성)
- 평가셋 확대는 7주차 작업으로 명시 기록

---

## 3. 패키지 설치

```bash
uv add langgraph
uv add grandalf   # 선택: workflow diagram 시각화
# langchain-tavily 는 설치하지 않음 (Web Search 제외 결정)
```

---

## 4. LangGraph StateGraph 구현 (필수)

> 산출물: `notebooks/week6_agentic_rag.ipynb`

### 4.1 State 정의

```python
from typing_extensions import TypedDict
from typing import List, Any, Optional

class GraphState(TypedDict):
    question: str
    rewritten_question: str
    metadata_filter: Optional[dict]  # Self-Query 결과 저장 (§5.3)
    documents: List[Any]
    answer: str
    grade_result: str         # "relevant" / "not_relevant"
    retry_count: int
    route_history: List[str]   # 디버깅·분석용
    latency_breakdown: dict    # 노드별 latency 기록
```

> `metadata_filter`, `route_history`, `latency_breakdown`은 스터디 기본 가이드에는 없지만, **운영 분석을 위한 필수 확장**. ADR Trade-off에서 "관찰 가능성 비용"으로 기록.

### 4.2 4개 필수 노드

#### (1) `retrieve`
- 5주차 최종 Hybrid+Rerank retriever 그대로
- 단, `metadata_filter`가 state에 있으면 적용 (Self-Query 결과 활용 — §5.3)
- top-k=20 → reranker top-5

#### (2) `grade_documents` — LLM Judge 방식 채택
- 5주차에서 reranker score를 이미 활용했으므로, threshold 방식은 정보 중복
- LLM Judge로 "관련성 + 충분성" 둘 다 판단:
  - "이 문서들이 질문에 답하는 데 충분히 관련 있는가? yes/no"
- **본 도메인 가이드**: 한국어 매뉴얼 질문이므로 LLM Judge prompt도 한국어로
- 출력: `grade_result = "relevant" | "not_relevant"`

#### (3) `rewrite_query` — **Self-Query 통합 (본인 결정 (나))**
- 단순 query 재작성이 아니라 **두 가지 기능 결합**:
  - (a) **Self-Query 부분**: 질문에서 명시적 category keyword 감지("공기청정기", "정수기", "청소기", 모델명 패턴) → `metadata_filter` 생성
  - (b) **Query 재작성 부분**: 도메인 키워드 추가, 동의어, 더 구체적 표현
- 두 기능을 같은 노드에서 처리하되, **분기로 명확히 구분**:
  ```
  if 첫 retry & explicit category keyword 감지됨:
      → Self-Query (metadata_filter 추가, query는 유지)
  elif 두 번째 retry:
      → Query 재작성 (도메인 키워드 추가)
  ```
- **이 설계의 정당화 (ADR에 기록)**: Q20처럼 explicit keyword 있는 케이스는 filter 추가만으로 충분. Q03/Q19처럼 keyword 없는 케이스는 query 재작성 필요. 두 처방을 별도 retry 단계로 분리.

#### (4) `generate`
- 5주차 RAG prompt 재사용
- 조건: context 기반 답변 / 근거 부족 시 "제공된 문서에서 확인할 수 없습니다" / 출처 명시

### 4.3 조건부 엣지 (Routing)

```
START → retrieve → grade_documents
  ├─ relevant → generate → END
  └─ not_relevant:
      ├─ retry_count < 2 → rewrite_query → retrieve (재시도)
      └─ retry_count >= 2 → cannot_answer → END
```

**필수 조건:**
- `retry_count` 최대 **2회**로 제한 (스터디 가이드의 2~3회 중 보수적 선택)
- 같은 query로 재검색 금지: `rewrite_query`는 반드시 metadata_filter 변경 OR query 텍스트 변경
- 2회 retry 후에도 실패 시 답변 거절

---

## 5. Self-Query를 rewrite 노드 안에서 구현 (본인 결정 (나) 상세)

### 5.1 LangChain Self-Query vs 직접 구현

- LangChain `SelfQueryRetriever`는 자체 retriever라 6주차 LangGraph 구조에 직접 호환 안 됨
- **추천: 가벼운 LLM 기반 추출기를 rewrite_query 노드에 직접 구현**
  - 입력: 사용자 질문
  - 출력: `{"category": "airpurifier" | "waterpurifier" | "vacuumcleaner" | None, "model_name": "..." | None}`
  - LLM call 1회 (gpt-4o-mini)
- 추출된 값으로 `metadata_filter = {"category": ...}` state에 저장

### 5.2 metadata 활용 — 4주차 메타데이터 스키마와 연결

- 4주차에서 부착한 `category`, `complexity`, `model_name` 메타데이터를 6주차에서 처음 활용
- **이게 4주차 메타데이터 설계의 약속이 6주차에서 회수되는 순간** — ADR에 명시

### 5.3 retrieve 노드의 filter 적용

- `retrieve` 노드가 state의 `metadata_filter`를 확인
- 있으면 Chroma retriever의 `filter` 인자에 전달
- 없으면 일반 hybrid+rerank

### 5.4 본인 결정 (나)의 reasoning을 ADR에 기록

> Self-Query를 (가) retrieve 안의 gate-keeper로 두는 대신 (나) rewrite_query 노드에서 처리한 이유: **첫 검색은 모든 정보로 시도하고, 실패 시에만 metadata 제약을 추가**하는 progressive constraint 접근. 처음부터 filter를 강제하면 Q17처럼 category-agnostic 질문에서 잘못된 제약이 됨. 실제로 Q17은 ground truth가 의심스럽고, 첫 검색에서 다른 카테고리 문서가 충분히 답할 수 있을 가능성이 있음.

---

## 6. Workflow Diagram (필수)

> 산출물: `docs/week6_workflow_diagram.md`

- Mermaid 형식 권장: `graph.get_graph().draw_mermaid()` 출력을 그대로 저장
- PNG 출력은 선택 (`grandalf` 또는 `draw_mermaid_png()`)
- diagram에 **본인 설계의 분기 의도가 보이도록** 노드 라벨에 짧은 설명 첨부 (예: `rewrite_query [Self-Query 또는 키워드 추가]`)

---

## 7. Agentic RAG vs Baseline 정량 비교 (필수)

> 산출물: `docs/week6_retrospective.md` 의 표 섹션

### 7.1 비교 표 — 22문항 / 23문항 둘 다 표기

| 구성 | Top-1 (22q) | Top-1 (23q) | Faithfulness | Answer Relevancy | Context Precision | 평균 Latency(s) |
|---|---|---|---|---|---|---|
| Baseline: 5주차 Hybrid+Rerank | ? | 91.3% | ? | ? | ? | 6.73 |
| Agentic RAG | ? | ? | ? | ? | ? | ? |

- 22q = Q17 제외 / 23q = Q17 포함 (코드는 돌리되 분석에서 제외)
- Faithfulness / Answer Relevancy / Context Precision = RAGAS 3지표 (5주차에서 환경 풀렸으면 측정)
- Latency는 wall clock + 노드별 breakdown 둘 다 기록

### 7.2 변수 통제

- chunking: 4주차 C3 그대로
- embedding: 5주차와 동일
- retriever: 5주차 Hybrid+Rerank 그대로
- LLM: gpt-4o-mini, temperature=0.1
- 변경된 것은 **graph routing 구조**뿐

### 7.3 점수 해석 — 5주차 결과에 비춘 현실적 기대치

- Top-1 91.3% → Agentic으로 갈 수 있는 최대치는 100% (8.7%p 추가). 1문항 더 풀면 +4.3%p
- **점수가 안 오르더라도 실패가 아님** — Faithfulness 향상 (근거 없을 때 거절), latency 동향, routing 통계가 진짜 산출물
- 점수 떨어지면 (라우팅 오작동, rewrite 오작동, retrieval 악화) 그 원인 분석이 회고의 핵심

---

## 8. 운영 관점 회고 (필수)

> 산출물: `docs/week6_retrospective.md`

### 8.1 필수 기록 항목

- **재검색이 도움이 된 사례 1개** (id + 첫 검색 결과 + rewrite 후 query/filter + 두 번째 검색 결과)
- **재검색이 불필요했거나 악영향을 준 사례 1개** (rewrite로 query가 산으로 갔거나, 첫 검색에서 충분했는데 grade가 잘못 잡은 경우)
- **Baseline 대비 평균 latency 변화** (절댓값 + 배수)
- **Agentic RAG가 본 도메인에 필요한가**에 대한 본인 판단

### 8.2 본 도메인 특화 분석 항목

- **Q20 (공기청정기 소음) Agentic RAG로 해결되었는가** — Self-Query metadata filter가 작동했는지
- **Q17 (와이파이) Agentic RAG의 routing 동작** — 분석에서 제외하지만 동작 기록은 함. retry를 얼마나 했는지, 답변을 거절했는지
- **22문항 중 5주차에서 이미 풀린 21문항이 Agentic RAG에서도 풀리는가** — regression 발생 여부. 5주차에 풀던 걸 못 풀면 라우팅 비용이 정확도를 깎은 케이스
- **불필요한 retry 통계**: 첫 검색이 옳았는데 grade가 not_relevant로 잘못 판단해서 retry 들어간 케이스 비율

### 8.3 질문 유형별 분석

5주차에서 categorical 분석 이미 했음 (category, question_type, retrieval_bias). 6주차에서도 같은 라벨로:
- 어떤 카테고리에서 Agentic RAG 효과 큰가 (5주차 airpurifier가 가장 어려웠음)
- 어떤 question_type에서 효과 큰가 (5주차 troubleshooting, part_location이 약했음)
- 어떤 retrieval_bias에서 효과 큰가

---

## 9. ADR-009 작성 (필수)

> 산출물: `docs/adr/week6_agentic_rag.md`
> `PROJECT_CONTEXT.md`의 ADR 목록에 **ADR-009로 등재 예정**

### 9.1 구조 (스터디 가이드 형식)

1. **Decision** — Agentic RAG 구조 도입 여부 + 형태
   - 예: "5주차 Hybrid+Rerank를 baseline으로 두고, LangGraph 기반 retrieve → grade → rewrite (Self-Query 통합) → generate 4노드 구조 + retry 2회 + 답변 거절 도입"
2. **Context** — 왜 필요했는가
   - 5주차 잔여 케이스의 본질이 "검색 알고리즘 한계"가 아닌 "질문 유형에 맞는 다른 처리 필요"
   - 4주차 메타데이터(category)가 6주차에서 처음 활용되는 회수 지점
3. **Alternatives** — 검토했으나 안 쓴 것
   - Web Search Fallback: **closed-domain 매뉴얼 QA에서 부적합** (잘못된 모델 정보 혼입 위험) — 명시적 제외 결정
   - Reranker threshold 조정만: routing 의사결정 능력 부재
   - Self-RAG / CRAG: 6주차 범위 밖, 7주차 비교 분석 후보
   - Self-Query를 retrieve gate-keeper로: §5.4 reasoning 참조 (progressive constraint 선택)
4. **Trade-off** — 비용
   - Latency 증가 (배수 기록)
   - LLM 호출 비용 증가 (grade + Self-Query/rewrite 추가)
   - 디버깅 복잡도 (route_history 도입으로 완화)
   - Routing 실패 가능성 (grade가 잘못 잡으면 불필요한 retry)
   - **평가 설계 복잡도** (Q17 같은 ground truth 의심 케이스 별도 처리 필요)
5. **Consequence** — 7주차 Evaluation으로 연결
   - **평가셋 확대** (23 → 30~50문항)가 7주차의 첫 의제
   - Routing accuracy, Refusal accuracy, Citation accuracy 등 도메인 특화 메트릭
   - Q17 같은 ambiguous ground truth 처리 방침 정립
   - Agentic RAG가 필요한 질문 유형 vs 필요 없는 질문 유형 구분 (모든 질문에 Agentic 발동은 비효율적)

---

## 10. 산출물 체크리스트

### 필수
- [ ] `notebooks/week6_agentic_rag.ipynb` (§4, §5)
- [ ] `docs/week6_retrospective.md` (§1, §7 표, §8 운영 회고)
- [ ] `docs/week6_workflow_diagram.md` (§6)
- [ ] `docs/adr/week6_agentic_rag.md` (§9)

### 선택 제출
- [ ] `docs/week6_workflow_diagram.png`

### 도전 과제 (선택, 시간 여유 시)
- [ ] **Reflect 노드 추가**: generate 후 hallucination 검증. 본 도메인에선 가전 매뉴얼이라 hallucination 비용이 낮은 편이라 우선순위 낮음.
- [ ] **LangSmith Trace 시각화**: 노드별 latency, retry 통계 자동 수집. 본인 운영 회고와 자연스럽게 결합.
- [ ] **Self-RAG / CRAG 비교 분석** (구현 없이 논문 기반): ADR-009의 Alternatives 섹션을 깊게 채울 수 있음.
- [ ] **Multi-hop 질문 처리**: 본 도메인 평가셋엔 multi-hop 질문 부재. 새 평가 질문 추가 필요. 7주차 평가셋 확대와 함께 진행 권장.
- [ ] ~~Web Search Fallback~~: **본인 결정으로 제외** (§3, §9.3)

> **도전 과제 우선순위**: LangSmith가 가장 ROI 높음 (운영 회고 자동화). Self-RAG/CRAG 분석은 ADR 깊이 보강. 나머지는 7주차 이후로 미뤄도 됨.

---

## 11. 작업 순서 권장

1. `PROJECT_CONTEXT.md` + `docs/week5_retrospective.md` + `docs/adr/week5_retrieval_strategy.md` 다시 읽기
2. **§3 패키지 설치** (langgraph)
3. **§1 5주차 회고 정리** — retrospective 첫 섹션
4. **§4 LangGraph 4노드 구현** — 단순 흐름부터 (retrieve → grade → generate, retry 없이) 동작 확인 후 retry/rewrite 추가
5. **§5 Self-Query를 rewrite 노드에 통합** — category 추출기 → metadata_filter → retrieve에 전달
6. **§6 Workflow diagram 출력**
7. **§7 Agentic RAG 23문항 실행** — Q17 포함, latency_breakdown 수집
8. **§7 비교 표 작성** — 22q / 23q 둘 다, RAGAS 환경 풀렸으면 3지표 포함
9. **§8 운영 회고** — 재검색 도움/악영향 사례, latency 변화, routing 통계
10. **§9 ADR-009 작성**
11. (여유) §10 도전 과제 — LangSmith 우선

---

## 12. 6주차 → 7주차 연결 메모

- **평가셋 확대 (23 → 30~50)** — 6주차 회고의 핵심 next step
- **Q17 같은 ground truth 의심 케이스 처리 방침 정립**
- **Routing accuracy, Refusal accuracy, Citation accuracy** 도메인 특화 메트릭 설계
- **Agentic RAG 발동 조건의 정교화** — 모든 질문에 발동이 아니라 조건부 발동
- Phase 1 마무리. Phase 2 (Multimodal) 시작 준비.

---

## 13. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| (초안) | 5주차 회고(Hybrid+Rerank 91.3%, 잔여 Q17/Q20) 반영. 본인 결정 5개 반영: Self-Query를 rewrite_query에 통합 (나) / 평가셋 23개 유지 / Web Search 제외 / baseline=5주차 최종 / Q17은 코드 실행 포함하되 분석 제외. |