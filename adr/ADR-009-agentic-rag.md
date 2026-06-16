# ADR-009: Agentic RAG (LangGraph 4노드 + Self-Query 통합) 도입

## 상황

5주차 Hybrid + Rerank (ADR-008)로 Top-1 91.3% (21/23), Top-5 100.0% 달성. 그러나 잔여 실패 2건의 본질이 검색 알고리즘 한계가 아니라 **질문 유형에 따른 다른 처리 필요**로 판명:

| ID | Question | 본질적 문제 |
|---|---|---|
| Q17 | "와이파이 연결이 안 될 때" | Wi-Fi가 모든 카테고리에 등장 (category-agnostic) |
| Q20 | "공기청정기 소음이 심해요" | "공기청정기" explicit keyword가 reranker score를 압도 못 함 |

단일 retriever 튜닝으로는 두 케이스 모두 해결 불가. **검색 결과를 보고 다음 행동(필터 추가·query rewrite·거절)을 조건부로 결정할 수 있는 시스템 구조**가 필요했다.

또한 7주차 평가 인프라(Baseline vs Agentic 비교) 및 Phase 2 cross-modal 확장의 토대가 필요하다.

## 고려한 선택지

| 옵션 | Top-1 (23q) | Latency | 설명 |
|---|---|---|---|
| Baseline 유지 (Hybrid+Rerank) | 91.3% | 6.73s | 추가 구현 없음. Q17·Q20 미해결. |
| Self-Query Retriever만 추가 | 추정 ~95% | ~8s | Metadata filter 자동 추출. retry/거절 메커니즘 없음. |
| **LangGraph 4노드 + Self-Query 통합** | **95.7%** | **27.01s** | retrieve→grade→rewrite(Self-Query)→generate. retry≤2 후 거절. |
| LangGraph + ReAct multi-tool agent | 미측정 | 추정 30s+ | Tool 선택 단계까지 LLM. 본 단일 매뉴얼 검색 도메인에 과대설계. |

## 최종 결정

**LangGraph StateGraph 4노드 + Self-Query 통합 (`rewrite_query` 노드 내).**

```
retrieve → grade_documents → (relevant?) ─Y→ generate
                              ─N→ rewrite_query → retrieve (retry ≤ 2)
                                    └ 최종 실패 시 cannot_answer (거절)
```

- 모듈: `src/agent.py` (`GraphState`, `make_filterable_hybrid_retriever`, `build_agent_graph`, `run_agent`, `REFUSAL_MESSAGE`).
- `make_filterable_hybrid_retriever`는 `src/retrieval.py`의 Hybrid+Rerank를 wrap하면서 `metadata_filter` 인자를 받는 callable을 반환 — Self-Query가 추출한 `{'category': ...}` 필터를 격리.
- LLM: temperature=0 (재현성), gpt-4o-mini.

## 이유

1. **잔여 케이스의 본질에 맞는 구조.** Q20 같은 케이스는 "1차 검색 실패 → category 추출 → 2차 검색"의 multi-step decision이 본질이고, single-pass retriever로는 표현 불가.
2. **잘못된 답 대신 정직한 거절.** Baseline은 Q20에서 vacuumcleaner 답변을 자신 있게 반환. Agentic은 grader가 not_relevant 판정 → retry → 최종 거절. **매뉴얼 도메인의 운영 안전성은 답 속도보다 정직한 거절이 우선.**
3. **7주차/Phase 2 공유 토대.** 같은 `run_agent` 인터페이스를 7주차 평가 하네스가 호출. Phase 2 cross-modal은 retrieve 노드만 교체.
4. **실측 결과 (`data/week6_agentic_rag_results.json`, 2026-06-15):**

   | 지표 | Baseline | Agentic | Δ |
   |---|---|---|---|
   | Top-1 (22q, Q17 제외) | 90.9% | **100.0%** | +9.1pp |
   | Top-1 (23q, all) | 91.3% | **95.7%** | +4.4pp |
   | 평균 Latency | 6.73s | 27.01s | **4.01x** |
   | 거절률 (cannot_answer) | 0% | 정확도와 분리 측정 (7주차 §6) | — |

5. **5주차 21문항 regression 통과** — Baseline이 푼 21q 모두 Agentic에서도 정답 유지.

## Trade-off 수용

| Trade-off | 정량 | 수용 근거 |
|---|---|---|
| Latency 4.01x (6.73 → 27.01s) | 절댓값 +20.28s/문항 | retrieve 라운드 누적이 latency의 ~86%. 실시간 SLA가 절대 기준인 도메인에는 부적합하나, **본 매뉴얼 QA는 정확성·안전성 우선**으로 합의됨 (CLAUDE.md §1.1). |
| 불필요 retry ~50% (2/4건) | Q7·Q22는 grader false-negative 추정 | grader prompt 개선·confidence threshold 추가로 후속 개선 가능. 본 ADR 범위 밖. |
| Q17 routing 한계 | 1건 미해결 | grader가 category-agnostic 질문에서 회수 문서를 relevant로 판정 → retry 미발동. ground truth 의심 케이스라 22q 분석에서 제외(ADR-008 §잔여 한계 합의). |
| LLM 비용 증가 | grade·self-query·rewrite 각 1회 추가 LLM 호출 / retry당 동일 | gpt-4o-mini 사용으로 절대 비용 부담 낮음. 7주차에서 상세 측정. |

## 잔여 한계 (Phase 2 / 후속 입력)

- **Image-required 케이스의 텍스트 한계.** Q22 (공기청정기 프리필터 위치)는 6주차 평가에서 retry=2 끝에 정답 표시됐으나, 본문 위치 서술 부재. 텍스트 RAG의 본질적 한계 → Phase 2 cross-modal 도입의 motivation. (LIM-002, 7주차 §9 modality별 분해.)
- **Grader false-negative.** retry=2 케이스 4건 중 2건(Q7·Q22)이 grader 오판으로 인한 불필요 retry로 추정. grader가 본 latency의 주요 변동 요인.
- **7주차에서 RAGAS 정식 측정 미반영.** 본 ADR은 Top-1·latency·routing 통계까지만 다룬다. Faithfulness·AR·CP·CR 4지표 + Refusal accuracy + Citation accuracy는 7주차 `docs/week7_retrospective.md` §4~6에서 측정.

## 향후 계획

1. **7주차 (Phase 1 종료):** 같은 `run_agent` 모듈을 Golden Set v2(35 + image-required 8) 위에서 Baseline과 동시 측정. RAGAS·Refusal·Citation 산출. q_type·modality별 분해.
2. **Phase 2 (cross-modal):** `src/agent.py`의 `GraphState`에 비워둔 `image_context: Optional[...]` 자리를 활용. retrieve 노드만 cross-modal로 교체하고 grade·rewrite·generate는 재사용. 그래프 구조 변경 없음.
3. **(후속 개선 후보)** grader prompt에 confidence-threshold/explicit category 검출 추가 → 불필요 retry 감소. Phase 1 마무리 후 evaluation 결과를 보고 결정.

---

**관련 ADR:**
- ADR-008 (Hybrid + Rerank 검색 전략) — `retrieve` 노드의 backbone.
- ADR-010 (Evaluation framework) — 본 ADR의 정량 평가 인프라.
- ADR-011 (Cross-modal evaluation, 초안 예정) — Phase 2 연결.

**참조 문서:**
- `docs/week6_retrospective.md` §2~4 — 본 ADR 정량 근거.
- `docs/WEEK6_REVISED_TASKS.md` §6 — 본 ADR 작성 지시.
- `src/agent.py` — 구현.
- `data/week6_agentic_rag_results.json` — 실측 데이터.
