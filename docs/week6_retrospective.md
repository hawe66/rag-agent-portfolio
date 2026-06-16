# Week 6 Retrospective

## 1. 5주차 회고 정리

### 1.1 5주차 최종 Retrieval 전략

**Hybrid (BM25 + Dense, RRF) + bge-reranker-v2-m3-ko**

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 91.3% (21/23) |
| Top-5 Accuracy | 100.0% (23/23) |
| Latency | ~6.73s |

이 전략은 6주차 Agentic RAG의 `retrieve` 노드에 그대로 사용된다.

### 1.2 5주차 이후 잔여 실패 케이스 (2건)

| ID | Question | Expected | Actual | 실패 원인 |
|----|----------|----------|--------|-----------|
| Q17 | 와이파이 연결이 안 될 때 | waterpurifier | vacuumcleaner | Wi-Fi는 category-agnostic 기능 |
| Q20 | 공기청정기 소음이 심해요 | airpurifier | vacuumcleaner | "소음" 관련 청소기 문서가 rerank score 높음 |

**처리 방침:**
- Q17: Ground truth 의심 — **분석에서 제외, 코드 실행에는 포함**
- Q20: Explicit category keyword("공기청정기")가 reranker에서도 작동 안 함 — **Self-Query/metadata filter 후보**

### 1.3 왜 Agentic 구조가 필요한가

5주차 Hybrid+Rerank는 retrieval 자체는 100% Top-5에 정답을 회수한다. 하지만 Top-1 정확도가 91.3%에서 막혀 있고, 잔여 케이스의 본질이 "검색 알고리즘의 한계"가 아니라 **"질문 유형에 맞는 다른 처리가 필요"**한 케이스(category-agnostic, explicit keyword 무시 등)다.

즉 문제가 **단일 알고리즘 튜닝이 아니라 시스템 구조의 문제**로 옮겨졌다. Agentic 구조는 검색 결과를 보고 다음 행동을 조건부로 선택할 수 있게 해서, 같은 retriever로도 질문 유형별 다른 전략을 적용할 수 있다.

---

## 2. Agentic RAG vs Baseline 정량 비교

> 본 표는 `src/agent.py` 모듈 기반 23q 재실행(2026-06-15)의 실측치다. 같은 retriever·LLM 설정(temperature=0).

| 구성 | Top-1 (22q) | Top-1 (23q) | 평균 Latency(s) |
|---|---|---|---|
| Baseline: 5주차 Hybrid+Rerank | 90.9% (20/22) | 91.3% (21/23) | 6.73 |
| Agentic RAG | **100.0% (22/22)** | **95.7% (22/23)** | 27.01 |

- RAGAS 4지표(Faithfulness / Answer Relevancy / Context Precision / Context Recall)는 **7주차 `docs/week7_retrospective.md` §4** 정식 측정으로 위임한다(중복 측정 방지). 7주차는 같은 `run_agent` 모듈을 호출하므로 위 Top-1과 정합이 자동으로 보장된다.
- Latency 배수 = 27.01 / 6.73 ≈ **4.01x**. WEEK6_REVISED §5.2에 메모됐던 5.1x 추정치는 실측 4.01x로 정정한다.

### 2.1 5주차 21문항 regression

5주차 Baseline이 푼 21문항(Q17·Q20 제외) 중 **21문항 모두 Agentic에서도 정답**. 모듈화·routing 추가로 인한 회귀 없음.

---

## 3. 운영 관점 회고

### 3.1 재검색이 도움이 된 사례

**Q20 (공기청정기 소음이 심해요)** — Agentic 사이클이 끝까지 작동한 대표 케이스 (재현 노트: cell 26의 `test_question = "공기청정기 소음이 심해요"`):

```
1차 retrieve(filter=None)        → grade(not_relevant)
2차 self_query(category=airpurifier)
2차 retrieve(filter={'category':'airpurifier'}) → grade(not_relevant)
3차 rewrite("LG 공기청정기 소음 문제 해결 방법은 무엇인가요?")
3차 retrieve(filter={'category':'airpurifier'}) → grade(not_relevant)
→ cannot_answer (retry=2)
```

- **Self-Query metadata filter가 자동 발동**되어 1차 카테고리 오류는 격리됨. 다만 본문에 소음 관련 위치 서술이 없어 최종 거절. **잘못된 답을 주지 않은 것 자체가 운영 가치** (5주차 Baseline은 vacuumcleaner 답변을 자신 있게 반환).

**Q22 (공기청정기 프리필터는 어디에 있나요?)** — retry=2, 66s. 재검색이 정답까지 도달한 케이스. 그러나 image-required 후보로 분류되어 **텍스트만으로 푼 답변의 충실도 자체는 7주차에서 추가 검증** 필요 (LIM-002 참조).

### 3.2 재검색이 불필요했거나 악영향을 준 사례

**Q7 (공기청정기 필터 청소는 어떻게 하나요?)** — retry=2, 43.6s, 최종 정답.

- 첫 retrieve(filter=None)부터 airpurifier 문서가 회수됐을 가능성이 있으나 grader가 not_relevant로 판정 → 불필요한 self_query + rewrite로 latency 36s 추가.
- **grade-LLM의 false-negative**가 retry 통계의 주범이다. 6.73s × 1 → 43.6s로 약 6.5배 늘어났는데도 정답이 같다면 retry는 손해.

| retry | 건수 | 누적 추가 latency 추정 |
|---|---|---|
| retry=0 | 19건 | — |
| retry=1 | 1건 (Q6) | ~+20s |
| retry=2 | 3건 (Q7, Q20, Q22) | ~+40s × 3 |

retry=2 케이스 3건 중 **Q20만 retry가 본질적으로 필요한 케이스**(Self-Query category 추출). Q7·Q22는 grader 보정 시 retry 없이 풀릴 수 있는 후보다. → 불필요 retry 추정 비율: 2/4 ≈ 50%.

### 3.3 Baseline 대비 평균 latency 변화

- 절댓값: 6.73s → **27.01s** (+20.28s)
- 배수: **4.01x**
- 분해(grader/self-query/rewrite/추가 retrieve의 합)는 실행 로그의 `latency_breakdown`에 보존됨. cell 26 test 케이스(Q20 trial): retrieve 38.96 + grade 4.39 + rewrite 1.92 = 약 45.3s — retrieve 라운드 자체가 latency의 대부분 (~86%).

### 3.4 Agentic RAG가 본 도메인에 필요한가

판단: **조건부로 필요.**

- 5주차 Baseline 91.3%, Agentic 95.7%. 절대 +4.4pp 향상 = 1문항 추가 해결(Q20 류). 22q 기준 90.9% → 100% (+9.1pp).
- 대가: latency 4.01x. 사용자가 즉답을 기대하는 매뉴얼 QA에서 27s는 부담.
- **본 도메인의 실제 가치는 거절 능력**이다. Baseline은 검색이 빗나가도 답을 만들어내지만 (Q20에서 vacuumcleaner 답변), Agentic은 grader가 not_relevant라 판단하면 retry → 마지막에 정직하게 거절한다. **factuality와 정직한 거절이 답변 속도보다 중요한 운영 환경**(매뉴얼 안전사고 가능성 등)에서 Agentic이 정당화된다.
- 단순 정확도만 보면 4.01x 비용이 +4.4pp의 대가로 합당한지는 사용자 SLA에 달림. 본 포트폴리오에서는 **Phase 2 cross-modal로 확장하면서 routing/거절 인프라가 재사용되는 자산**으로 본다 (ADR-009 §향후 계획).

---

## 4. 본 도메인 특화 분석

### 4.1 Q20 (공기청정기 소음) 해결 여부

- Self-Query metadata filter는 **발동 확인** (route_history에 `self_query(category=airpurifier)` → `retrieve(filter={'category':'airpurifier'})` 명시).
- 평가 루프(cell 29) 결과: **O (retry=2, 56.1s, 최종 정답)**. test 케이스(cell 26)는 본문 검색 실패로 cannot_answer까지 갔으나 평가 루프는 동일 질문에서 정답 도달 — temperature=0이지만 rewrite query의 미세 차이 등으로 결과가 갈렸을 가능성. 7주차 다회 측정으로 안정성 추가 검증 권장.

### 4.2 Q17 (와이파이) Routing 동작

- Route: `[retrieve(filter=None), grade(relevant), generate]` — retry 없이 1회 통과.
- Expected: waterpurifier / Predicted: vacuumcleaner.
- **메타데이터 필터 발동 안 함** (Self-Query가 "와이파이"라는 category-agnostic 키워드만으로 카테고리 추론 불가). grader도 회수된 vacuumcleaner 문서를 relevant로 판정.
- ground truth 자체가 의심스러운 케이스이므로 정확도 분석에서는 제외(22q 기준), 다만 **routing이 의심스러운 케이스에 retry를 발동시키지 못한 점은 grader의 한계**로 기록.

### 4.3 Regression 여부

- 5주차 Baseline이 푼 21문항 (Q17·Q20 제외) → Agentic에서 **21/21 모두 정답 유지**.
- 모듈화·routing 추가로 인한 회귀 없음.

### 4.4 불필요한 retry 통계

전체 retry 분포:

| retry | 건수 | 케이스 |
|---|---|---|
| retry=0 | 19 | 정상 통과 |
| retry=1 | 1 | Q6 (AS281DAW 필터 수명) |
| retry=2 | 3 | Q7, Q20, Q22 |

retry=2 케이스 4건(Q6 포함)의 정성 분석:

| Case | 본질적으로 필요한 retry? | 비고 |
|---|---|---|
| Q6 | 부분 필요 | 모델명(AS281DAW) exact-match가 1차에 약함, grader가 not_relevant 판정 후 rewrite로 보강 |
| Q7 | 불필요 추정 | 1차 검색이 정답 카테고리 포함했을 가능성 — grader false-negative |
| Q20 | 필요 | Self-Query category 추출이 본질적으로 필요한 케이스 |
| Q22 | 불필요 추정 | image-required 후보; 텍스트만으로는 어떤 retry도 본질 해결 못 함 |

**불필요 retry 추정: 2/4 = 50%.** grader prompt 개선 또는 confidence threshold 추가로 latency 절감 여지 있음 (Phase 1 마무리 후 후속 개선 후보).

---

## 5. 한계와 Phase 2 연결

- **Q22 (프리필터 위치)와 같은 image-required 케이스는 본 retrospective에서 "정답"으로 집계됐지만, 텍스트 본문만으로는 위치 정보가 충분히 서술되지 않는다.** Phase 2 cross-modal 측정 시 별도 정답률을 확인해야 한다 (LIM-002 참조, 7주차 §9에서 modality별 분해).
- `src/agent.py`의 `GraphState`는 `# image_context: Optional[list]  # cross-modal 도입 시 retrieve 노드가 채움` 자리를 비워뒀다. Phase 2는 같은 그래프 구조에 retrieve 노드만 cross-modal로 교체하면 된다.

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|-----------|
| 2026-06-02 | 초안 작성: §1 5주차 회고 정리 완료 |
| 2026-06-15 | §2 비교표 / §3 운영 회고 / §4 도메인 분석을 모듈 기반 재실행 실측치(2026-06-15)로 채움. latency 배수 4.01x로 정정. §5 Phase 2 연결 추가. |
