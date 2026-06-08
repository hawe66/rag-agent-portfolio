# WEEK 7 — Evaluation 전담 (Phase 1 마무리, Pre-Freeze)

> **이 작업은 `PROJECT_CONTEXT.md` Phase 1 (Advanced RAG)의 마지막 평가 주차다.**
> 8주차는 1차 Freeze 주차이므로, 7주차 결과가 그대로 freeze된다.
> 시작 전 `PROJECT_CONTEXT.md`, `docs/week5_retrospective.md`, `docs/week6_retrospective.md` (§1만 채워진 상태)를 반드시 읽는다.
> 이번 주차 발표자는 **아님**. 8주차 세미나 발표자도 아님 (보안 섹션 한 단락만 별도 작업).

---

## 0. 이 주차의 핵심 원칙 (시간 압박 인지)

- **"만드는 능력보다 측정하는 능력"** — 스터디 가이드의 메시지.
- **(다) 우선순위 기반 진행**: 평가 인프라 (Golden Set v1 + Baseline RAGAS + Refusal Accuracy + Citation Accuracy + RAGAS 한계) 먼저, 6주차 Agentic 마무리는 시간 되면. 안 되면 (나) 미완성자 경로로 폴백.
- **5주차 Baseline 결과 재사용 가능** — 7주차 가이드 명시. 5주차 retriever (Hybrid+Rerank, 91.3%) 그대로 사용.
- **Phase 2 확장성**: 평가 체계는 Phase 2 cross-modal에서도 재사용. Golden Set 라벨링에 "image-required" 같은 modality 라벨을 미리 비워두면 좋음.

---

## 1. 우선순위 트랙 (시간 빠듯 시 위에서부터)

| 우선순위 | 작업 | 산출물 | 비고 |
|---|---|---|---|
| **P0** | Golden Set v1 구축 (23 → 20~30, 도메인 재해석) | `data/eval/golden_set_v1.csv` | §3 |
| **P0** | Baseline RAGAS 4지표 측정 (Context Recall 포함) | retrospective §4 표 | §4 |
| **P0** | Refusal Accuracy 측정 | retrospective §6 | §6 |
| **P0** | Citation Accuracy 측정 | retrospective §6 | §6 |
| **P0** | RAGAS 한계 사례 분석 (2건: 점수 ↔ 실제 어긋남) | retrospective §5 | §5 |
| **P0** | ADR-010 evaluation framework | `docs/adr/week7_evaluation.md` | §9 |
| **P1** | 6주차 Agentic LangGraph 마무리 | `notebooks/week6_agentic_rag.ipynb` | WEEK6_TASKS §4-5 참조 |
| **P1** | Agentic RAG RAGAS 4지표 측정 | retrospective §4 표 (Agentic 행) | §4 |
| **P1** | Pairwise judge 비교 (Baseline vs Agentic) | retrospective §7 | §7 |
| **P1** | 질문 유형별 결과 분석 | retrospective §8 | §8 |
| **P2** | (8주차) README 보안 섹션 한 단락 | README.md | §11 |

> **P0 못 끝나면 그게 위기.** P1은 못 해도 (나) 미완성자 경로로 8주차 freeze 가능. P0가 평가 인프라 자체.

---

## 2. 6주차 회고 마무리 (Agentic 진행 여부와 무관하게)

> 산출물: `docs/week7_retrospective.md` 첫 섹션

WEEK6_TASKS.md §8의 운영 회고를 옮기되, **Agentic 미완 상태**라면 다음과 같이 처리:

- 본인 추측/가설로라도 "어떤 질문 유형에서 Agentic이 도움 될 것인가" 한 단락
  - 예: out_of_scope, multi_hop은 Agentic 우위 예상 / factual은 baseline 충분 예상
- 5주차 잔여 케이스(Q17, Q20) 기준으로 가설 설정
- 실제 Agentic 결과는 §4에서 채우거나, P1 못 하면 가설로만 남기고 7주차 retrospective §8에 명시

---

## 3. Golden Set v1 구축 — 본 도메인 재해석

> 산출물: `data/eval/golden_set_v1.csv`

### 3.1 도메인 재해석 (스터디 가이드의 의료 예시를 가전 매뉴얼로)

| q_type | 본 도메인 정의 | 예시 |
|---|---|---|
| **factual** | 매뉴얼에 답이 명확한 단답/사양 | "AS281DAW의 필터 권장 교체 주기는?" "WD325AS의 정수 용량은?" |
| **comparison** | 모델 간 비교, 옵션 비교 | "WD325AS와 WD520AWB의 차이점은?" "정수 모드와 냉수 모드 차이?" |
| **multi_hop** | 여러 단계/근거 필요 | "공기청정기 필터 교체 후 어떤 초기화가 필요한가?" (교체 절차 + 후속 작업 두 섹션 참조) |
| **out_of_scope** | 매뉴얼에 없는 정보 | "이 모델 단종됐나요?" "최신 모델은 뭔가요?" "수리 비용은?" "다른 회사 제품과 비교하면?" |
| **safety** | 안전 안내 필요 | "필터를 직접 분해해도 되나요?" "물 묻은 손으로 콘센트 만져도 되나요?" "고장 났을 때 직접 수리?" |

> **본 도메인 safety 약점 인지:** 가전은 의료/법률만큼 safety가 강하지 않음. 그래도 "직접 분해/수리 금지", "감전 위험" 같은 안내가 매뉴얼에 있으니 이를 활용. 약한 safety 카테고리는 retrospective에서 도메인 특성으로 정직하게 기록.

### 3.2 문항 수 권장

- 필수: 20개 (각 q_type 최소 2개, q_type 5종)
- 권장 분포 예시:
  - factual 6 / comparison 4 / multi_hop 4 / out_of_scope 4 / safety 2 = 20

### 3.3 기존 23문항 재활용 전략

본인이 5주차에 만든 23문항(`docs/eval_questions_v2.json`이라고 가정)을 v1의 기반으로:

1. 기존 23문항을 q_type 라벨 부착 (대부분 factual/procedural이 많을 듯)
2. 부족한 q_type (특히 out_of_scope, safety) 신규 추가 — 최소 6~8개 추가
3. `reference_context` 라벨 부착 — 문서명 + 페이지 번호 (Context Recall 필수)
4. `ground_truth` 텍스트 — 정답 요지 1~2 문장

### 3.4 라벨링 노동 감축 옵션

**Context Recall은 reference 라벨 없으면 측정 불가** (RAGAS, DeepEval 둘 다). 자동 위임이 불가능. 다만:

**옵션 A — 수동 라벨링 (안전, 시간 듦)**
- 23문항 + 신규 7문항 = 30문항 × (페이지 + 정답 텍스트) 라벨링
- 한 문항당 2~3분, 총 1~1.5시간

**옵션 B — RAGAS `TestsetGenerator` 활용 (시간 절약, 검수 필요)**
- LLM이 문서에서 (question, ground_truth, reference_context) 트리플을 자동 생성
- 본인 매뉴얼 PDF 청크에서 10~15문항 자동 생성 → 본인이 검수/선별
- 옵션 A와 결합: 자동 생성으로 factual/comparison 확보, out_of_scope/safety는 수동 (자동 생성 불가)

**추천:** 옵션 A 메인 + 옵션 B로 시간 부족하면 보조. 옵션 B는 도전 과제 성격이라 P0 일정에서는 옵션 A 권장.

### 3.5 CSV 스키마

```csv
question,ground_truth,reference_context,q_type,modality_label,notes
"AS281DAW의 필터 권장 교체 주기는?","약 6~12개월","airpurifier_complex_AS281DAW.pdf p.15",factual,text-only,
"공기청정기 필터를 직접 분해해도 되나요?","서비스센터를 통해서만 분해/수리 권장","airpurifier_complex_AS281DAW.pdf p.3 (안전주의)",safety,text-only,
"이 모델 단종됐나요?","제공된 문서에서 확인할 수 없습니다","N/A",out_of_scope,N/A,
```

> `modality_label`은 Phase 2 확장 자리 (text-only / image-helpful / image-required) — 지금은 모두 text-only지만 컬럼 비워두면 Phase 2 평가에 그대로 재사용.

---

## 4. RAGAS 4지표 측정 (필수)

> 산출물: `notebooks/week7_evaluation.ipynb`, 결과 표는 retrospective §4

### 4.1 측정 지표

- Faithfulness
- Answer Relevancy
- Context Precision
- **Context Recall** (7주차 신규)

### 4.2 비교 대상

| 구성 | 출처 | 비고 |
|---|---|---|
| Baseline RAG | 5주차 최종 Hybrid+Rerank | 5주차 결과 재사용 가능 (가이드 명시) |
| Agentic RAG | 6주차 LangGraph | **P1 — 6주차 마무리 시에만** |

### 4.3 RAGAS 환경 점검

5주차에서 RAGAS 환경 풀렸으면 그대로 사용. 막혔으면 §1.2 WEEK5_TASKS의 폴백 기준 재적용. macOS 14라 가능성 높음.

### 4.4 Judge 모델 선택

- Generator: gpt-4o-mini
- **Judge 권장: gpt-4o 또는 gpt-4o-mini와 다른 계열 (self-preference 회피)**
- 비용 우려 시 gpt-4o-mini도 가능하지만 self-preference 한계는 retrospective에 명시
- 본인 환경/예산에 맞게 선택, 선택 이유 ADR에 기록

---

## 5. RAGAS 한계 직접 확인 (필수)

> 산출물: retrospective §5

다음 2건 이상 발굴:
- **점수는 높은데 실제 답변이 나쁜 사례** — 예: Faithfulness 높지만 질문 핵심 안 다룸 (Evidence Coverage 부족)
- **점수는 낮은데 실제 답변은 괜찮은 사례** — 예: Context Precision 낮지만 핵심 근거 포함됨 (정밀도 ≠ 답변 품질)

각 사례마다:
- 어떤 지표가 어긋났는가
- 왜 어긋났는가 (가설)
- 어떤 추가 지표가 필요한가

> **본 도메인 후보:** "필터 교체 방법" 같은 procedural 질문에서 RAGAS Faithfulness/Answer Relevancy가 높아도 실제로는 단계 일부 누락 — Evidence Coverage 결핍의 전형.

---

## 6. 도메인 특화 메트릭 (필수)

> 산출물: retrospective §6

### 6.1 Refusal Accuracy (필수)

**정의:**
- True Positive (TP): out_of_scope / safety 질문에 시스템이 올바르게 거절
- False Positive (FP): factual / comparison 질문에 시스템이 잘못 거절 (오거절)

**측정:**
```
Refusal Accuracy = (올바른 거절 + 올바른 응답) / 전체
오거절률(FP rate) = 답해야 하는데 거절한 수 / 답해야 하는 전체 수
```

본 도메인 거절 정답 패턴: "제공된 문서에서 확인할 수 없습니다" + 출처 미명시

> **6주차 ADR-009의 답변 거절 로직(retry 2회 후 거절)이 여기서 검증됨.**

### 6.2 Citation Accuracy (필수, 본인 결정)

**본 도메인 적합성 강함**: 가전 매뉴얼은 "어느 모델 어느 페이지" 출처가 본질. 4주차 메타데이터(model_name, page)가 회수되는 두 번째 지점 (첫 번째는 6주차 Self-Query).

**정의:**
- 답변에 명시한 (model_name, page) 출처가 reference_context와 일치하는가
- 모델명이 일치하면서 페이지가 ±2 범위 내면 정답 (도메인 휴리스틱)

**측정:**
```
Citation Accuracy = 올바른 출처 명시 답변 수 / 출처를 요구한 전체 답변 수
```

**계산 단계:**
1. 답변 텍스트에서 출처 추출 (regex: "AS281DAW p.15" 같은 패턴)
2. reference_context와 매칭
3. out_of_scope는 분모에서 제외 (출처 없어야 정답)

### 6.3 (P1) Routing Accuracy — Agentic 마무리 시에만

6주차 `route_history`를 활용. 본 도메인 정의:
- out_of_scope: "rewrite_query → retry → cannot_answer" route가 올바름
- factual: "retrieve → generate" 직행이 올바름
- 잘못된 route를 탄 비율

---

## 7. Pairwise LLM-as-judge (P1 — Agentic 마무리 시)

> 산출물: retrospective §7

### 7.1 비교 방식

같은 질문에 대해 A(Baseline)/B(Agentic) 답변을 judge가 비교, 더 나은 답변과 이유 출력.

### 7.2 평가 기준 (judge prompt)

- 질문에 직접 답했는가
- 제공된 context에 근거했는가
- 출처가 적절한가 (본 도메인은 model_name + page)
- 불확실 시 무리하게 답하지 않았는가
- safety 질문에서 안전 안내했는가
- 장황하거나 모호하지 않은가

### 7.3 샘플링

전체 30문항 부담스러우면 q_type별 2~3개씩 샘플링 (총 10~15문항). 시간 빠듯하면 우선 샘플링부터.

### 7.4 Self-preference 회피

generator와 다른 judge 모델 사용 (§4.4).

---

## 8. 질문 유형별 분석 (P1)

> 산출물: retrospective §8

| q_type | 문항 수 | Faithfulness | Context Recall | Refusal Acc | Citation Acc | 주요 관찰 |
|---|---|---|---|---|---|---|
| factual | ? | ? | ? | ? | ? | ? |
| comparison | ? | ? | ? | ? | ? | ? |
| multi_hop | ? | ? | ? | ? | ? | ? |
| out_of_scope | ? | N/A | N/A | ? | ? | ? |
| safety | ? | ? | N/A | ? | ? | ? |

**Baseline / Agentic 둘 다 같은 표** (Agentic 진행 시).

### 8.1 분석 질문

- 어떤 유형에서 Baseline 충분한가
- 어떤 유형에서 Agentic 효과 큰가 (or 본인 가설)
- 어떤 유형에서 Agentic 적용해도 이득 없는가
- 여전히 약한 유형은
- **8주차 freeze 전 가장 먼저 고쳐야 할 약점**

---

## 9. ADR-010 작성 (필수)

> 산출물: `docs/adr/week7_evaluation.md`
> `PROJECT_CONTEXT.md` ADR 목록에 **ADR-010**으로 등재

### 9.1 구조

1. **Decision** — 평가 체계
   - RAGAS 4지표 + Refusal Accuracy + Citation Accuracy + (가능 시) Pairwise judge
2. **Context** — RAGAS만으로 부족한 이유
   - 답변 거절 품질 측정 안 됨
   - 도메인 특화 출처 정확성 (model + page) 측정 안 됨
   - Agentic routing 적절성 측정 안 됨
3. **Alternatives** — 검토하고 안 쓴 것
   - 사람 전수 평가 (비용)
   - DeepEval 단독 사용 (RAGAS와 차이 검증할 시간 부족, 도전 과제 후보)
   - Pointwise judge만 (Pairwise가 더 robust)
   - 모든 문항 수동 채점 (확장성)
4. **Trade-off**
   - Golden Set 라벨링 공수 (수동 라벨)
   - Judge 모델 호출 비용
   - reference_context 라벨 품질이 Context Recall 신뢰도 결정
   - Judge 모델 편향 가능성 (mitigation: 다른 모델 사용)
   - **본 도메인 safety 카테고리 약함** (가전 도메인 한계, 정직하게 기록)
5. **Consequence** — 8주차 freeze로 연결
   - 7주차 수치 = 8주차 benchmark 표
   - 평가셋 v1 = freeze 후 회귀 테스트 기반
   - 질문 유형별 약점 → 8주차 README의 "Known Limitations" 섹션 / Phase 2 의제

---

## 10. 산출물 체크리스트

### P0 (필수)
- [ ] `data/eval/golden_set_v1.csv` (20~30문항, q_type 5종)
- [ ] `notebooks/week7_evaluation.ipynb` (§4 RAGAS 측정)
- [ ] `docs/week7_retrospective.md` (§2 6주차 회고 마무리, §4 RAGAS 표 baseline, §5 RAGAS 한계, §6 Refusal+Citation)
- [ ] `docs/adr/week7_evaluation.md` (§9 ADR-010)

### P1 (시간 되면)
- [ ] 6주차 `notebooks/week6_agentic_rag.ipynb` 마무리 (WEEK6_TASKS §4-5)
- [ ] retrospective §4 Agentic 행 채우기
- [ ] retrospective §7 Pairwise judge
- [ ] retrospective §8 질문 유형별 표

### P2 (8주차 별도)
- [ ] README 보안 섹션 한 단락 (§11)

---

## 11. 8주차 세미나 후 보안 섹션 (별도)

> 8주차 세미나(LlamaFirewall)는 본인 발표자 아님. 세미나 후 README에 한 단락 추가만 필요.

### 11.1 작성 시점

8주차 세미나 끝나고 (또는 미리 세미나 자료 훑은 후) README에 다음 항목 한 단락:
- 고려한 리스크 (예: prompt injection via PDF content, malicious metadata)
- prompt injection 대응 방향 (가전 도메인은 closed-domain이라 외부 입력 영향 제한적)
- context guardrail 필요성 (현재 없음, 향후 추가 가능)
- 도구 추가 시 권한 제어 (Phase 2 multimodal 도구 도입 시 고려)
- 향후 개선 계획

### 11.2 본 도메인의 보안 특성 (한 단락 예시 재료)

- **Closed-domain QA**: 외부 웹 접근 없음 (Web Search Fallback 명시적 제외 — ADR-009). 공격 표면 좁음.
- **Input layer**: 사용자 질문이 유일한 외부 입력. 가전 매뉴얼 도메인 특성상 jailbreak 인센티브 낮음.
- **Tool layer**: 현재 retrieval 외 도구 없음. Phase 2에서 multimodal 도구 추가 시 재평가.
- **Output layer**: 매뉴얼 외 정보 답변 거부 로직(6주차 retry 후 cannot_answer)이 기본 guardrail 역할.

> 이건 발표 자료가 아니라 README에 들어갈 한 단락의 재료. 본인이 세미나 듣고 추리거나, 별도 turn에서 같이 만들어도 됨.

---

## 12. 작업 순서 권장 (시간 박스)

1. `PROJECT_CONTEXT.md` + week5/6 retrospective 다시 읽기 (10분)
2. **§3 Golden Set v1 구축** — 기존 23문항 q_type 라벨 부착 + 신규 7~10문항 + reference_context (1.5~2시간)
3. **§4 Baseline RAGAS 4지표** — 5주차 데이터 재사용 (1시간, 환경 풀려 있으면)
4. **§6.1 Refusal Accuracy 측정** (30분)
5. **§6.2 Citation Accuracy 측정** (1시간 — 출처 추출 regex 작성 포함)
6. **§5 RAGAS 한계 사례 2건 발굴 + 분석** (30분)
7. **§9 ADR-010 작성** (30분)
8. P0 완료 — 여기서 멈추면 8주차 freeze 가능
9. (P1) 6주차 Agentic 마무리 — WEEK6_TASKS §4-5 참조 (반나절~하루)
10. (P1) §4 Agentic 행 + §7 Pairwise + §8 질문 유형별 표 (2~3시간)
11. (P2 / 8주차) §11 보안 섹션 한 단락 (30분)

---

## 13. 8주차 Freeze 연결 메모

- **7주차 retrospective = 8주차 benchmark 표 원본**
- **Golden Set v1 = freeze 후 회귀 테스트 기반**
- **ADR-008(retrieval) + ADR-009(agentic) + ADR-010(evaluation) = 8주차 README의 핵심 서사**
- Phase 1 마무리. Phase 2 (Multimodal) 시작 직전.
- 7주차 약점 분석 → 8주차 README "Known Limitations" 섹션 → Phase 2 의제

---

## 14. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| (초안) | 본인 결정 5개 반영: 절충 (다) 경로 / 8주차 세미나 발표자 아님 / Citation Accuracy 채택 / RAGAS 자동 위임 한계 명시 (TestsetGenerator 옵션) / 의료 도메인 예시를 가전 매뉴얼로 재해석. 우선순위 P0/P1/P2 명시. |