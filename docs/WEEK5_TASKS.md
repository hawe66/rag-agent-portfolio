# WEEK 5 — Retrieval 고도화 (Hybrid Search + Re-ranking + Ablation)

> **이 작업은 `PROJECT_CONTEXT.md` Phase 1 (Advanced RAG)의 세 번째 단계다.**
> 시작 전 `PROJECT_CONTEXT.md`와 `docs/week4_retrospective.md`를 반드시 읽는다.
> 이번 주차 발표자는 **아님**. 발표 자료 산출물 없음. 대신 ADR 작성이 필수 산출물에 포함된다.

---

## 0. 이 주차의 핵심 원칙

- **"내 데이터에 가장 잘 맞는 검색 전략을 수치로 증명한다"** — 스터디 가이드의 메시지.
- **Ablation = 한 번에 한 변수**: Hybrid를 한 방에 적용하면 BM25/RRF/Reranker 중 무엇이 효과였는지 구분 불가. 단계별로 쌓아 올리며 비교한다.
- **변수 통제**: chunking과 embedding은 4주차 최종(C3)으로 고정. **retrieval만 바꾼다.**
- **점수 + 가설 + latency**: production 관점에서 latency도 함께 측정. 점수가 떨어진 구성도 "왜 떨어졌는가"를 기록하면 그게 6주차 Agentic RAG의 출발점.

---

## 1. 5주차 Baseline 확정 (선결)

### 1.1 Baseline retriever = 4주차 C3

4주차 ablation에서 종합적으로 가장 균형 잡힌 **C3 (pymupdf4llm + clean_markdown, 339 chunks)** 를 5주차 baseline vector store로 사용한다.

> **주의:** 4주차 표에서 C3는 LIM-005/008 모두 4/5로 "어디서도 1등은 아니지만 골고루 잘함"이었다. B1/C2가 LIM-008에서 5/5였지만 LIM-005에서 약했다. C3를 baseline으로 택하는 근거는 "단일 전략 중 가장 균형적"이라는 점 — 이 reasoning을 ADR에 기록.

만약 C3 vector store가 현재 디스크에 없으면 4주차 chunking_experiments 노트북의 C3 파이프라인을 재실행해서 `data/chroma_db_c3/` 로 다시 구축.

### 1.2 RAGAS 환경 구축 (이번 주차의 진짜 선결 과제)

> **배경:** 4주차에는 RAGAS를 못 돌렸다(환경 이슈 + 시간). 5주차 과제는 RAGAS 4구성 비교가 **필수**다. macOS 14 업그레이드로 torch 2.4+ 가능해졌으니 이제 환경이 풀렸을 가능성이 높다.

**작업:**
1. RAGAS 설치/버전 확인. macOS 14 + torch 2.4+ 환경에서 재시도
2. RAGAS가 요구하는 LLM/embedding 연결 (OpenAI로 통일 권장 — 평가의 일관성)
3. **smoke test**: 질문 2~3개로 RAGAS 3지표(Faithfulness / Answer Relevancy / Context Precision)가 실제로 산출되는지 먼저 확인
4. 성공하면 본 실험으로, 실패하면 §1.4 폴백

> **Context Recall:** 7주차 작업(ground truth context 라벨링)으로 미룸. 단, §1.3 평가셋 확대 시 일부 질문에 ground truth를 미리 달아두면 7주차가 수월해진다 (선택).

### 1.3 평가셋 확대 (필수 — 4주차 한계 직접 해결)

> **배경:** 4주차 평가 쿼리가 4개뿐이라 5%p 차이가 통계적 의미를 갖기 어려웠다. 5주차에서 ablation 비교를 신뢰하려면 평가셋을 늘려야 한다.

**작업:**
- 기존 4~5개 → **최소 15개, 권장 20개**로 확대
- 카테고리/질문유형 균형 (4주차 라벨링 체계 계승):
  - 카테고리: 정수기 / 공기청정기 / 청소기 골고루
  - 질문유형: 단계설명 / 부품역추적 / troubleshooting / 모델간비교 / 주의사항
  - **BM25가 유리할 질문**(고유명사·모델명·숫자: "AS281DAW 필터 수명", "배터리 충전 시간 몇 시간")과 **Dense가 유리할 질문**(의미 기반: "공기가 탁할 때 어떻게 하나요")을 의도적으로 섞는다 → hybrid 효과를 드러내기 위함
- 각 질문에 정답 또는 정답 페이지/모델 라벨 부착 (정성 평가 + 추후 ground truth용)
- 저장: `docs/eval_questions_v2.json` (v1 보존하고 v2로 확대)

### 1.4 폴백 기준

- RAGAS가 또 막히면: 본 실험(§3, §4)을 **정성 평가 + category accuracy**로 먼저 완료하고, RAGAS는 환경 해결 후 별도 추가. 단 이번엔 macOS 14라 가능성 높으니 1~2시간은 진지하게 시도.
- 폴백한 경우에도 ablation 표 구조(§5)는 동일하게 채우되 지표만 정성 기반으로.

---

## 2. 4주차 회고 정리 (선결 산출물)

> 산출물: `docs/week5_retrospective.md` 의 첫 섹션

스터디 과제 1번 항목. 다음을 작성:

1. **4주차 최종 chunking 전략 = C3**, 이걸로 인덱싱한 vector store가 5주차 baseline임을 명시
2. **4주차 이후에도 검색 품질 낮았던 케이스 2~3개** — LIM-005 잔여 실패(C3 4/5의 그 1개), LIM-008 잔여 실패, 그리고 평가셋 확대로 새로 드러난 케이스가 있으면 추가
3. **"왜 retrieval 단계에서 더 개선해야 하는가"** 한 단락 — chunking(4주차)으로 데이터 표현은 개선했지만, 검색 알고리즘 자체(어떻게 찾을 것인가)는 아직 단순 dense similarity뿐이다. 공통 용어 false match(LIM-006)나 고유명사 정확 매칭은 dense만으론 한계. → lexical signal(BM25)과 정밀 재정렬(reranker)이 필요하다.

---

## 3. Hybrid Search 구현 (필수)

> 산출물: `notebooks/week5_hybrid_search.ipynb`

### 3.1 구성 요소

- **Sparse**: BM25 (`rank_bm25` + LangChain `BM25Retriever`)
- **Dense**: 기존 C3 vector store retriever
- **결합**: RRF (Reciprocal Rank Fusion) — LangChain `EnsembleRetriever` 또는 직접 구현

### 3.2 한국어 BM25 토크나이징 주의

> **본 도메인 특이사항:** BM25는 토크나이징에 민감하다. 한국어는 공백 단위로 자르면 조사가 붙어서("필터를", "필터는") 매칭이 약해진다.

- 1차: 단순 공백 토크나이징으로 baseline BM25
- 2차(권장): 한국어 형태소 분석기(`kiwipiepy` 또는 `konlpy`) 또는 최소한 명사 추출로 토크나이징 개선
- 어느 쪽을 썼는지, 차이가 있었는지 기록 → 발표/면접 거리
- **형태소 분석기 설치가 막히면**(자주 막힘) 공백 토크나이징으로 진행하고 한계로 기록. 이것도 좋은 ADR 트레이드오프 항목.

### 3.3 비교

3가지 retriever를 같은 평가셋(§1.3 v2)으로 비교:
- BM25 only
- Dense only
- Hybrid (BM25 + Dense, RRF)

각 구성에서 동일 질문의 top-5를 정성 확인. 특히:
- 고유명사/모델명 질문에서 BM25가 Dense를 이기는지
- 의미 기반 질문에서 Dense가 BM25를 이기는지
- Hybrid가 둘의 장점을 합치는지 (혹은 어중간해지는지)

---

## 4. Re-ranking 구현 (필수)

> 산출물: `notebooks/week5_reranking.ipynb` (또는 hybrid 노트북에 통합)

### 4.1 2-stage 구조

- 1차 검색: Hybrid로 **top-k=20** 확보
- 2차 재정렬: Cross-Encoder reranker로 **top-5** 선별

### 4.2 모델 선택

한국어 도메인이므로:
- 1순위: **`dragonkue/bge-reranker-v2-m3-ko`** (한국어 특화)
- 2순위: `BAAI/bge-reranker-v2-m3` (다국어, 한국어 강함)
- 경량/영어: `cross-encoder/ms-marco-MiniLM-L6-v2` (본 도메인엔 부적합할 듯)

> macOS 14 + torch 2.4+ 환경이면 이제 이 모델들 로컬 실행 가능. LIM-004 해결의 직접적 수혜.

### 4.3 LangChain 연결

`ContextualCompressionRetriever` + `CrossEncoderReranker` + `HuggingFaceCrossEncoder` 조합. base_retriever는 §3의 Hybrid.

### 4.4 적용 전후 비교

동일 질문에서 reranker 적용 전 top-5와 적용 후 top-5의 **순위 변화**를 표로. "어떤 chunk가 몇 위에서 몇 위로 올라왔는가"가 발표/면접 핵심.

---

## 5. Ablation 비교 실험 (필수)

> 산출물: `notebooks/week5_ablation_comparison.ipynb` (또는 위 노트북에 통합)
> 결과 표는 `docs/week5_retrospective.md`에 정리

### 5.1 4구성 비교

| 구성 | Faithfulness | Answer Relevancy | Context Precision | 평균 Latency(s) |
|---|---|---|---|---|
| Dense only | ? | ? | ? | ? |
| BM25 only | ? | ? | ? | ? |
| Hybrid (BM25+Dense) | ? | ? | ? | ? |
| Hybrid + Rerank | ? | ? | ? | ? |

### 5.2 변수 통제 체크리스트

- [ ] chunking: 4주차 C3로 고정
- [ ] embedding: 4주차와 동일 (OpenAI text-embedding-3-small)
- [ ] LLM: gpt-4o-mini, temperature=0.1 고정
- [ ] vector DB: 동일 Chroma 인스턴스
- [ ] 평가셋: §1.3 v2 동일 적용
- [ ] 변경된 것은 retrieval 구성만인가

### 5.3 Latency 측정

production 관점 필수. 각 구성의 질문당 평균 응답 시간. 특히 reranker 적용 시 latency 증가폭 — "정확도 향상 대비 latency 비용"이 ADR 트레이드오프의 핵심.

### 5.4 정성 분석 병행

표 숫자만 보지 말 것. §1.3에서 의도적으로 섞은 "BM25 유리 질문 / Dense 유리 질문"이 실제로 그렇게 갈렸는지 확인.

---

## 6. Error Case 분석 (필수 — 6주차의 출발점)

> 산출물: `docs/week5_retrospective.md` 의 error case 섹션

5주차 최종 구성(아마 Hybrid+Rerank)에서도 **여전히 실패하는 케이스 최소 3개**:

```markdown
### Error Case #1
- 질문: "..."
- 검색된 chunk (실제 top-5):
  - ...
- 정답이 있어야 할 곳:
  - ...
- 왜 실패했는가 (가설):
  - ...
- 다음 단계(Query 변환 / Agentic RAG)에서 어떻게 해결할 수 있을지 (한 줄 가설):
  - 예: "단일 검색으론 부족 — grade 후 query rewrite로 도메인 키워드 추가하면 회수 가능할 듯"
```

> **이 3개가 6주차 Agentic RAG의 입력이다.** WEEK6_TASKS.md의 placeholder를 이걸로 채우게 된다. 따라서 가능한 한 **6주차에서 routing/rewrite로 풀릴 법한 유형**을 의도적으로 고르면 좋다 (예: 첫 검색은 실패하지만 query를 바꾸면 풀리는 케이스).

---

## 7. ADR 작성 (필수)

> 산출물: `docs/adr/week5_retrieval_strategy.md` (한 페이지)
> **이건 우리 프로젝트의 ADR 체계와 직접 연결된다.** `PROJECT_CONTEXT.md`의 ADR 목록에 ADR-008로 등재 예정.

구조:
1. **Decision**: 최종 채택한 retrieval 전략 (예: Hybrid + Rerank)
2. **Context/근거**: 왜 그 전략인가 — RAGAS 수치 + 정성 분석. 본 도메인에서 BM25 vs Dense 어느 쪽이 강했고 왜인지.
3. **Trade-off**: latency, 메모리, 구현 복잡도 측면에서 포기한 것. (예: reranker로 latency 2배지만 정확도 향상이 그만한 가치)
4. **Alternatives**: 검토했으나 안 쓴 것 (예: Dense only — 단순하지만 고유명사 약함 / linear combination — RRF가 점수 스케일 무관해서 선택)

> "왜 이걸 선택했는가"가 면접 핵심 질문. ADR의 Trade-off와 Alternatives 섹션이 그 답변의 재료.

---

## 8. 산출물 체크리스트

### 필수
- [ ] `notebooks/week5_hybrid_search.ipynb` (§3)
- [ ] `notebooks/week5_reranking.ipynb` (§4, 통합 가능)
- [ ] `notebooks/week5_ablation_comparison.ipynb` (§5, 통합 가능)
- [ ] `docs/week5_retrospective.md` (§2, §5 표, §6 error case)
- [ ] `docs/adr/week5_retrieval_strategy.md` (§7)
- [ ] `docs/eval_questions_v2.json` (§1.3, 평가셋 확대)

### 선택 (도전 과제 — 흥미 있으면, 6주차로 미뤄도 됨)
- [ ] Query 변환 (Multi-Query / HyDE / RAG-Fusion 중 1) — **6주차 rewrite_query 빌드업이라 미리 해두면 유리**
- [ ] Reranker 모델 비교 (bge vs ms-marco 정량)
- [ ] RRF 가중치 튜닝 (BM25:Dense = 0.3:0.7 / 0.5:0.5 / 0.7:0.3)
- [ ] Self-Query Retriever (4주차 메타데이터 활용 자동 필터링) — **LIM-005 직접 공격, 강력 추천**

> **Self-Query Retriever 한 마디:** 4주차에 category 메타데이터를 LIM-005 공격용으로 설계해뒀다. Self-Query Retriever는 질문에서 "정수기"를 감지해 category 필터를 자동 생성한다. 이게 LIM-005/006의 가장 직접적 해결책이라 도전 과제 중 ROI가 가장 높다. 시간 되면 우선.

---

## 9. 작업 순서 권장

1. `PROJECT_CONTEXT.md` + `docs/week4_retrospective.md` 다시 읽기
2. **§1.2 RAGAS 환경 smoke test** — 이게 풀려야 나머지가 정량화됨. 막히면 §1.4 폴백 인지하고 1~2시간 시도
3. **§1.3 평가셋 v2 확대** — BM25/Dense 유리 질문 의도적 배치
4. **§2 4주차 회고 정리** — retrospective 첫 섹션
5. **§3 Hybrid (BM25 → Dense → Hybrid 순서로 쌓기)**
6. **§4 Reranking 얹기**
7. **§5 Ablation 4구성 비교 + latency**
8. **§6 Error case 3개** — 6주차 입력이므로 신중히 선정
9. **§7 ADR 작성**
10. (여유) §8 도전 과제 — Self-Query 우선

---

## 10. 6주차 연결 메모

- **§6 error case 3개 = 6주차 Agentic RAG의 입력.** WEEK6_TASKS.md가 이걸 기다린다.
- **§8 Query 변환 도전 과제 = 6주차 rewrite_query 노드의 예행연습.** 해두면 6주차가 빨라진다.
- **§7 ADR = ADR-008.** 6주차 ADR-009(Agentic RAG)와 한 쌍을 이룬다.
- 5주차 최종 retriever가 6주차 `retrieve` 노드에 그대로 들어간다.

---

## 11. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| (초안) | 5주차 가이드 + 4주차 회고(C3 baseline, RAGAS 미측정, 평가셋 4개 한계) 반영. RAGAS 환경 구축 + 평가셋 확대를 선결 과제로 명시. 한국어 BM25 토크나이징, dragonkue 한국어 reranker 등 도메인 특화 반영. |