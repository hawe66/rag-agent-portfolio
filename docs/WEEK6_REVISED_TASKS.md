# WEEK 6 REVISED — Agentic RAG 정직한 마무리 + 모듈화 (Phase 1 Closeout, 1단계)

> **이 문서는 `WEEK6_TASKS.md`를 대체(supersede)한다.** 기존 명세는 "Agentic 첫 구현"용이었고, 1회 sprint로 구현·실행까지 됐으나 **회고가 비어 있고(§2~4 TBD) 코드가 노트북에만 있다.** 이 문서는 그 미완을 닫고, Phase 2(cross-modal) 재사용이 가능하도록 모듈화하는 것이 목표다.
> 시작 전 `PROJECT_CONTEXT.md`, `docs/week5_retrospective.md`, `docs/week6_retrospective.md`(§1만 차 있음), `docs/week7_retrospective.md`(§1.1에 Agentic 실측치 있음)를 읽는다.
> **경로 주의:** 실제 repo는 `w6/`·`w7/`·`adr/ADR-xxx-*.md`·`docs/weekN_retrospective.md` 구조다. (기존 명세의 `notebooks/`·`docs/adr/`는 오기.)

---

## 0. 현재 상태 진단 (이 sprint의 출발점)

이미 된 것:
- `w6/week6_agentic_rag.ipynb` — LangGraph 4노드(retrieve → grade_documents → rewrite_query(Self-Query 통합) → generate, retry 2회 후 거절) 구현 및 실행 완료.
- **모듈화 완료** — `src/agent.py`에 `GraphState`·`make_filterable_hybrid_retriever`·`build_agent_graph`·`run_agent` 존재, 노트북이 이를 import. (당초 "노트북에만 존재"로 봤으나 실제로는 이미 분리돼 있음.)
- `data/week6_agentic_rag_results.json` — 실행 결과 존재.
- 실측치(`week7_retrospective.md §1.1`): **Top-1 95.7%(22/23), 22q 기준 100%, latency 34.46s, retry 분포 retry0=19/retry1=1/retry2=3, 유일 실패 Q17.**

안 된 것 (이번에 닫는다):
1. **메모리/런타임 비효율로 재실행이 머신을 다운시킴 (현재 진행 블로커).** 원인 §3. 회고를 재실측치로 채우려면 재실행이 안정적이어야 하므로 이게 선결.
2. `week6_retrospective.md` §2~4가 전부 **TBD** — Baseline vs Agentic 비교표, 운영 회고, Q20 해결 여부, regression, 불필요 retry 통계가 비어 있음.
3. **ADR-009 파일 부재** — `adr/`에 008·010만 있고 009 없음.
4. **store 문서 수 불일치** — 노트북 실행 시 `lg_manuals_c3` = 258 docs인데 4주차 회고는 339(C3). 회고에 현행 수치로 정정 기록.

---

## 1. 이 sprint의 핵심 원칙

- **숫자를 다시 만든다, 베껴오지 않는다.** §1.1의 실측치는 노트북 1회 실행 결과다. 모듈화 후 재실행하여 같은 숫자가 나오는지(재현성) 확인하고, 회고는 재실행 산출물 기준으로 채운다.
- **모듈화는 리팩토링이지 재설계가 아니다.** 노트북의 노드 로직을 그대로 `src/agent.py`로 옮긴다. 동작이 바뀌면 안 된다(regression = Top-1 95.7% 유지).
- **Phase 2 자리만 비워둔다, 구현은 안 한다.** `GraphState`에 이미지 컨텍스트가 들어올 자리를 주석/Optional 필드로 남기되 이번엔 텍스트만 흐른다 (CLAUDE.md 단순성 원칙 — 지금 안 쓰는 코드는 만들지 않는다).

---

## 2. 우선순위 트랙

| 우선순위 | 작업 | 산출물 | 비고 |
|---|---|---|---|
| **P0** | 런타임/메모리 하드닝 (BM25 build-once + reranker 양자화 + thread cap) | `src/agent.py`·`src/retrieval.py` | §3 |
| **P0** | 재실행 → regression 확인 (Top-1 95.7% 유지) + 메모리 측정 | `data/week6_agentic_rag_results.json` 갱신 | §4 |
| **P0** | `week6_retrospective.md` §2~4 실측치로 채움 | retrospective | §5 |
| **P0** | ADR-009 작성 + `PROJECT_CONTEXT.md` 등재 | `adr/ADR-009-agentic-rag.md` | §6 |
| **P1** | 불필요 retry / Q20 / Q17 routing 심화 분석 | retrospective §4.x | §5.3 |

> §3 하드닝이 선결이다. 현재 재실행이 16GB 머신을 다운시켜 회고를 채울 수 없다. 7주차(Baseline vs Agentic 동시 평가)도 같은 retriever를 쓰므로 이 수정의 수혜를 받는다.

---

## 3. 런타임/메모리 하드닝 (P0 — 재실행 선결)

> 산출물: `src/agent.py`·`src/retrieval.py` 수정. **로직(검색 결과)은 동일해야 하고 메모리·속도만 개선.**

원인 진단(코드 읽기로 확정):
1. **BM25 인덱스를 매 검색마다 재생성.** `src/agent.py`의 `make_filterable_hybrid_retriever.retrieve()`가 호출마다 `BM25Retriever.from_documents(all_docs, ...)` 실행(line 96·98, 필터 유무 무관). 23문항×재시도 = 30~40회 전체 코퍼스(258 docs) 재토크나이즈 → retrieve 11~15s의 주범 + 메모리 출렁임.
2. **reranker fp32 ~2.2GB 상주.** `dragonkue/bge-reranker-v2-m3-ko`(XLM-R large ~568M). python 4.73GB의 절반.
3. **16GB 머신 포화.** Claude 앱+VM+VS Code C++ 툴링까지 14.79GB 사용 → 13분 평가가 swap 폭증 → 다운. (즉시 완화: VS Code/cpptools 종료, 커널 재시작 후 모델 로드 셀 1회만.)

### 3.1 Fix A — BM25 build-once (가장 큰 효과)

factory 진입 시 전체 코퍼스 BM25를 **1회** 만들어 클로저로 재사용. 카테고리 필터는 카테고리별 BM25를 1회씩 캐시.

```python
def make_filterable_hybrid_retriever(vectorstore, all_docs, reranker, first_stage_k=20, top_k=5):
    bm25_all = BM25Retriever.from_documents(all_docs, k=first_stage_k)   # ← 1회만
    bm25_by_cat: dict[str, BM25Retriever] = {}                           # 필터용 캐시
    def retrieve(query, metadata_filter=None):
        ...
        if metadata_filter:
            cat = metadata_filter.get("category")
            if cat not in bm25_by_cat:
                docs_c = [d for d in all_docs if d.metadata.get("category") == cat]
                bm25_by_cat[cat] = BM25Retriever.from_documents(docs_c, k=first_stage_k)
            bm25 = bm25_by_cat[cat]
        else:
            bm25 = bm25_all
        bm25_docs = bm25.invoke(query)
        ...
```
- 같은 문서·같은 k면 결과 동일 → **regression 없음.**

### 3.2 Fix B — reranker int8 동적 양자화 + CPU 고정 + thread cap

`src/retrieval.py`의 `create_reranker`에서 CPU 로드 + Linear int8 동적 양자화 → RAM ~2.2GB→~0.8~1GB, CPU 추론도 가속.

```python
def create_reranker(model_name="dragonkue/bge-reranker-v2-m3-ko", quantize=True, num_threads=4):
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(num_threads)
    ce = CrossEncoder(model_name, device="cpu")
    if quantize:
        ce.model = torch.quantization.quantize_dynamic(
            ce.model, {torch.nn.Linear}, dtype=torch.qint8)
    return ce
```
- 양자화로 score가 미세하게 달라질 수 있음 → 재실행 Top-1 95.7% 유지 확인. 어긋나면 `quantize=False`로 폴백하고 Fix A+C만 적용.

### 3.3 Fix C — 부차

- `first_stage_k` 20→10 검토(rerank 쌍 절반; Top-5 유지 확인 후 적용).
- 평가 루프에서 문항마다 `gc.collect()` 1회.
- reranker 단일 인스턴스를 7주차 Baseline·Agentic에 **공유 주입**(두 번 로드 금지).

### 3.4 검증 기준

- 재실행 중 python RSS가 안정적으로 **~1.5GB 이하** 유지(`psutil` 또는 Activity Monitor로 before/after 측정, 회고 §3에 기록).
- retrieve 평균 latency 유의미 하락(11~15s → 한 자리 초반 목표).
- **Top-1 95.7%(22/23) 재현** — 하드닝이 정확도를 바꾸지 않았음을 확인.

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

1. 진단 문서 읽기(§0 대상 파일들) + 노트북 현 상태 확인. (즉시 완화: VS Code/cpptools 종료, 커널 재시작.)
2. §3 런타임/메모리 하드닝(Fix A→B→C) 적용.
3. §4 재실행 + regression 확인(Top-1 95.7% 재현) + RSS before/after 측정.
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
| 2026-06-14 | `WEEK6_TASKS.md` supersede. 1회 sprint 후 미완(회고 §2~4, ADR-009) 마감으로 재정의. 실제 repo 경로(w6/, adr/ADR-xxx) 반영. |
| 2026-06-15 | 모듈화는 이미 완료(`src/agent.py` 존재) 확인 → "모듈화" 항목 제거. 재실행이 16GB 머신을 다운시키는 메모리 문제 진단(BM25 매 호출 재생성 + reranker fp32 2.2GB) → §3을 "런타임/메모리 하드닝(BM25 build-once + int8 양자화 + thread cap)"으로 교체, 재실행의 선결로 지정. store 258 docs 불일치 기록 추가. |
