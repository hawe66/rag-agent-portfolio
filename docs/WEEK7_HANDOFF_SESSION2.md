# WEEK7 — 새 세션(Claude Code) 인수인계 브리프 (Session 2)

> 이 문서는 cold start용이다. 이것만 읽고 이어받을 수 있게 쓴다.
> 선행 문서: `docs/WEEK6_REVISED_TASKS.md`, `docs/WEEK7_REVISED_TASKS.md`, `PROJECT_CONTEXT.md`.
> 작업 방식: 전략/계획은 별도 세션, **이 세션은 구현**. 결정은 ADR로 굳힌다.

---

## 0. 현재 위치 (한눈에)

Phase 1 Closeout rework 진행 중. 완료/미완:

- ✅ **A1/A2** — `src/agent.py` 모듈화 + 23q 재실행 Top-1 95.7% regression 확인.
- ✅ **B** — `src/evaluation.py` 골든셋 로더 + `modality_label` 스키마 + Refusal/Citation 구현.
- ✅ **C1/C2** — `data/eval/golden_set_v2.csv`(41행) 구축 + image-required 8건 다양화(위치2/배치2/방향2/아이콘2) + reference 규약 통일.
- ✅ **D1** — Baseline RAGAS+도메인 메트릭 측정 실행(결과 pkl 저장).
- 🔶 **D1 디버그** — Citation 0% / AR 0.29 원인 점검 완료(§2). **수정·재측정은 미실행.**
- ⬜ **D2~D4** — `week7_retrospective.md` §4~9 채우기 + LIM-002 갱신.
- ⬜ **E1~E4** — ADR-010 보강 + ADR-011 + `PROJECT_CONTEXT.md` 갱신.

---

## 1. 디버그로 확정된 사실 (cold start가 알아야 할 수치)

### 1.1 Citation 0% — regex 정상, 원인은 page 정보 부재

- `_CITATION_RE`는 실제 답변 포맷 3종(`..._OM_WEB.pdf p.1`, `complex.pdf p.1`, `(출처: ... p.1)`) 모두 매칭 → **regex/B3 fix는 정상.**
- 샘플 답변: 대부분 `p.1` 인용, 일부 무인용, 1건만 `p.36`. expected page는 p.8~p.20 → ±2 tolerance로도 불일치.
- **두 갈래 가설(§3 T1에서 반드시 먼저 검증):**
  - (가) 프롬프트가 retrieved doc의 page를 LLM에 안 흘려줌 → LLM이 p.1로 디폴트. **싼 수정.**
  - (나) 더 깊은 원인 의심: `vectorstore.create_vectorstore`가 `by_page=False`(기본) → `parse_pdf`가 문서 전체를 page_num=1로 만듦 → **C3 청크 page가 전부 1.** 이 경우 프롬프트만 고쳐도 소용없고 **by_page=True로 재임베딩 필요.**

### 1.2 AR 0.29 — RAGAS 결함 아님, refusal 평균 효과

`baseline_ragas_results.pkl` 기준 n=37(=41−out_of_scope 4 추정). refused 8 / answered 29. 그룹 분리:

| 지표 | overall | answered-only | refused-only |
|---|---|---|---|
| Faithfulness | 0.737 | 0.872 | 0.250 |
| AnswerRelevancy | 0.290 | 0.370 | 0.000 |
| ContextPrecision | 0.847 | 0.939 | 0.515 |
| ContextRecall | 0.658 | 0.741 | 0.354 |

answered-only AR by q_type: factual 0.360 / multi_hop 0.400 / comparison 0.737(n=1, 무시) / safety 0.236.

해석: (1) refusal 분리 시 Faithfulness 0.87·CP 0.94 → retrieval·grounding은 양호, baseline은 답한 케이스에선 정직히 잘 답함. (2) answered-only AR 0.37 → RAGAS judge가 매뉴얼식 다중 안내/길이에 보수적(judge×도메인 mismatch).

### 1.3 폐기된 가설 (연속성 위해 기록)

처음 보고했던 "Citation regex가 새 출처 포맷을 못 잡음"은 **틀림.** B3 정상 동작. 진짜 원인은 위 §1.1.

### 1.4 주의 — 올바른 결과 아티팩트 사용

디버그가 본 `data/eval/_w7_checkpoint/baseline_answers.pkl`은 **인터럽트 직전 7개 부분 결과**다(본실험 아님). 회고는 **37행짜리 `baseline_ragas_results.pkl`**(및 agentic 대응 결과)을 써야 한다. agentic 결과 pkl 존재 여부부터 확인할 것.

---

## 2. 디버그가 함의하는 결정사항

- **AR 보고:** 단일 평균 금지. **overall / answered-only 2줄**로 보고(refusal 분리). q_type·modality별도 분해.
- **분모 명시:** 41(golden) → RAGAS는 answerable 37, Refusal은 전체 41(또는 out_of_scope+image-required 기준), image-required headline은 8건. 어떤 수치가 어떤 분모인지 표마다 명기.
- **Citation:** §3 T1 검증 결과에 따라 분기(싼 수정 vs 재임베딩). 재임베딩이 필요하면 **비용 때문에 사용자에게 확인 후 진행.**

---

## 3. 새 세션 작업 지시 (P0 → P2)

### T1 (P0) — Citation 원인 확정 후 수정

1. **검증 먼저:** C3 스토어(`data/chroma_db_c3`, collection `lg_manuals_c3`) 청크의 `page` 값 분포 출력.
   - page가 2~40으로 분산 → (가) 경로. page가 전부 1 → (나) 경로.
2. **(가) 싼 수정:** 답변 생성 프롬프트에 chunk별 출처를 명시 주입.
   - `src/evaluation.py`의 `run_rag_pipeline` context 구성: `[source p.N] page_content` 형태로.
   - `src/agent.py` `_build_generate_node`의 context도 동일하게(`[{source} p.{page}] ...`) — **agentic도 같은 버그 공유**.
   - 재측정은 Citation 관련 지표만 재계산하면 되도록(가능하면 답변 재생성 최소화).
3. **(나) 재임베딩 경로:** `create_vectorstore(..., by_page=True)`로 C3 재빌드 필요. **비용 발생 → 진행 전 사용자 확인.** 재빌드 후 (가)의 프롬프트 수정도 함께 적용.
4. **검증:** 수정 후 Citation Accuracy가 0%가 아니라 의미 있는 값으로 나오는지, expected page와 ±2 매칭 사례가 실제로 잡히는지 샘플 3건 육안 확인.
- ADR-010에 "Citation 측정 인프라 버그 발견·수정" 서사로 기록(포트폴리오 강점: eval 인프라 자체를 검증함).

### T2 (P0) — `week7_retrospective.md` §4~9 채우기 (D2~D4)

§1의 디버그 수치를 정직하게 반영:

- **§4 RAGAS 표:** Baseline·Agentic 각각 **overall / answered-only 2줄.** (Agentic 결과 pkl 확인; 없으면 재측정 — 단 §5 비용 고려.)
- **§5 RAGAS 한계 사례 2건:** (a) refusal이 AR 평균을 끌어내리는 효과, (b) 매뉴얼식 다중 안내 답변에 RAGAS judge가 보수적인 케이스. 각각 질문·답변·점수·원인가설·필요보완지표(Evidence Coverage 등).
- **§6 도메인 메트릭:** Refusal Accuracy(TP/FP, FP rate) + Citation(T1 결과). FP refusal 사례(예: "유선/무선 장단점", "AS181DAW vs AS281DAW 차이") 적시 — comparison q_type이 과도히 거절되는 문제.
- **§8 q_type별 표 + §9 modality별 표.** **§9의 헤드라인 = "text-only baseline의 image-required 정확도/실패율"**(WEEK7_REVISED §7). image-required 8건은 대부분 refusal/오답으로 baseline이 실패하는 게 정상이자 cross-modal 기준선. 이 숫자를 명확히.
- **§10 / LIM-002 갱신:** 도면이 vector graphics(raster XObject 추출 불가, 페이지 rasterize→vision 필요)임을 정량 기록. Phase 2 추출 경로 확정 근거.
- **분모 명시**(§2) 전 표에 적용.

### T3 (P1) — ADR + PROJECT_CONTEXT (E1~E4)

- **ADR-010 보강:** modality-aware 평가, Citation 버그·수정, AR 이중 보고 결정.
- **ADR-011 (신규) `adr/ADR-011-crossmodal-eval.md`:** modality 3분류 + 분해 메트릭 + `context_provider` 훅 + LIM-002(vector→rasterize) 근거 + 3단계(이미지 추출·임베딩)가 다음임.
- **`PROJECT_CONTEXT.md` 갱신:** §2 ADR 008~011 등재, §3 sanity check에 LIM-002 정량결과, §7 변경이력 + Phase 1 종료 상태, store=258 docs 불일치(4주차 339와) 기록.

### T4 (P0) — 검증 (서브에이전트 권장)

- 회고의 모든 수치가 결과 pkl에서 재현되는지 재계산 대조.
- 표 내부 정합성(분모·합계), Citation 샘플 육안, image-required 헤드라인이 §9에 존재하는지 체크.

---

## 4. 실행 경제 (비용 주의)

- **재실행 최소화:** answered 답변·contexts는 결과 pkl에 이미 있음 → RAGAS/도메인 재계산은 LLM judge만 필요할 수 있음. 전체 41q agentic 재생성은 latency·$ 큼(agentic ~34s/q). Citation 수정이 (가)면 답변 재생성 없이 출처 부분만 재계산 가능한지 먼저 검토.
- **메모리:** Baseline·Agentic 동시 평가 시 reranker는 **단일 인스턴스 공유 주입**(WEEK6_REVISED §3.2/§3.3: int8 양자화 + thread cap + build-once 적용 상태 확인). VS Code/cpptools 종료 권장.
- (나) 재임베딩은 사용자 확인 후에만.

---

## 5. 파일·규약 레퍼런스

- 코드: `src/agent.py`, `src/evaluation.py`, `src/retrieval.py`, `src/vectorstore.py`, `src/chunking.py`, `src/parsing.py`.
- 데이터: `data/eval/golden_set_v2.csv`(41행), `data/chroma_db_c3`(lg_manuals_c3, 258 docs), 결과 pkl `data/eval/_w7_checkpoint/`(부분) + 본실험 결과(위치 확인).
- 회고/ADR: `docs/week7_retrospective.md`, `adr/ADR-010-evaluation-framework.md`, (신규)`adr/ADR-011-crossmodal-eval.md`.
- **reference 규약:** `reference_context = "model p.N (한글 도면/섹션명)"`, `figure_ref = "model p.N fig:figname"`(풀 경로). image-required 6행을 이 규약으로 정리하는 잔여 작업도 포함(WEEK7_REVISED §6.3 표 참조).

---

## 6. 새 세션 첫 행동

1. 이 브리프 + `WEEK7_REVISED_TASKS.md` 읽기.
2. **T1-1(page 분포 검증)부터** — Citation 원인 (가)/(나) 확정. (나)면 멈추고 사용자에게 비용 확인.
3. 본실험 결과 pkl(agentic 포함) 존재·행수(41/37) 확인.
4. 그 다음 T1 수정 → T2 회고 → T3 ADR → T4 검증.

---

## 7. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-06-16 | Session 1(전략) 작성. D1 디버그(Citation 0%/AR 0.29) 결과 + by_page 의심 + 새 세션 P0~P2 지시 정리. |
