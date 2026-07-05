# Week 9 — 1차 구현 결함 목록 & 재검토 계획

> **상태: week9_evaluation.md의 "findings"는 검증 전이다 — 인용 금지.**
> 1차 구현(rasterize→C1→C2→비교)은 코드·산출물은 만들었으나, **평가 하네스가
> "측정하려는 것"을 실제로 측정하는지 검증하지 않았다.** 아래 결함이 수정되기 전까지
> C1≤C0 / "캡션은 도움 안 됨" / "CLIP은 OOD" 등 결론은 **artifact 가능성**이 높다.
> 다음 세션은 ADR-012를 쓰지 말고, `WEEK9_TASKS.md`를 §3부터 다시 걸으며 아래를 고친다.

## A. 결함 (defects)

### D1 — C0가 ADR-011 baseline(page match 25%)을 재현하지 못함 (가장 근본)
- 측정값: C0 IR8 page-hit **75%**(6/8, ±2·±0 동일). ADR-011 anchor = **25%**.
- `src/week9_eval.py`의 `page_hit`는 **새로 만든 ad-hoc 정의**(같은 매뉴얼+page±tol). ADR-011/Week7이 실제로 쓴 page-match 정의를 확인하지 않고 "더 관대해서 그렇다"고 hand-wave함.
- 결과: C0/C1/C2는 서로는 비교 가능하나 **프로젝트의 기준선(tech_choice가 anchor하는 25%)과 비교 불가** → 비교 실험의 토대가 흔들림.
- **할 일:** Week7/ADR-011의 page-match·model-match 정의를 코드에서 확인 → **동일 정의 재사용**. C0가 25%(±)에 수렴하는지 먼저 확인하고, 그 위에 C1/C2를 얹는다.

### D2 — C1 "캡션 무용" 결론은 사실상 retrieval 실패 artifact (실측으로 확인됨)
- **실측(8 IR 문항, mm 스토어 top-5의 modality 추적):**
  - **정답 page의 캡션이 top-5에 든 경우 = 2/8** (IR-W2, IR-V3)뿐. 나머지 6/8은 정답 캡션이 아예 검색 안 됨.
  - **IR-A3: top-5가 전부 text 청크(캡션 0개).** ← 1차 분석에서 "C1이 캡션으로 답이 달라졌다"고 본 건 **오판**. C1의 "공기제균 아이콘" 답변은 p22 *텍스트* 청크에서 나옴. (답변으로 캡션 사용을 추론하면 안 된다는 증거.)
  - **IR-W1: 캡션 5/5 검색됐지만 전부 엉뚱한 page**(32·18·27·28·20, 정답은 29). 캡션이 떠도 틀린 page가 뜸.
- 진짜 원인 분해: (a) **text cross-encoder reranker(bge-reranker-v2-m3-ko)가 캡션 청크를 정답 단계까지 못 올림** → 캡션 *내용 품질*은 거의 시험조차 안 됨. (b) 떠도 IR-V3는 캡션이 빗금 디테일을 누락(품질 문제, §4.3 spot-check와 일치). → 즉 retrieval 실패가 1차 원인, 캡션 품질은 2차.
- 코드 결함: `docs_to_pages`가 `modality`를 버려 1차 평가 땐 이걸 전혀 몰랐음.
- **할 일:** retrieved doc마다 `modality` 보존 + **caption-hit 지표**("정답 page 캡션이 top-k에 들었나") 1급 지표로 측정. modality-aware retrieval(캡션·텍스트 분리 검색 후 융합, 또는 figure_ref로 page 묶어 동반 회수) 검토. 그 뒤에야 캡션 품질을 논함.

### D3 — C2/CLIP 12% + score 평탄(~0.27)은 "버그" 시그니처일 수 있음
- near-random page-hit + 모든 페이지 score가 0.27 근처로 평탄 = 임베딩 파이프라인 고장 패턴.
- 미검증: image encoder(`clip-ViT-B-32`) ↔ text encoder(`clip-ViT-B-32-multilingual-v1`) **공간 정렬**, 정규화. sanity check(영어 쿼리/자명한 케이스) 0회.
- **할 일:** (1) 영어 쿼리로 명백한 시각 페이지가 상위에 오는지, (2) 한 페이지를 그 내용 묘사 쿼리로 찾는 sanity, (3) score가 쿼리별로 peaked한지. 버그면 고치고, **버그 아님을 확인한 뒤에만** "한국어·도면 OOD" 결론.

### D4 — judge 관대 + modality 라벨 의심
- IR-A2(image-required, GT="앞면 **상단**"): 텍스트-only C0가 "앞면"이라 답했는데 judge=1. 구분 디테일(상단) 누락인데 통과.
- 함의: (a) IR-A2 라벨이 틀렸거나(본문에 위치 있음) (b) judge가 vague 부분일치를 보상. **둘 다 실험을 오염**(text-only가 image-required를 "맞추면" 대조가 무의미).
- **할 일:** IR8 각 문항을 매뉴얼 원본과 대조해 "정말 텍스트엔 없고 도면에만 있는가" 재확인(라벨 audit). judge 루브릭을 핵심-디테일 일치로 엄격화(부분일치 0).

### D5 — 테스트셋 작고 불순
- 대조 3개 중 **Q15는 image-helpful인데 text-only 대조로 분류**. n=3에서 "33%"는 1문항 노이즈.
- **할 일:** 대조군을 순수 text-only로만, 개수 늘림(최소 5~6). image-helpful은 별도 버킷.

### D6 — baseline가 평범한 텍스트 질문에서 실패(미조사)
- Q08(text-only, ref airpurifier_complex p.10): C0가 p2/p25 검색, page_hit=False, judge=0. Week7 스택이 맞춰야 할 질문.
- 함의: `reference_target`의 `reference_context` 파싱이 틀렸거나, 본 하네스의 retriever가 Week7과 다르게 구성됨.
- **할 일:** 하네스가 **Week7 known-good 수치를 재현**하는지 먼저 회귀 확인(이게 D1과 함께 신뢰의 토대).

### D7 — C2b 교란(confounded)
- TPM 회피로 `max_images=1`. C2b 0%는 C2a 12% 검색실패의 통과일 뿐, late-fusion 자체를 검증 못함.
- **할 일:** CLIP 검색을 고친(D3) 뒤, page-hit가 성공한 부분집합에서만 C2b answer를 봐야 late-fusion 효과가 보임.

## B. 다음에 할 일 (재검토 순서)

> **[2026-07-05] 이 절은 `docs/WEEK9_REWORK_PLAN.md`로 대체됨** — 전략 세션 리뷰에서 진단 프레이밍 5건 보정(D1=layer 착오, 대조군 Q22 silent drop으로 사실상 Q01 1개, C2a 0%=하드코딩이라 헤드라인 결론은 "미증명", judge self-preference, 캡션-only arm). §A 결함 목록 자체는 유효.

> ADR(예: ADR-012)는 **쓰지 않는다**. 결론이 검증 전이므로. `WEEK9_TASKS.md`를 다시 걸으며 위 결함을 고치고, 그 뒤에 결론을 다시 쓴다.

1. **신뢰 토대 먼저 (D1·D6):** Week7/ADR-011의 page-match·model-match 정의를 코드에서 찾아 `week9_eval`이 **동일 정의를 재사용**하게 바꾼다. C0가 ADR-011 baseline(image-required page≈25%, model≈50%)을 재현하는지 확인. 재현 못 하면 그 이유(reference 파싱/retriever 구성)부터 수정.
2. **C1 계측 (D2):** retrieved에 `modality` 보존 → caption-hit 지표 추가. 캡션이 top-k에 드는지부터 측정. (이미 실측: 정답캡션 top-5 진입 2/8.) modality-aware retrieval 도입 검토 후에야 answer를 논한다.
3. **C2 sanity (D3):** CLIP 파이프라인 버그 여부를 영어/자명 케이스로 확인. 버그면 수정.
4. **라벨·judge audit (D4):** IR8을 원본 매뉴얼과 대조해 라벨 검증, judge 루브릭 엄격화.
5. **테스트셋 정리 (D5):** 순수 text-only 대조 확대, image-helpful 분리.
6. **(C2b는 D3 이후) (D7).**
7. 위가 끝난 뒤 **`week9_evaluation.md`를 재작성**하고, 그때 비로소 결론의 ADR화 여부를 판단.

## C. 변경 이력에 적을 것 (changelog OK)
- CLAUDE.md §8: "9주차 1차 구현(rasterize/C1/C2/비교 하네스·노트북) 완료. **단 평가 하네스 미검증 — C0가 ADR-011 baseline 미재현(75% vs 25%), C1 정답캡션 top-5 진입 2/8(검색실패가 1차원인), CLIP 12% 버그의심.** findings 보류, `docs/WEEK9_DEFECTS.md` 기준으로 §3부터 재검토 예정. ADR-012는 검증 후로 연기."

## D. 1차 구현 산출물 (재사용 가능, 버리지 말 것)
- `src/rasterize.py`(252p PNG @95dpi, IR8 검증됨) · `src/multimodal.py`(캡션 캐시 `data/week9_captions.json` 252건, `data/chroma_db_mm` 590docs) · `src/clip_index.py`(`data/clip_index`) · `src/week9_eval.py`(하네스 — D1·D2·D7 수정 대상) · `w9/week9_multimodal_rag.ipynb`(실행됨) · `data/week9_results.json`.
- 캡션은 비싼 one-time(고해상 ~26k tok/장, 200k TPM에서 ~7/분). **캐시 키는 (category,complexity,page)** — 재캡션 금지.
