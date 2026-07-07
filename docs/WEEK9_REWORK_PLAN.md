# WEEK 9 재작업 계획 (Rework Plan)

> **[2026-07-07 완료]** 전 Phase 실행 완료. 결과·수동채점·결론은 `docs/week9_evaluation.md`(재작업판) 참조. 이 문서는 실행 근거 기록으로 보존.

> **이 문서가 `WEEK9_DEFECTS.md` §B(재검토 순서)를 대체한다.** 결함 목록(§A)은 유효하나,
> 전략 세션 리뷰(2026-07-05)에서 진단 프레이밍 5건이 보정되었다(§0). 결함 코드(D1~D7)는 그대로 참조한다.
> 원칙 재확인: **ADR-012는 Phase 3 완료 전 금지. 재캡션 금지(캐시 키 = (category,complexity,page)). 1차 findings 인용 금지.**

---

## 0. 진단 보정 — 계획이 1차 결함 문서와 다른 지점

### 0.1 헤드라인 결론은 "artifact"가 아니라 **아직 미증명**

"추론 필요시 VLM" 가설의 증거 구조를 코드로 확인한 결과:

- **C2a answer 0%는 측정이 아니라 정의다.** `week9_eval.py`의 `_run_clip_configs`가 judge=0을 하드코딩(설계상 읽기 단계 없음). "CLIP 검색만으로는 못 푼다"를 이 수치로 주장할 수 없다.
- 가설의 **유일한 유효 증거 arm은 C2b(CLIP 검색+VLM 읽기) vs C1**인데, C2b는 CLIP 검색 고장 의심(D3) + `max_images=1`(D7)로 이중 교란.
- C1의 answer는 검색 실패(D2)에 가려져 캡션 품질이 시험조차 안 됨.

→ 현재 가설을 지지하거나 기각하는 유효 데이터는 **0건**. 재작업의 목표는 "수치 교정"이 아니라 **실험을 처음으로 성립시키는 것**이다.

### 0.2 D1은 baseline 미재현이 아니라 **layer 착오**

- week9 `page_hit` = **retrieval 층** (top-5에 정답 page±2가 있는가).
- ADR-011의 "page match 25% / model match 50%" = **citation 층** (`src/evaluation.py:304 citation_accuracy` — **답변이 인용한** page가 맞는가, category+page±2).
- 75% vs 25%는 모순이 아니라 예상되는 격차: *검색은 정답 페이지를 자주 찾지만(75%), 답변이 그 페이지를 인용/사용하지 못한다(25%)*.
- 따라서 할 일은 "C0를 25%에 수렴시키기"가 아니라: ① C0 답변에 Week7 citation 코드를 그대로 돌려 citation 층에서 ~25%/50%가 재현되는지(회귀 앵커) 확인, ② retrieval 층 지표는 별도 이름으로 라벨링해 25% 앵커와 직접 비교하지 않는 것.

### 0.3 대조군은 "작고 불순"(D5)이 아니라 **사실상 부재**

`TEXT_ONLY_CONTRAST_IDS = (Q01, Q08, Q15, Q22)`의 실제 상태 (golden_set_v2 로더로 확인):

| id | 상태 |
|---|---|
| Q22 | **v2에 존재하지 않음** → `if i in by_id`에서 **조용히 탈락** (n=3의 원인, 하네스가 경고 안 함 = 그 자체가 결함) |
| Q15 | **image-helpful** (text-only 아님) |
| Q08 | text-only지만 **baseline이 실패** (D6 미조사) |
| Q01 | 유일한 유효 대조 |

→ 유효한 순수 text-only 대조 = **1문항**. v2에는 순수 text-only가 27개 있으므로 재구축 여지는 충분.

### 0.4 judge 결함에 self-preference 추가

judge 모델 = 답변 모델 = **gpt-4o-mini 동일** (`week9_eval.py:286-287`). D4의 관대함에 self-preference가 겹친다. IR8은 n=8이므로 **루브릭 엄격화 + 8문항×구성별 수동 채점**이 가장 싸고 확실한 신뢰 확보책.

### 0.5 캡션은 "무용"이 아니라 미시험 + **희석(harm) 가능성**

mm 스토어는 텍스트+캡션을 한 인덱스에 평면적으로 섞는다. 캡션이 top-5에 못 들 뿐 아니라(caption-hit 2/8), 엉뚱한 캡션이 들어가면 유효한 텍스트 청크를 밀어내 **C1 < C0(희석)** 를 만들 수 있다(실측: IR-A1/A2에서 C0 hit→C1 miss). caption-hit 지표에 더해 **캡션-only arm**(텍스트 distractor 없이 캡션만 검색)을 두면 reranker 간섭과 캡션 내용 품질을 분리할 수 있다.

---

## 1. 산출물 현황 (재사용 지도)

| 산출물 | 상태 | 재작업 |
|---|---|---|
| `src/rasterize.py` + `data/sample_images/` (252p PNG) | 검증됨 (IR8 페이지 존재 확인) | 없음 |
| `data/week9_captions.json` (캡션 252건) | 유효, 비싼 one-time | **재생성 금지** |
| `src/multimodal.py` + `data/chroma_db_mm` (590 docs) | 스토어 자체는 유효 | T6에서 캡션-only 검색 경로 추가 |
| `src/clip_index.py` + `data/clip_index` | **버그 의심** (D3) | T8 sanity → 필요시 수정·재빌드 |
| `src/week9_eval.py` (비교 하네스) | **주 수정 대상** (D1·D2·D4·D5·D7 + silent drop) | T2~T7, T9 |
| `data/week9_results.json` | 1차 실행 결과 — **인용 금지** | Phase 3에서 재생성 |
| `docs/week9_evaluation.md` | findings 미검증 — **인용 금지** | T10에서 재작성 |
| `w9/week9_multimodal_rag.ipynb` | 실행됨 | Phase 3에서 재실행 |

---

## 2. 작업 목록

> 순서 원칙: 신뢰 토대(Phase 0) 없이는 어떤 수치도 의미가 없으므로 **Phase 0 완료 전 answer 수치를 논하지 않는다.**
> D4의 라벨 audit는 1차 문서의 4번째에서 **Phase 0으로 승격** — IR8 라벨이 틀리면 실험 전제 자체가 무너지기 때문.

### Phase 0 — 신뢰 토대

**T1. IR8 라벨 audit (D4a) — 가장 먼저, 수작업 ~30분**
- IR8 각 문항을 매뉴얼 원본(PDF/PNG)과 대조: "정답 정보가 정말 본문 텍스트에 없고 도면에만 있는가?"
- 특히 IR-A2 (GT="앞면 상단" — text-only C0가 "앞면"으로 통과한 케이스): 본문에 위치 서술이 있으면 image-helpful로 강등.
- 대조군 후보(T4)도 같은 방식으로 "정말 텍스트만으로 충분한가" 확인.
- **산출:** 문항별 audit 표(문항, GT, 본문에 있는가, 판정: 유지/강등/수정) → `docs/week9_evaluation.md` 재작성 시 부록.
- **검증:** IR8 전 문항에 명시적 판정 기록. 강등 문항이 생기면 IR-n을 갱신하고 이후 모든 수치는 갱신된 셋 기준.

**T2. 지표 층 분리 + citation 회귀 앵커 (D1 보정판)**
- `page_hit` → `retrieval_page_hit@5`로 개명하고 docstring·결과 JSON·표에 "retrieval 층, ADR-011 citation 층과 비교 금지" 명시.
- C0 answer 경로에 Week7 citation 인프라를 정합: 답변 프롬프트에 Week7의 `[source p.N]` 인용 태그 지시 추가(현 `ANSWER_PROMPT`에는 없음 — `_CITATION_RE`가 매칭할 태그가 답변에 안 나올 수 있음) → C0 답변에 `citation_accuracy`(src/evaluation.py:304)를 그대로 실행.
- **검증:** C0 IR8 citation 층 수치가 ADR-011 앵커(page ≈25%, model ≈50%)에 ±1문항(12.5%p) 내로 재현. 재현 실패 시 원인(프롬프트/retriever 구성/파싱)을 규명하기 전에 다음 단계 진행 금지.

**T3. Q08 실패 조사 + Week7 회귀 (D6)**
- Q08(text-only, airpurifier_complex p.10)에서 C0가 p2/p25를 검색한 원인 규명: `reference_context` 파싱 vs retriever 구성(Week7과 first-stage k, 스토어, reranker 설정 diff).
- **검증:** week9 하네스의 C0 구성이 Week7 known-good과 동일함을 구성 diff로 확인하고, Q08의 실패가 (a) 하네스 결함이면 수정, (b) 진짜 retrieval 실패면 그대로 기록.

**T4. 대조군 재구축 + silent drop 제거 (D5 + §0.3)**
- `week9_eval.py`의 id 선택이 golden set에 없는 id를 만나면 **경고가 아니라 실패**하도록 수정 (silent `if i in by_id` 제거).
- 순수 text-only 27개 중 **5~6개** 선정: 3개 카테고리 커버 + T1 audit 통과분만. Q15는 image-helpful 버킷으로 이동(별도 보고 또는 제외).
- **검증:** 하네스 실행 로그에 "요청 id N개 = 로드 N개" 일치. 대조군 전원이 audit 통과 text-only.

### Phase 1 — C1 계측 (retrieval을 answer보다 먼저)

**T5. modality 보존 + caption-hit 지표 (D2)**
- `docs_to_pages`가 `modality`를 보존하도록 수정. per-question 결과에 retrieved 문서별 modality 기록.
- **`caption_hit@5`** ("정답 page의 캡션 청크가 top-5에 들었는가")를 1급 지표로 추가.
- **검증:** 1차 실측(정답캡션 top-5 진입 2/8)이 재현되는지 확인 — 계측 자체의 회귀 테스트.

**T6. 캡션-only arm (C1-cap, §0.5 신규)**
- 캡션 청크만으로 검색하는 arm 추가(텍스트 distractor 없음 — mm 스토어에서 modality 필터 또는 캡션 전용 검색). 동일 Hybrid+Rerank, 동일 top-5.
- 목적: reranker 간섭(캡션이 텍스트에 밀림)과 캡션 내용 품질을 **분리**. C1-cap의 caption_hit이 높은데 C1(혼합)에서 낮으면 → 간섭 문제. C1-cap에서도 낮으면 → 캡션 텍스트 자체가 질문과 정렬 안 됨.
- answer까지 돌리면 "캡션이 실제로 답을 담았는가"를 처음으로 직접 시험.
- **검증:** IR8에 대해 C1-cap의 caption_hit@5·answer가 기록되고, C1(혼합) 대비 분해표가 나옴.

**T7. judge 재설계 (D4b + §0.4)**
- judge 모델을 답변 모델(gpt-4o-mini)과 **다른 모델**로 분리.
- 루브릭 엄격화: GT의 구분 디테일(위치 수식어·모양·수치)이 빠지면 0. 부분일치 보상 금지.
- **IR8 × 전 구성 답변을 수동 채점**(n≈32, 문항당 수분)하여 judge와 대조 — judge-인간 불일치율을 보고.
- **검증:** IR-A2("앞면"만 답한 케이스)가 새 루브릭에서 0으로 채점됨. judge-인간 불일치 목록 기록.

### Phase 2 — C2 sanity

**T8. CLIP 파이프라인 sanity check (D3)**
- (1) 영어 쿼리로 시각적으로 자명한 페이지(표지·큰 도면)가 상위에 오는가, (2) 한 페이지를 그 내용 묘사 쿼리로 재검색하면 그 페이지가 1위인가, (3) score 분포가 쿼리별로 peaked한가(현재 ~0.27 평탄 = 고장 시그니처).
- 점검 포인트: image encoder(`clip-ViT-B-32`) ↔ text encoder(`clip-ViT-B-32-multilingual-v1`) 공간 정렬, 정규화, 이미지 전처리.
- **검증:** sanity 3종 통과 → "한국어·도면 OOD" 결론 허용. 실패 → 버그 수정 후 인덱스 재빌드(로컬이라 저렴).

**T9. C2b 재설계 (D7)**
- C2a의 answer는 0%가 아니라 **"n/a (설계상 읽기 단계 없음)"** 로 보고 — 하드코딩 0을 측정치처럼 표에 싣지 않는다.
- C2b answer는 **retrieval_page_hit 성공 부분집합에서만** 해석(검색 실패의 통과가 아니라 late-fusion 자체를 시험). `max_images=1` 제약(TPM)은 결과표에 명시하고, 가능하면 top-3 이미지로 소규모 재확인.
- **검증:** C2b 보고가 "page-hit 성공 조건부 answer"로 분리 집계됨.

### Phase 3 — 재실행 · 재작성 · 판정

**T10. 전체 재실행 + `week9_evaluation.md` 재작성**
- Phase 0~2 수정 반영 후 C0 / C1 / C1-cap / C2a(retrieval만) / C2b 재실행 → `data/week9_results.json` 재생성.
- 결과표를 **층별로 분리**: retrieval 층(retrieval_page_hit@5, caption_hit@5), citation 층(Week7 정의), answer 층(엄격 judge + 수동채점 대조).
- 헤드라인 가설의 판정 기준을 사전에 명시: **"추론 필요시 VLM"의 유효 증거 = (검색이 성공한 조건에서) C2b vs C1-cap/C1의 answer 차이 + IR 문항에서 캡션/이미지 없이는 실패함을 보이는 대조.** 데이터가 부족하면 "미증명"으로 정직하게 기록.
- **검증:** WEEK9_TASKS §10 공정성 체크(동일 테스트셋·answer 모델·judge) + 노트북 재현.

**T11. ADR-012 여부 판단**
- T10의 결론이 검증된 뒤에만 ADR화 여부 결정. 결론이 "미증명"이면 ADR 없이 10주차(crop·modality-aware retrieval) 입력으로만 넘긴다.

---

## 3. 성공 기준 (Phase별 gate)

| Gate | 기준 |
|---|---|
| Phase 0 통과 | IR8 라벨 audit 완료 + C0 citation 층이 ADR-011 앵커 재현(±1문항) + Q08 원인 규명 + 대조군 5~6개(전원 audit 통과) + silent drop 제거 |
| Phase 1 통과 | caption_hit@5 계측 재현 + C1-cap arm 동작 + 새 judge에서 IR-A2 재채점 0 + 수동채점 대조표 |
| Phase 2 통과 | CLIP sanity 3종 판정(버그 수정 또는 OOD 확정) |
| Phase 3 통과 | 층별 분리 결과표 + 사전 명시된 판정 기준으로 헤드라인 가설 판정(지지/기각/미증명) |

## 4. 하지 말 것

- 재캡션 (캐시 재사용 — `data/week9_captions.json`, 키=(category,complexity,page))
- Phase 3 전 ADR-012 작성
- 1차 `week9_evaluation.md`·`week9_results.json` 수치 인용
- retrieval 층 수치와 ADR-011 citation 층 수치(25%/50%)의 직접 비교
- C2a의 하드코딩 answer 0%를 측정치로 보고

## 5. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-07-05 | 신규. WEEK9_DEFECTS §B를 대체. 전략 세션 보정 5건 반영: ① D1=layer 착오(citation 회귀 앵커 + retrieval 층 분리), ② 대조군 사실상 Q01 1개(Q22 silent drop), ③ 헤드라인 결론 "미증명"(C2a 0%=하드코딩), ④ judge self-preference + IR8 수동채점, ⑤ 캡션-only arm으로 간섭/품질 분리. |
