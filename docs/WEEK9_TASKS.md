# WEEK 9 TASKS — 멀티모달 RAG MVP (Phase 2 착수, cross-modal 비교 실험)

> 본 과제(cross-modal)의 첫 구현 주차. 목표는 "많이 구현"이 아니라 **두 cross-modal 파이프라인(C1 캡션 / C2 CLIP)을 baseline(C0) 대비 통제 비교**해 "추론이 필요한 멀티모달 파싱엔 VLM이 필요하다"를 데이터로 보이는 것.
> 선행: `docs/week9_tech_choice.md`, `adr/ADR-011-crossmodal-eval.md`, `docs/WEEK7_RUN2_RETROSPECTIVE.md`(§9 기준선).
> **경로 주의:** 실제 repo는 `w9/`(노트북)·`src/`·`data/` 구조. 과제 안내문의 `notebooks/`는 `w9/`로 대응.

## 0. 결정사항 (확정 — 임의 변경 금지)

- 트랙: 멀티모달 RAG(옵션 A). 멀티에이전트 아님.
- 답변/Vision 모델: **gpt-4o-mini** (caption + answer 공용).
- 이미지 추출: **전체 페이지 rasterize** (95dpi). 영역 crop은 10주차.
- CLIP(C2 비교군): **`sentence-transformers/clip-ViT-B-32-multilingual-v1`**(텍스트) + `clip-ViT-B-32`(이미지) joint space. 로컬.
- 평가 대상: **image-required 8건 + text-only 대조 일부**, C0/C1/C2 비교.
- 공정성: 결과 정직 기록. CLIP이 retrieval에서 선전하면 "CLIP=찾기 / VLM=읽기" 분업 결론으로 적는다. 가설 강제 금지.

## 1. 핵심 원칙

- **재사용 우선:** 기존 Chroma·Hybrid+Rerank·`src/agent.py`·`src/evaluation.py`·`context_provider` 훅·golden_set_v2·`figure_ref`를 최대한 그대로. 새 인프라는 cross-modal 인덱스 2개뿐.
- **비교는 동일 조건:** C0/C1/C2 모두 같은 테스트셋·같은 answer 모델(gpt-4o-mini)·같은 채점(LLM-judge)으로. 차이는 "시각정보가 들어오는 방식"뿐.
- **단순성:** late-fusion·crop·OCR 등은 P1/P2. MVP는 C1·C2 최소 경로 + 비교표.
- **비용:** 캡션·임베딩은 one-time(오프라인). 쿼리 단계 VLM 호출 최소화. 비용 우려 시 §2 fallback.

## 2. 우선순위 트랙

| 우선순위 | 작업 | 산출물 | 절 |
|---|---|---|---|
| **P0** | 페이지 rasterize 파이프라인 | `data/sample_images/`, `src/rasterize.py` | §3 |
| **P0** | C1: VLM 캡션 → cross-modal 인덱스 | `src/multimodal.py`(caption), `data/chroma_db_mm/` | §4 |
| **P0** | C2: CLIP 이미지 인덱스 + 검색 | `src/clip_index.py`, `data/clip_index/` | §5 |
| **P0** | C0/C1/C2 비교 평가 (image-required 8 + 대조) | `docs/week9_evaluation.md`, `data/week9_results.json` | §6 |
| **P0** | 노트북(전 파이프라인 묶음 + 테스트≥3) | `w9/week9_multimodal_rag.ipynb` | §7 |
| **P1** | C2 late-fusion(검색 페이지 VLM 읽기) 정교화 | 노트북/`src/multimodal.py` | §5.3 |
| **P1** | week9_tech_choice 평가절 채움 + 정성 평가 | `docs/week9_tech_choice.md` | §8 |
| **P2** | 표→markdown, 도면 crop, OCR, citation 연결 | (도전 과제) | §9 |

> **비용 fallback:** 전 매뉴얼(≈252p) 캡션이 부담이면, 우선 **복합 3종(air_C·water_C·vacuum_C) 전 페이지 + 단순 3종의 도면 페이지**만 인덱싱. 단 retrieval 비교의 distractor 확보를 위해 테스트 대상 매뉴얼의 전 페이지는 포함할 것.

## 3. 페이지 Rasterize (P0)

> 산출물: `src/rasterize.py`, `data/sample_images/{category}_{complexity}/p{NNN}.png`

- PyMuPDF로 6개 PDF의 페이지를 95dpi PNG로 렌더(파일명에 page 보존). audit서 95dpi 도면 가독성 확인됨.
- 메타데이터(파일명→category/complexity, page)는 `src/parsing.parse_filename` 재사용. vaccumcleaner typo 처리 기존 로직 사용.
- 검증: 6개 매뉴얼 페이지 수만큼 PNG 생성, IR8 참조 페이지(air_C p18·p22, water_C p29, vacuum_C p15·p19)가 존재하는지 확인.

## 4. C1 — VLM 캡션 → cross-modal 인덱스 (P0)

> 산출물: `src/multimodal.py`(`caption_page`, `build_mm_store`), `data/chroma_db_mm/`

### 4.1 캡션 생성
- gpt-4o-mini(vision)로 각 페이지 이미지를 **구조화 캡션**으로 변환. 프롬프트 요지:
  - 도면에서 보이는 **부품 위치(상/하/좌/우/앞/뒤), 조작·끼움·회전 방향(화살표), 제어판 아이콘/기호 모양, 표·수치**를 한국어로 기술.
  - **보이는 것만** 기술, 추측 금지("이미지에서 확인 불가" 허용) — 거절 원칙과 일관.
- 캡션은 "image-derived 청크"로 변환: `page_content=캡션`, metadata `{source, category, complexity, page, modality:"image-derived", figure_ref:"{model} p.{page} fig:page"}`.

### 4.2 인덱스
- **cross-modal 스토어(`chroma_db_mm`)** = 기존 텍스트 청크(C3) + image-derived 캡션 청크를 함께 적재 → Hybrid+Rerank가 둘 다 검색.
- `src/evaluation.py`의 `context_provider`가 이 스토어를 쓰도록 분기(텍스트-only=C3, C1=mm).

### 4.3 검증
- 캡션 샘플 3건 육안: IR8 해당 페이지 캡션에 정답 정보(예: "조작부 우측 상단", "Wi-Fi 끊김=빗금 부채꼴")가 실제로 담겼는지.

## 5. C2 — CLIP 이미지 임베딩 검색 (P0)

> 산출물: `src/clip_index.py`(`build_clip_index`, `clip_retrieve`), `data/clip_index/`

### 5.1 인덱스
- `clip-ViT-B-32`로 페이지 이미지 임베딩 → 벡터 인덱스(Chroma 또는 numpy+cosine). page 메타 보존.
### 5.2 검색
- 질문 텍스트를 `clip-ViT-B-32-multilingual-v1`로 임베딩 → 이미지 인덱스에서 top-k 페이지 검색 → **page-hit** 측정(C2a).
### 5.3 답변 (late-fusion, C2b — P1로 정교화)
- 검색된 top 페이지 이미지를 gpt-4o-mini에 투입해 답변 생성. C2a(검색만, 추론 없음) vs C2b(검색+VLM 읽기) 둘 다 기록 → "추론 단계엔 VLM 필수"가 드러나게.

## 6. 비교 평가 (P0)

> 산출물: `docs/week9_evaluation.md`, `data/week9_results.json`

- 대상: **image-required 8 + text-only 대조 ≥3**(golden_set_v2). 동일 answer 모델·동일 LLM-judge.
- 지표 두 축:
  - **Retrieval page-hit**: 검색 top-k에 정답 page(±2)가 들어오는가 (C0/C1/C2 비교). C0/C1는 `figure_ref`/reference_context, C2는 CLIP top-k.
  - **Answer correctness**: ground_truth 대비 LLM-judge(0/1 또는 점수). C0(text)/C1(caption)/C2a(검색만)/C2b(검색+VLM).
- 비교표(채울 것):

| 구성 | Retrieval page-hit (IR8) | Answer correct (IR8) | 대조(text-only) | 비고 |
|---|---|---|---|---|
| C0 text-only | 25%(기준선) | (측정) | (측정) | ADR-011 baseline |
| C1 캡션→텍스트RAG | (측정) | (측정) | (측정) | 메인 |
| C2a CLIP 검색만 | (측정) | (측정·낮을 것) | (측정) | 추론 없음 |
| C2b CLIP+VLM 읽기 | (=C2a) | (측정) | (측정) | late-fusion |

- **결론 도출:** C2a→C2b의 answer 상승폭과 C1의 answer가 "추론 단계에 VLM이 필요함"을 보이는가. CLIP retrieval(C2a page-hit)이 C1 대비 어떤지도 정직히 기록.

## 7. 노트북 (P0)

> 산출물: `w9/week9_multimodal_rag.ipynb`

- §3~6을 순서대로 호출(모듈 함수 사용, 로직은 `src/`에). 테스트 케이스 ≥3(image-required 중심) 실행·출력.
- 메모리: reranker는 WEEK6 하드닝(int8·thread cap·단일 인스턴스) 적용 상태 사용. CLIP은 소형이라 부담 적음.

## 8. 정성 평가 (P1)

`week9_tech_choice.md` 평가절 + (선택)`week9_evaluation.md`에: C1 vs C2 ROI, 가장 어려운 점(캡션 품질/CLIP 한국어·도면 OOD/late-fusion 비용), 복잡도 대비 이득, 10주차 개선.

## 9. 도전 과제 (P2)

표→markdown 구조화, 도면 영역 crop별 캡션, OCR 통합, 다중 이미지 비교, Vision 결과 citation 연결.

## 10. 검증 (P0, 권장 서브에이전트)

- C0/C1/C2가 **동일 테스트셋·동일 answer 모델·동일 judge**로 비교됐는지(공정성).
- week9_results 수치가 노트북에서 재현되는지, 비교표·결론이 데이터와 일치하는지.
- 캡션 "추측 금지" 준수 샘플 점검.

## 11. 연결 / 다음

- 기준선·근거: ADR-011, WEEK7_RUN2 §9. 결과는 10주차(crop·표·late-fusion 정교화, GraphRAG 검토)와 ADR(필요 시 ADR-012: 멀티모달 검색 결정) 입력.

## 12. 변경 이력

| 날짜 | 내용 |
|---|---|
| 2026-06-23 | 9주차 신규. 멀티모달 RAG(옵션 A) 확정, C1 캡션/C2 CLIP 비교 실험 설계(gpt-4o-mini, 전체 페이지 rasterize, image-required 8+대조). "추론 필요시 VLM" 가설을 공정 비교로 검증. |
