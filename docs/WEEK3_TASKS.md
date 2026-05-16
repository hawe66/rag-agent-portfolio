# WEEK 3 — Naive RAG (Baseline) 구현

> **이 작업은 `CLAUDE.md`의 Phase 1 (Advanced RAG)의 첫 단계다.**
> 시작 전 `CLAUDE.md`를 반드시 읽고, ADR-001~007을 인지한 상태에서 진행한다.
> 이 주차의 목적은 **"이후 개선의 기준점"**이 되는 Baseline RAG를 만드는 것. 화려할 필요 없음.

---

## 0. 이 주차의 핵심 원칙

- **Baseline은 단순해야 의미가 있다.** RecursiveCharacterTextSplitter + 일반 임베딩 + Chroma + similarity search top-k. 이게 baseline. 여기서 욕심내지 않는다.
- **개선은 4주차 이후에 한다.** 메타데이터 필터링, hybrid search, re-ranking 등은 이번 주차에 들어가지 않는다.
- **하지만 Phase 2 확장성은 처음부터 고려한다** (ADR-004). 청크 메타데이터 스키마를 짤 때 이미지 ID 필드 자리를 비워둔다 — 이번 주차에 채우지 않더라도.
- **평가 가능한 형태로 기록**한다. 이것이 RAGAS 점수보다 우선이라고 스터디 가이드에 명시됨.

---

## 1. 환경 준비

### 1.1 패키지 설치

```bash
pip install langchain-chroma ragas sentence-transformers
```

> **의존성 관리 메모.** 본인은 uv 환경에 익숙하므로, `pyproject.toml`을 메인으로 관리하고 `requirements.txt`는 동기화하는 방식 유지. 위 명령은 baseline 전제 — 본인 환경에 맞게 `uv add` 등으로 변환해도 됨.

### 1.2 데이터 폴더 구조

```
data/
├── raw_pdfs/        # LG 매뉴얼 PDF 6개 원본
└── processed/       # 청크 / 임베딩 처리 결과 (gitignore 권장)
docs/
w{week_number}/      # 각 주차별 테스트를 위한 폴더
├── TODO.md
├── *.ipynb
└── *.py
src/                 # 최종 산출 파이프라인
```

---

## 2. 데이터 수집 (선결 조건)

### 2.1 다운로드 대상

`CLAUDE.md` ADR-005 기준 6개 매뉴얼.

| # | 카테고리 | 모델명 | 비고 |
|---|---|---|---|
| 1 | 정수기 (단순) | WD325AS | 다운로드 완료 ✅ |
| 2 | 정수기 (복잡) | WD520AWB | 다운로드 완료 ✅ |
| 3 | 공기청정기 (단순) | AS181DAW | 다운로드 완료 ✅ |
| 4 | 공기청정기 (복잡) | AS281DAW | 다운로드 완료 ✅ |
| 5 | 유선청소기 (단순) | AS9000WR | **미확인 — 검증 필요** |
| 6 | 무선청소기 (복잡) | AX920BWE | **미확인 — 검증 필요** |

### 2.2 필요시 추가

1. `https://www.lge.co.kr/support/product-manuals` 접속
  - 혹은 `https://gscs-manual.lge.com/Total/HQ/GatewayPage/main.html`
2. 모델명 입력 → Owner's Manual PDF 다운로드 시도
3. **다운로드 성공 시:** `data/raw_pdfs/`에 저장하고 6개 모두 갖춤
4. **다운로드 실패 시:** Strategy Thread로 돌아가서 swap 모델 결정 (이 주차 진행은 우선 4개로 시작 가능)

### 2.3 파일 명명 규칙 (수정 필요)

```
data/raw_pdfs/
├── waterpurifier_simple_*.pdf
├── waterpurifier_complex_*.pdf
├── airpurifier_simple_*.pdf
├── airpurifier_complex_*.pdf
├── vacuumcleaner_simple_*.pdf
└── vacuumcleaner_complex_*.pdf
```

이 규칙이 채택되면 메타데이터 추출 시 카테고리/난이도 자동 부여 가능.

---

## 3. Naive RAG 파이프라인 구현

### 3.1 단계별 작업

1. **Document Loader**: PyPDFLoader 또는 PDFPlumberLoader로 PDF 로딩 (어느 것이 텍스트 추출 품질 좋은지 1~2개 PDF로 비교 후 선택)
2. **Text Splitter**: `RecursiveCharacterTextSplitter`, 시작값 `chunk_size=1000`, `chunk_overlap=200`
3. **Embedding**: 한국어이므로 sentence-transformers 한국어 지원 모델 (예: `jhgan/ko-sroberta-multitask` 또는 `BM-K/KoSimCSE-roberta-multitask`) 또는 OpenAI `text-embedding-3-small` 등 — baseline에서는 한 가지만 정해서 일관되게 사용
4. **Vector DB**: Chroma (`langchain-chroma`)
5. **Retriever**: similarity search, `k=5` (baseline)
6. **Generator**: LLM에 context + question 넣고 답변 생성 (system prompt는 아래 §3.3 참고)

파일 용량이 매우 큼 주의. 실제 질의와 무관한 내용 많음 주의.

### 3.2 청크 메타데이터 스키마 (중요 — Phase 2 확장 고려)

각 청크에 다음 메타데이터를 부착한다:

```python
{
    "source": "vacuum_complex_AX920BWE.pdf",
    "category": "vacuum",                  # water_purifier / air_purifier / vacuum
    "complexity": "complex",                # simple / complex
    "model_name": "AX920BWE",
    "page": 12,                             # PDF 페이지 번호
    "chunk_id": "AX920BWE_p012_c003",       # 매뉴얼-페이지-청크 순서
    # Phase 2 확장 자리 (지금은 비워둠)
    "section": None,        # 안전주의/설치/사용/유지보수/문제해결/사양 — 추출 가능하면
    "step_number": None,    # 단계가 명시된 경우만
    "image_ids": [],        # Phase 2에서 채움
}
```

> ADR-004의 "이미지 노드 자리 비워두기"가 여기서 구체화됨. `image_ids` 필드는 빈 리스트로 두되 스키마에는 존재한다.

### 3.3 RAG Prompt (baseline)

스터디 가이드 예시 prompt를 그대로 시작점으로 사용. 단, **출처에 모델명을 명시**하도록 보강 (멀티 매뉴얼 코퍼스이기 때문).

```
당신은 LG 가전제품 사용설명서를 안내하는 QA Assistant입니다.

아래 제공된 context만 사용해서 질문에 답변하세요.
context에 없는 내용은 추측하지 말고 "문서에서 확인할 수 없습니다"라고 답변하세요.
답변 마지막에는 참고한 매뉴얼의 모델명과 페이지 번호를 함께 적어주세요.

[Context]
{context}

[Question]
{question}

[Answer]
```

> 첫 turn에서 "어떤 모델에 대해 묻는가" 식별은 baseline에서는 구현하지 않는다 (ADR-006은 Phase 1 후반 또는 Phase 2의 작업). Baseline에서는 6개 매뉴얼 전체에서 검색.

---

## 4. 평가용 질문 세트 작성 (최소 5개, 가능하면 10개)

### 4.1 라벨링 원칙 (Phase 2 대비)

각 질문에 다음 라벨을 부착한다 — Phase 2에서 cross-modal 효과를 분리 평가하기 위함.

| 라벨 | 의미 |
|---|---|
| `text-only` | 텍스트만으로 답 가능 |
| `image-helpful` | 텍스트로 가능하지만 그림 있으면 더 명확 |
| `image-required` | 그림 없이는 답 불가능 (Phase 1에서는 이 질문에 baseline이 약할 것 — 이게 정상) |

### 4.2 질문 카테고리 (다양성 확보)

스터디 가이드의 일반적 5개 외에, **본 도메인 특화 카테고리**를 섞는다:

- **단계 설명**: "필터 교체는 어떻게 하나요?" (절차)
- **부품 역추적**: "이 모델에 들어 있는 흡입구 종류는?" (부품)
- **Troubleshooting**: "전원이 안 켜져요. 어떻게 해야 하나요?" (문제해결)
- **모델 간 비교**: "WD325AS와 WD520AWB의 차이점은?" (cross-document — baseline에는 도전적)
- **주의사항**: "이 제품 사용 시 주의해야 할 점은?" (안전)

### 4.3 작성 가이드

- 답이 매뉴얼에 **실제로 존재**하는 질문이어야 함 (검증 가능해야 함)
- 정답 또는 정답을 포함하는 페이지 번호를 미리 적어둠 → context recall 평가에 사용
- 6개 매뉴얼 골고루 커버

### 4.4 저장 형식 (제안)

```
docs/eval_questions_v1.md  또는  docs/eval_questions_v1.json
```

JSON 예시:
```json
[
  {
    "id": "Q001",
    "question": "WD325AS의 필터 교체는 어떻게 하나요?",
    "category": "step",
    "modality_label": "image-helpful",
    "ground_truth_model": "WD325AS",
    "ground_truth_page": [12, 13],
    "ground_truth_answer": "..."
  }
]
```

---

## 5. 산출물 (이번 주 deliverable)

### 5.1 필수
- [ ] `data/raw_pdfs/` 에 PDF 6개 (또는 4개 + 청소기 swap 결정)
- [ ] `notebooks/week3_baseline_rag.ipynb` (또는 `docs/week3_baseline_rag.md`)
- [ ] `docs/eval_questions_v1.md` (또는 `.json`) — 질문 5~10개
- [ ] 청크 메타데이터 스키마가 §3.2 형식대로 구현됨 (Phase 2 자리 포함)

### 5.2 발표/기록 항목 (스터디 공유용)
- 어떤 데이터로 RAG 구현했는지 (LG 가전 매뉴얼 6개)
- chunk_size / chunk_overlap / 문서 개수 / 생성된 chunk 개수
- 어떤 질문으로 테스트했는지 (위 평가 셋)
- 검색된 chunk가 질문과 잘 맞았는지 (정성 평가)
- LLM 답변이 문서에 근거했는지
- 현재 baseline의 아쉬운 점
- 다음 주차에 개선하고 싶은 부분
- (가능 시) RAGAS 점수: Faithfulness / Answer Relevancy / Context Precision / Context Recall

### 5.3 4주차 이후 개선 후보 메모 (이번 주차에 발견된 limitation 기록)

baseline 돌려보면서 어디가 약한지 메모. 예시 항목:
- 매뉴얼 간 cross-document 질문에 약함 (모델 간 비교)
- 표 데이터 (사양표 등) 검색 품질 떨어짐
- 그림 봐야 답이 명확해지는 질문 (`image-required`)에서 baseline 한계
- ...

이 메모가 4주차 데이터 전처리 / 5주차 retrieval 고도화 / 6주차 Agentic RAG 의제가 된다.

---

## 6. 예습 자료 체크리스트 (스터디 가이드 기준)

- [ ] LangChain Document Loaders 섹션
- [ ] LangChain Text Splitters 섹션
- [ ] LangChain Vector Stores 섹션
- [ ] LangChain Retrievers 섹션
- [ ] RAGAS 공식 문서
- [ ] 테디노트 RAG 기초 영상

---

## 7. 발표자라면 (해당 시)

발표 시간 30분. 자유 형식.

내용 구성 (스터디 가이드 그대로):
1. Naive RAG 전체 파이프라인
2. Baseline RAG를 먼저 만드는 이유
3. Text Splitter와 Chunking 전략
4. Retriever와 검색 품질
5. RAG Prompt 설계

> 본인이 발표자라면 마지막 슬라이드에 "다음 주차 의제" 섹션을 넣고, §5.3 메모를 그대로 활용.

---

## 8. 작업 순서 권장 (Claude Code 기준)

1. **읽기 단계** — `CLAUDE.md` 전체 + 이 파일 §0~3 정독
2. **데이터 단계** — `data/raw_pdfs/` 채우기 (청소기 검증 포함)
3. **PDF 로딩 단일 검증** — PyPDFLoader vs PDFPlumberLoader 1~2개 PDF로 텍스트 추출 비교, 한 가지 선택
4. **메타데이터 스키마 구현** — §3.2 그대로
5. **파이프라인 구현** — Splitter → Embedding → Chroma → Retriever → LLM
6. **질문 셋 작성** — §4
7. **돌려보고 정성 평가** → §5.2 항목 채우기
8. **(여유 시) RAGAS 점수 산출**
9. **§5.3 limitation 메모 작성** — 4주차 입력

---

## 9. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| 2026-05-09 | 초안 작성 (Strategy Thread 17턴 + 스터디 3주차 가이드 통합) |
