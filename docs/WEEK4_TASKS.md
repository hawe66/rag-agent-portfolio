# WEEK 4 — 데이터 전처리 진단 + Chunking 비교 실험

> **이 작업은 `PROJECT_CONTEXT.md` Phase 1 (Advanced RAG)의 두 번째 단계다.**
> 시작 전 `PROJECT_CONTEXT.md`와 3주차 회고(`w3/TODO.md` 또는 동등 정리물)를 반드시 읽는다.
> 이번 주차의 본인은 **스터디 발표자**다. 따라서 마지막 §10에서 발표 자료용 정리물을 별도 산출한다.

---

## 0. 이 주차의 핵심 원칙

- **"내 데이터를 가장 잘 이해하는 사람이 되기"** — 스터디 가이드의 메시지. 화려한 기법보다 진단이 우선.
- **변수 통제**: chunking 전략만 바꾸고 나머지(embedding, retriever k, prompt, 평가 셋)는 3주차 baseline 그대로 유지.
- **수치 + 가설**: RAGAS 점수 변화 자체보다, "왜 그렇게 변했는가"를 가설로 설명할 수 있어야 한다.
- **Phase 2 확장성**: 메타데이터 스키마에서 이미지 자리(`image_ids: []`)는 여전히 비워둔다.
- **3주차 한계(LIM-001~008) 인지**: 이번 주차의 핵심 공격 대상은 **LIM-001**(텍스트 순서 깨짐), **LIM-005**(category 혼합), **LIM-006**(공통 용어 contamination), **LIM-008**(특정 사실 retrieval 누락). LIM-002(이미지 검출)는 Phase 2. LIM-007(prompt citation)은 5주차+. LIM-003/004는 인프라 이슈로 별도 처리.

---

## 1. 사전 조건

### 1.1 패키지 설치

```bash
uv add tiktoken matplotlib
# RAGAS는 이미 3주차에 설치되어 있어야 함. 버전 호환성 점검 (§1.3)
```

### 1.2 3주차 산출물 전제

- `data/raw_pdfs/` 에 매뉴얼 PDF (현재 6개로 확정 가정)
- 3주차 baseline 파이프라인 동작 (Chroma 0.4.22 + PDFPlumberLoader + RecursiveCharacterTextSplitter + OpenAI text-embedding-3-small + gpt-4o-mini)
- 평가 질문 셋 (3주차 사용한 것 그대로)
- `data/chroma_db/` 벡터 스토어

### 1.3 RAGAS 버전 호환성 확인 (선결, macOS 13 환경)

> **배경:** 3주차 LIM-004에서 macOS 13 ARM의 torch 의존성 이슈 확인됨. RAGAS는 내부적으로 sentence-transformers/torch를 끌어올 수 있어 동일한 함정 가능.

**작업:**
1. 현재 환경에서 `pip show ragas` 로 설치 버전 확인
2. RAGAS 의존성이 chromadb 0.4.22 + onnxruntime 1.16.3 + protobuf 제약과 충돌하는지 점검
3. 충돌 시 RAGAS 버전 다운그레이드 또는 LLM/embedding을 OpenAI로만 강제 (RAGAS 내부 모델 사용 회피)
4. 시도한 버전 조합과 결과를 `docs/week4_retrospective.md` 부록에 짧게 기록 (LIM-004 후속 보고 성격)

> **베일 아웃 기준:** 30분~1시간 안에 RAGAS 설정 안 끝나면 일단 정성 평가 + 부분 RAGAS로 진행하고, 실험 다 끝낸 후 RAGAS만 별도 시도. 환경 디버깅에 4주차 본 작업을 인질로 잡지 않는다.

> **Context Recall 제외:** 스터디 가이드 명시 — ground truth context 라벨링은 7주차 작업. 이번 주차는 Faithfulness / Answer Relevancy / Context Precision 3개 지표만 본다.

---

## 2. 3주차 Baseline 회고 (선결)

> 산출물: `docs/week4_retrospective.md` 의 첫 섹션

### 2.1 잘못 검색된 chunk 사례 발굴 — 실제 chunk 원문 캡처 필수

3주차 LIM-005, LIM-008은 추상 수준에서 정리되어 있다. 4주차에서는 **실제 chunk 원문**을 캡처해서 발표 자료의 살아있는 evidence로 만든다.

**최소 2개, 권장 3개 사례** 발굴. 각 사례마다:

```markdown
### 사례 #1 [LIM-005 instance — category cross-contamination]
- 질문 (원문): "..."
- 검색된 chunk top-1 (잘못된 것):
  - source: vacuum_complex_AX920BWE.pdf
  - page: 15
  - chunk_id: AX920BWE_p015_c002
  - **chunk 원문 (그대로 붙여넣기)**:
    > "..." (실제 텍스트, 최소 2~3 문장 분량)
- 정답이 있어야 할 chunk:
  - source: water_purifier_simple_WD325AS.pdf
  - page: 12
  - chunk_id: WD325AS_p012_c003
  - **chunk 원문**:
    > "..."
- 유사도 점수 (있으면): 잘못된 chunk vs 정답 chunk
- 왜 잘못 검색됐다고 생각하는가 (가설):
  - 공통 용어 "필터" "교체"가 cross-category false match 유발 (LIM-006)
  - 카테고리 메타데이터 필터링 부재 (LIM-005)
```

**우선순위 사례 (반드시 캡처):**
- **LIM-005 instance** — category 혼합 (정수기 질문 → 공기청정기 chunk 등)
- **LIM-008 instance** — "청소기 배터리 충전 시간" 실패 사례. 어떤 chunk가 top-k에 들어왔고, 정답이 어디에 있는데 왜 못 찾았는지

이 두 사례는 발표 §10 항목 A의 핵심 재료다.

### 2.2 데이터 전처리 관점 개선 가설 (1~2개)

§2.1 사례에서 도출. **3주차 회고의 LIM 분석과 일관되게** 작성.

```markdown
### 개선 가설 A — Layout-aware parsing
- 관찰: LIM-001(멀티컬럼 텍스트 순서 깨짐)로 청크 내 문맥이 이미 깨져 있음. chunk_size를 어떻게 조정해도 이 문제는 안 풀림.
- 가설: layout-aware loader (unstructured / docling / marker-pdf)로 컬럼 순서 보존하면 청크 의미가 회복될 것
- 검증 방법: 전략 C3 (§4.4)

### 개선 가설 B — 노이즈 제거 + 단계 단위 separator
- 관찰: 머리말/꼬리말 반복, 단계 중간 절단
- 가설: 머리말 regex 제거 + 단계 패턴(`\n1.`, `\n①` 등) separator로 의미 단위 보존
- 검증 방법: 전략 C2 (§4.4) — C3 fallback 겸용
```

---

## 3. 데이터 품질 진단

> 산출물: `notebooks/w4/data_analysis.ipynb`

### 3.1 문서 구조 특성 파악

**일반 항목:**
- 표 존재 여부 및 빈도 (사양표, 트러블슈팅 표 등)
- 이미지 존재 여부 — **3주차 LIM-002 후속**: pdfplumber/PyMuPDF가 놓친 이미지 유형(vector graphics, nested XObjects) 통계도 함께 기록 (Phase 2 입력 자료)
- 헤더 / 섹션 구조 (안전주의 / 설치 / 사용 / 유지보수 / 문제해결 / 사양)
- footnote, 각주
- 메타데이터 (모델명, 발행일 등이 PDF 안에 명시되어 있는가)

**도메인 특화 항목:**
- 단계 번호 패턴 (예: "1.", "①", "STEP 1" 등 — 어떤 형식인가, 매뉴얼마다 다른가)
- 부품/도구 이름 등장 빈도 (가전 매뉴얼은 부품명이 핵심 entity)
- 같은 매뉴얼 안에 여러 변형 모델이 같이 다뤄지는가 (예: "WD325AS / WD325AW 공통")

**LIM-001 정량화 (전략 C3 선택 근거가 되는 작업):**
- 멀티컬럼 페이지가 전체의 몇 %인가
- 멀티컬럼 페이지에서 텍스트 순서 깨짐이 실제로 얼마나 심한가 (샘플 2~3페이지로 정량/정성 확인)

### 3.2 노이즈 요소 식별

- 머리말 / 꼬리말 (모든 페이지에 반복되는 텍스트)
- 페이지 번호
- 광고성 문구 (제품 홍보, A/S 안내 반복)
- 중복 문장 (안전주의가 여러 페이지에 반복되는 패턴 등)
- 표가 텍스트로 깨져 들어온 경우의 의미 없는 문자열

식별 결과는 표로 정리:

```markdown
| 노이즈 유형 | 등장 빈도 | 영향 정도 | 처리 방안 |
|---|---|---|---|
| 머리말 (모델명+제품군) | 모든 페이지 | 중 | regex 제거 |
| ... | ... | ... | ... |
```

### 3.3 문서 길이 분포 시각화

- 매뉴얼별 페이지 수, 추출된 텍스트 토큰 수 (tiktoken으로)
- chunk_size=1000 기준 청크 개수
- 최소 / 최대 / 평균 / 표준편차
- **시각화 1개 이상**: 매뉴얼별 토큰 수 막대그래프 또는 청크 길이 히스토그램

> 발표용으로도 쓸 시각화. 깔끔하게 만든다.

---

## 4. Chunking 전략 비교 실험

> 산출물: `notebooks/w4/chunking_experiments.ipynb`

### 4.1 실험 설계 원칙

- **변수 통제**: chunking 외 모든 것 고정
  - 같은 PDF 코퍼스
  - 같은 embedding 모델 (OpenAI text-embedding-3-small)
  - 같은 retriever (similarity, k=5)
  - 같은 prompt (3주차 그대로)
  - 같은 평가 질문 셋
  - 같은 LLM (gpt-4o-mini, temperature=0.1)
- **재현 가능성**: 사용한 모델 버전, 하이퍼파라미터 모두 기록.

### 4.2 전략 A: Baseline (3주차 설정 그대로) — **이번 주차에 RAGAS로 새로 측정**

- `RecursiveCharacterTextSplitter`
- `chunk_size=1000`, `chunk_overlap=200`
- PDFPlumberLoader

> **중요:** 3주차에는 RAGAS 점수가 없었다 (정성 평가만, 75% retrieval accuracy / 75% RAG correct). 이번 주차에 baseline 자체를 RAGAS로 재기록한다. 이게 §4.5 비교 표의 기준점이 된다.

산출:
- Faithfulness / Answer Relevancy / Context Precision 점수
- 청크 개수 (3주차 정리에 따르면 320개 — 재실행 시 동일해야 정상)

### 4.3 전략 B: 파라미터 변경

같은 splitter, 파라미터만 변경. **2개 변종**:
- B1: 작은 청크 — `chunk_size=500`, `chunk_overlap=100`
- B2: 큰 청크 — `chunk_size=1500`, `chunk_overlap=300`

**가설:**
- 작은 청크: 검색 정밀도 ↑, 문맥 부족 ↓ — LIM-008(특정 사실 retrieval 누락)에 도움이 될 수도
- 큰 청크: 문맥 풍부 ↑, 검색 정밀도 ↓ — Faithfulness에 유리할 수 있으나 LIM-006(공통 용어 contamination) 악화 우려

### 4.4 전략 C: Layout-aware (메인) + Custom Separator (Fallback)

> **본인 결정사항:** 3주차 LIM-001이 chunking 파라미터로는 풀리지 않는 근본 문제이므로, **C3(layout-aware)를 메인 카드로** 가되 **C2(custom separator)를 fallback으로** 준비.

#### 4.4.1 메인: C3 — Layout-aware parsing

**후보 라이브러리 (시도 순서):**

1. **unstructured** (`unstructured[pdf]`)
   - 장점: 가장 보편적, LangChain 통합 좋음 (`UnstructuredPDFLoader`)
   - 단점: 한국어 PDF에서 품질 들쭉날쭉 가능, 의존성 무거움
   - 시도 모드: `strategy="hi_res"` (layout 분석 강화)

2. **docling**
   - 장점: IBM 발 신생 도구, 표/레이아웃 보존 강력하다고 알려짐
   - 단점: 환경 설정 까다로움 가능 (LIM-004 유사 위험)

3. **marker-pdf**
   - 장점: PDF → markdown 변환, 구조 보존
   - 단점: GPU 필요할 수 있음, 설치 비용

**시도 프로토콜:**
- 6개 PDF 중 **멀티컬럼 비율이 가장 높은 1개**로 먼저 시도 (§3.1에서 식별)
- 추출 결과를 baseline(PDFPlumberLoader)와 텍스트 순서 비교
- 멀티컬럼 페이지에서 순서 보존 여부를 정성 확인 (paragraph 단위로 5~10개 샘플)
- **품질 합격 시** → 전체 6개에 적용 → RAGAS 측정
- **품질 불합격 또는 환경 설치 실패 시** → 베일아웃 → C2로 전환

**베일아웃 기준 (시간 박스):**
- 라이브러리 설치 + 1개 PDF 처리까지 **2시간 초과 시** C2로 전환
- 품질 정성 평가 결과 baseline 대비 명확한 개선 없으면 C2로 전환
- 전환 결정과 그 이유를 회고에 기록 (이게 발표 §10 항목 D + G의 핵심 콘텐츠)

#### 4.4.2 Fallback: C2 — Custom Separator

C3 실패 또는 부분 성공 시 적용. C2는 단독으로도 의미 있는 실험.

- `RecursiveCharacterTextSplitter`의 `separators` 파라미터를 도메인 맞춤
- 예시:
  ```python
  separators = [
      "\n\n",           # paragraph
      "\n1. ", "\n2. ", "\n3. ",   # 한글 단계 (숫자)
      "\n① ", "\n② ", "\n③ ",       # 원문자 단계
      "\nSTEP ",                       # 영문 STEP
      "\n■ ", "\n● ", "\n▶ ",         # 불릿
      ".\n", ". ", " "
  ]
  ```
- 정확한 단계 패턴은 §3.1 결과에서 식별한 것을 반영

#### 4.4.3 C3, C2 모두 시도된 경우

비교 표에 두 row 모두 기록. 발표에서 "메인 카드와 fallback 둘 다 시도, 결과 비교"는 매우 좋은 서사.

### 4.5 비교 결과 표

```markdown
| 전략 | 설명 | 청크 개수 | Faithfulness | Answer Relevancy | Context Precision |
|---|---|---|---|---|---|
| A (baseline) | Recursive 1000/200 | ? | ? | ? | ? |
| B1 (작은 청크) | Recursive 500/100 | ? | ? | ? | ? |
| B2 (큰 청크) | Recursive 1500/300 | ? | ? | ? | ? |
| C2 (custom sep) | 도메인 separator | ? | ? | ? | ? |
| C3 (layout-aware) | unstructured / docling / marker | ? | ? | ? | ? |
```

> C3가 실패해 시도 못 한 경우, 표에서 "C3 시도 결과: 실패 — 사유 기록"으로 명시. 빈칸 두지 말 것.

### 4.6 정성 분석 동반 (수치만 보지 않는다)

각 전략에서 동일 질문 2~3개에 대해 **실제 검색된 top-3 chunk 비교**. 특히:
- LIM-005 instance 질문 → 어떤 전략이 category 혼합을 줄였는가
- LIM-008 instance 질문 → 어떤 전략이 특정 사실을 회수했는가

수치 차이의 근원이 chunk 내용에서 어떻게 드러나는지 눈으로 본다.

---

## 5. 메타데이터 확장

> 산출물: `notebooks/w4/chunking_experiments.ipynb` 또는 별도 셀

### 5.1 메타데이터 스키마 갱신

3주차에서 사용한 스키마를 확장. 최소 2개 이상의 필드 추가.

**기존 (3주차):**
```python
{
    "source": "...",
    "category": "vacuum",
    "complexity": "complex",
    "model_name": "...",       # LIM-003 영향 — 가능한 범위에서
    "page": 12,
    "chunk_id": "...",
}
```

**4주차 추가:**
```python
{
    # ... 기존 필드 ...

    # 새로 추가 (WEEK4)
    "section": "사용",                       # 안전주의/설치/사용/유지보수/문제해결/사양
    "has_table": False,                       # 청크 내 표 데이터 포함 여부
    "has_step_number": True,                  # 단계 번호 포함 여부 (도메인 특화)
    "token_count": 487,                       # tiktoken 기준

    # Phase 2 확장 자리 (그대로 비워둠)
    "step_number": None,
    "image_ids": [],
}
```

> `section`은 §3에서 식별한 헤더/섹션 구조를 활용해 자동 부착. 자동화 어려우면 휴리스틱(페이지 범위 매핑).

### 5.2 검증

벡터 스토어에서 청크 1~2개 출력해서 메타데이터가 잘 들어갔는지 확인. 코드 한 셀이면 됨.

### 5.3 활용 아이디어 (다음 주차 빌드업)

`docs/week4_retrospective.md`에 한 단락. **3주차 LIM과 직접 연결**:

- `category` 필터 → **LIM-005, LIM-006 직접 공격** (5주차 Self-Query Retriever 빌드업)
- `section` 필터 → "설치 방법" 질문은 설치 섹션만, "고장났어요"는 문제해결 섹션만
- `has_table` 활용 → 사양 관련 질문 (LIM-008 후보)에 표 청크 가중치 ↑
- `model_name` 필터 → 첫 turn 매뉴얼 식별(ADR-006)의 자연스러운 구현 경로

---

## 6. 회고 작성

> 산출물: `docs/week4_retrospective.md`

다음 섹션 구조로 작성:

1. **3주차 baseline 회고**
   - 잘못 검색된 chunk 사례 (LIM-005, LIM-008 instance 포함, 실제 chunk 원문 캡처)
   - 개선 가설 (1~2개, LIM과 매핑)
2. **데이터 진단 요약** (§3 핵심 발견 3~5개 bullet, LIM-001 정량화 포함)
3. **Chunking 전략 비교**
   - 전략 C 선택 reasoning (한 단락) ← **면접 핵심**
   - C3 시도 결과 (성공/실패 무관 — 실패해도 그 자체가 학습)
   - 비교 결과 표
   - 정성 분석에서 발견한 것
4. **메타데이터 활용 아이디어** (§5.3, 한 단락)
5. **RAGAS 점수 해석**
   - 4주차에 처음 산출한 baseline 대비 어떤 지표가 올랐고 어떤 게 떨어졌는가
   - 그 변화의 가설
   - 본 도메인에 가장 잘 맞은 전략과 이유
6. **5주차 retrieval 고도화에서 시도하고 싶은 것**
   - LIM-005, LIM-006 → metadata filtering / self-query retriever
   - LIM-007 → prompt 개선 (citation 로직)
   - LIM-008 → 청크 커버리지 분석
7. **(부록) 환경/인프라 메모** — RAGAS 버전 호환성 결과, LIM-003/004 후속 상태

---

## 7. 산출물 체크리스트

### 필수
- [ ] `notebooks/w4/data_analysis.ipynb` (§3)
- [ ] `notebooks/w4/chunking_experiments.ipynb` (§4, §5)
- [ ] `docs/week4_retrospective.md` (§6)
- [ ] `docs/week4_presentation_materials.md` (§10)

### 선택 (도전 과제 — 흥미 있으면)
- [ ] Metadata Filtering Retriever 구현 (5주차 Self-Query 빌드업)
- [ ] SemanticChunker 적용
- [ ] OCR (LIM-002 후속 — PDF 이미지 영역 OCR 시도)

> C3 시도 자체가 사실상 도전 과제 성격. 별도 도전 과제는 시간 여유 있을 때만.

---

## 8. 작업 순서 권장

1. `PROJECT_CONTEXT.md` + 3주차 회고(`w3/TODO.md`) 다시 읽기
2. **§1.3 RAGAS 환경 점검** — 베일아웃 기준 인지
3. **§2 회고 먼저** — LIM-005, LIM-008의 실제 chunk 원문 캡처
4. **§3 데이터 진단** — 시각화 + LIM-001 정량화
5. **§4.2 전략 A — baseline을 RAGAS로 새로 측정**
6. **§4.3 전략 B — 파라미터 변경 2종**
7. **§4.4.1 전략 C3 시도** — 시간 박스 2시간 인지
8. **§4.4.2 전략 C2** — C3 실패 시 또는 병행
9. **§5 메타데이터 확장** — chunking 실험과 병행 가능
10. **§6 회고 작성**
11. **§10 발표 자료용 정리** ← 가장 마지막

---

## 9. 변수 통제 체크리스트 (실험 전 확인)

- [ ] embedding 모델: 3주차와 동일한가 (OpenAI text-embedding-3-small)
- [ ] retriever k: 동일한가 (k=5)
- [ ] prompt: 동일한가
- [ ] 평가 질문 셋: 동일한가
- [ ] LLM 모델 + 버전: 동일한가 (gpt-4o-mini, temperature=0.1)
- [ ] 변경된 것은 chunking 전략 (혹은 메타데이터 스키마)만인가

체크되지 않은 항목이 있으면 실험 결과 비교가 의미 없어진다.

---

## 10. 발표 자료용 정리물 (마지막에 별도 산출)

> **이 섹션은 Claude Code가 실험을 다 끝낸 후 마지막에 작성하는 별도 산출물이다.**
> 산출 위치: `docs/week4_presentation_materials.md` (이후 Strategy Thread에서 발표 자료로 가공)

스터디 발표(30분, 자유 형식)의 case study로 본인 4주차 실습을 활용하기 위해, 다음 형식으로 정리물을 만든다. **발표 슬라이드를 직접 만들지 않는다** — 그건 Strategy Thread의 일.

> **발표용 LIM 필터링:** 발표에서 다룰 LIM은 **001, 005, 006, 008**. LIM-002(이미지 검출)는 Phase 2 예고로 한 줄, LIM-007(citation)은 5주차 예고로 한 줄. **LIM-003(모델명 추출), LIM-004(macOS 13 의존성)는 발표 자료에 포함하지 않는다.** 도메인/인프라 특수 이슈로 발표 흐름을 끊는다.

### 10.1 정리 항목 (모두 본인 실험 결과에서 추출)

```markdown
## A. "잘못 검색된 chunk" 살아있는 사례 2~3개
- §2.1 사례를 발표용으로 다듬은 버전 (LIM-005, LIM-008 instance, 실제 chunk 원문 포함)
- 발표 §1 "왜 데이터 전처리가 중요한가"의 evidence로 사용

## B. 본 도메인 데이터 특성 요약
- §3.1, §3.2 핵심 발견 3~5개 (특히 LIM-001 멀티컬럼 정량화)
- §3.3 시각화 1개 (png 또는 ipynb 셀 참조)
- 발표 §2 "문서 종류별 전처리 전략"의 PDF 사례로 사용

## C. Chunking 전략 비교 결과
- §4.5 표 그대로
- §4.6 정성 분석에서 가장 흥미로운 비교 사례 1개
- 발표 §3의 메인 콘텐츠

## D. 전략 C 선택 reasoning + C3 시도 결과
- §6의 한 단락 그대로
- C3 시도 성공/실패와 그 이유
- 발표 §3 + §5 (A/B 테스트 기본기) 모두에서 사용

## E. 메타데이터 설계와 활용 계획
- §5.1 스키마 + §5.3 활용 아이디어
- 발표 §4 "메타데이터의 힘"의 살아있는 예시

## F. RAGAS 점수 해석
- §6의 5번 섹션 그대로 (단, 4주차 baseline 자체가 처음 RAGAS 산출이라는 점 명시)
- 발표 §6 "결과 해석 가이드"의 본인 사례

## G. 발견된 한계 / 다음 주차로 이월할 것
- 정량적으로 개선 안 된 부분
- C3 시도 결과 (성공/실패 무관)
- LIM-002 → Phase 2, LIM-007 → 5주차 예고
- 발표 마지막 "limitation + next step" 슬라이드 재료
```

### 10.2 발표용 정리물 출력 형식

각 항목 A~G마다 다음 정보를 포함:
- **핵심 메시지 (1문장)** — 발표 슬라이드 제목 후보
- **수치/사례** — 발표 본문 재료
- **시각화 참조** — 노트북 셀 번호 또는 png 파일 경로
- **(선택) 면접용 답변 1~2문장** — "왜 이 chunk_size?", "왜 layout-aware를 시도했나?" 같은 질문 대비

이렇게 정리되면 Strategy Thread에서 이 정리물을 받아 노션 붙여넣기용 발표 md를 만들 수 있다.

---

## 11. 변경 이력

| 날짜 | 변경 내용 |
|---|---|
| (초안) | 4주차 가이드 + Strategy Thread 합의 통합 |
| (개정 v2) | 3주차 회고 반영: LIM-001~008 명시, 전략 C3 메인+C2 fallback, baseline RAGAS 재측정, 발표용 LIM 필터링(003/004 제외), RAGAS 환경 호환성 점검 |