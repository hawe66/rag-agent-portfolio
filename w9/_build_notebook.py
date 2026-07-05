"""Generate w9/week9_multimodal_rag.ipynb from source cells.

Authoring helper (not a deliverable): keeps the notebook in version-friendly
plain Python and lets us regenerate it. Run from repo root:
    .venv/bin/python w9/_build_notebook.py
then execute with nbconvert.
"""

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(text):
    cells.append(new_markdown_cell(text))


def code(text):
    cells.append(new_code_cell(text))


md("""# Week 9 — 멀티모달 RAG MVP: C0 / C1 / C2 비교

Phase 2 착수. **두 cross-modal 파이프라인(C1 캡션 / C2 CLIP)을 baseline(C0 text-only) 대비 통제 비교**해
"추론이 필요한 멀티모달 파싱엔 VLM이 필요한가"를 데이터로 본다.

- **C0** text-only RAG (`chroma_db_c3`) — ADR-011 baseline
- **C1** VLM 캡션 → 텍스트 RAG 병합 (`chroma_db_mm`)
- **C2a** CLIP 검색만 / **C2b** CLIP 검색 + VLM 읽기 (late fusion)

로직은 전부 `src/`에 있고 노트북은 순서대로 호출만 한다(§3→§6). 무거운 산출물
(캡션·스토어·CLIP 인덱스·평가결과)은 **캐시를 재사용**한다 — 노트북 실행 시 재과금 없음.

> 명세: `docs/WEEK9_TASKS.md` · 결과 해석: `docs/week9_evaluation.md` · 근거: `adr/ADR-011-crossmodal-eval.md`""")

code("""import sys
from pathlib import Path

# repo root on sys.path so `import src.*` works when run from w9/
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

PDF_DIR = ROOT / "data/raw_pdfs"
IMG_ROOT = ROOT / "data/sample_images"
CAPTION_CACHE = ROOT / "data/week9_captions.json"
MM_DIR = ROOT / "data/chroma_db_mm"
CLIP_DIR = ROOT / "data/clip_index"
RESULTS = ROOT / "data/week9_results.json"
print("repo root:", ROOT)""")

md("""## §3 — 페이지 Rasterize (P0)

6개 PDF의 모든 페이지를 95dpi PNG로 렌더(`src/rasterize.py`). idempotent —
이미 렌더된 PNG는 덮어쓰기만 한다. IR8 참조 페이지 존재를 검증한다.""")

code("""from src.rasterize import rasterize_all, verify_ir8_pages

pages = rasterize_all(PDF_DIR, IMG_ROOT)
print(f"\\nTotal rendered: {len(pages)} pages")

ir8 = verify_ir8_pages(IMG_ROOT)
print("IR8 reference pages:", "ALL PRESENT" if all(ir8.values()) else ir8)""")

md("""## §4 — C1: VLM 캡션 → cross-modal 인덱스 (P0)

`src/multimodal.py`: 각 페이지를 gpt-4o-mini(vision)로 **구조화 캡션**(위치/방향/아이콘/표,
"보이는 것만")으로 변환 → image-derived 청크 → C3 텍스트 청크와 함께 `chroma_db_mm`에 적재.

캡션은 `data/week9_captions.json`에 캐시되어 있어 `caption_pages`는 **재호출 없이 즉시 반환**한다.""")

code("""from src.multimodal import caption_pages, load_mm_store

captions = caption_pages(pages, CAPTION_CACHE)   # cache hit → no API calls
print(f"captions: {len(captions)}")

# §4.3 샘플 육안: IR-V3(무선청소기 p19, Wi-Fi 연결끊김 아이콘) 캡션
sample = next(c for c in captions if c.category == "vacuumcleaner"
              and c.complexity == "complex" and c.page == 19)
print("\\n--- sample caption (vacuumcleaner_complex p19) ---")
print(sample.caption[:600])""")

code("""# cross-modal 스토어 로드 (이미 빌드됨: 338 텍스트 + 252 캡션 = 590)
mm_store = load_mm_store(MM_DIR)
print("mm store docs:", mm_store._collection.count())""")

md("""## §5 — C2: CLIP 이미지 임베딩 검색 (P0)

`src/clip_index.py`: 페이지 이미지를 `clip-ViT-B-32`로 임베딩, 질문은 멀티링궐
`clip-ViT-B-32-multilingual-v1`로 임베딩해 joint space에서 cosine top-k 페이지 검색.""")

code("""from src.clip_index import ClipIndex, clip_retrieve

clip_index = ClipIndex.load(CLIP_DIR)
print("CLIP index pages:", len(clip_index.metas))

demo = "무선청소기 상태 표시창에서 Wi-Fi 연결 끊김 아이콘"
print(f"\\nQuery: {demo}")
for h in clip_retrieve(clip_index, demo, k=5):
    print(f"  {h['category']}_{h['complexity']} p{h['page']}  score={h['score']:.3f}")
print("\\n→ 정답은 vacuumcleaner_complex p19. CLIP은 한국어·흑백 도면에서 변별력 낮음(score 평탄).")""")

md("""## §6 — C0/C1/C2 비교 평가 (P0)

`src/week9_eval.py`가 동일 테스트셋(IR8 + 대조 3)·동일 answer 모델·동일 judge로 4개 구성을
비교해 `data/week9_results.json`에 기록했다. 여기서는 저장된 결과를 로드해 표로 본다.
(재실행: `python -m src.week9_eval` — vision 호출 과금 발생.)""")

code("""import json
results = json.loads(RESULTS.read_text())

from src.week9_eval import print_summary
print_summary(results)

print("\\nIR8 question ids:", [q['id'] for q in results['questions'] if q['id'].startswith('IR')])""")

md("""**해석(상세는 `docs/week9_evaluation.md`):** MVP 규모에서 C1은 C0를 retrieval·answer 어느 축에서도
못 이겼다. (1) 캡션 청크가 reranking에서 텍스트 청크를 밀어내 retrieval을 해치고, (2) gpt-4o-mini
캡션이 정작 필요한 미세 아이콘 형태(빗금·분자모양)를 놓쳐 answer에 기여하지 못했다. C2/CLIP은 두 축 모두 붕괴.
→ "VLM이 필요하다 ≠ 전체페이지 캡션이면 된다". 10주차: 도면 영역 crop + modality-aware retrieval.""")

md("""## §7 — 테스트 케이스 (≥3, image-required 중심)

저장된 결과에서 image-required 3건의 C0 vs C1 vs C2b 답변을 나란히 본다.
(라이브 재실행이 아니라 `week9_results.json`의 per-question 기록을 재현 — 결정적·무과금.)""")

code("""qmeta = {q['id']: q for q in results['questions']}
def per_q(cfg): return {x['id']: x for x in results['results'][cfg]['per_q']}
C0, C1, C2b = per_q('C0'), per_q('C1'), per_q('C2b')

for qid in ['IR-A3', 'IR-V3', 'IR-W2']:
    q = qmeta[qid]
    print('=' * 72)
    print(f"[{qid}] {q['question']}")
    print(f"  GT       : {q['ground_truth']}")
    print(f"  C0  (j={C0[qid]['judge']}, page_hit={C0[qid]['page_hit']}): {C0[qid]['answer'][:130]}")
    print(f"  C1  (j={C1[qid]['judge']}, page_hit={C1[qid]['page_hit']}): {C1[qid]['answer'][:130]}")
    print(f"  C2b (j={C2b[qid]['judge']}): {C2b[qid]['answer'][:130]}")""")

md("""### 테스트 케이스 판독

- **IR-A3 (공기제균 아이콘 모양)**: C1은 정답 페이지(p22)를 찾았지만 캡션이 모양을 못 적어 "공기제균 아이콘"이라고 *이름*만 답 → judge=0. 캡션 단계 정보소실이 치명적임을 보여줌.
- **IR-V3 (Wi-Fi 끊김 아이콘)**: C0·C1 모두 "Y 아이콘" 환각. 정답은 부채꼴 빗금. 전체페이지 캡션이 작은 아이콘을 못 읽음.
- **IR-W2 (필터 화살표 면)**: C0·C1 모두 정답("돌출부에 맞춤") — 본문 텍스트로도 답 가능한 케이스라 캡션 기여 없이 맞음.

**종합:** "검색이 맞아도 미세 시각정보는 읽어야 한다"(VLM 추론 필요성)는 재확인되나, *전체페이지 캡션*이라는
형태로는 C0를 못 넘는다. 다음 주 영역 crop으로 아이콘을 화면 가득 채워 읽히는 것이 핵심 개선이다.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": ".venv (3.11.4)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11.4"},
}

out = Path(__file__).parent / "week9_multimodal_rag.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
