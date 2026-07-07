"""
C1 — VLM caption → cross-modal index (Week 9, WEEK9_TASKS §4).

Pipeline:
    rasterized page PNG -> gpt-4o-mini structured Korean caption
                        -> "image-derived" chunk
                        -> merged with C3 text chunks into chroma_db_mm

The merged store lets the existing Hybrid+Rerank retriever surface *both* the
original text chunks and the caption chunks, so visual-only information
(part positions, arrow directions, control-panel icon shapes) becomes
retrievable as text and keeps citations (``figure_ref``).

Captions are cached to ``data/week9_captions.json`` so rebuilding the store
never re-calls the VLM.
"""

import base64
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .rasterize import RasterizedPage

CAPTION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"  # matches C3 store
MM_COLLECTION = "lg_manuals_mm"

# Caption prompt v2 (Week 9 rework, WEEK9_DEFECTS E5). The v1 prompt forced a
# fixed slot list (parts/arrows/icons/table), which made the model invent
# panels and tables on pages that had none and confabulate callout mappings.
# v2: describe only what exists, transcribe callout labels verbatim from the
# printed list, and describe icon shapes (explicitly checking for slashes).
CAPTION_PROMPT = """이 이미지는 LG 가전제품 한국어 사용설명서의 한 페이지입니다.
TODO: BBOX랑 주변 텍스트를 우선 인식하라고 하고 그거에 맞게 json 만들어달라.
이 페이지의 그림·도식·아이콘·표를 시각적으로 설명하세요. 본문 단락 텍스트는 옮기지 마세요.

규칙:
- 페이지에 **실제로 있는 요소만** 설명하세요. 없는 요소(제어판, 표, 화살표 등)는 언급 자체를 하지 마세요. 그림이 전혀 없는 페이지면 "그림 없음"이라고만 답하세요.
- 콜아웃 번호(①②③… 또는 ❶❷❸…)가 있으면, 페이지에 **인쇄된 라벨/설명 텍스트를 찾아 그대로** 옮겨 적으세요. 라벨을 찾을 수 없으면 "라벨 없음"이라 쓰고, 번호가 무엇을 가리키는지 추측하지 마세요.
- 아이콘/기호는 명칭을 추측하지 말고 **보이는 모양**을 묘사하세요: 기본 도형(원/사각형/부채꼴/물방울 등), 내부 요소(점/선/숫자/기호), 그리고 **빗금·사선·X 표시가 있는지 반드시 확인**해서 적으세요.
- 화살표는 무엇을 어느 방향으로 움직이는지(끼움/회전/분리) 적으세요.
- 표가 있으면 열 제목과 각 행의 내용을 그대로 옮기고, 표 안에 아이콘이 있으면 그 모양도 묘사하세요.
- 확실하지 않은 세부는 "판별 불가"로 적으세요. 없는 것을 만들어내지 마세요."""


@dataclass
class PageCaption:
    """A VLM caption for one rasterized page."""

    category: str
    complexity: str
    page: int
    image_path: str
    caption: str


def _encode_image(image_path: Path) -> str:
    """Base64-encode a PNG for the OpenAI vision API."""
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _make_client() -> OpenAI:
    """OpenAI client with generous retries — high-detail page images cost
    ~26k tokens each, so a 252-page run is TPM-bound (~8 img/min on a 200k
    TPM tier) and 429s are routine; the SDK backs off and self-throttles."""
    return OpenAI(max_retries=8)


# Deterministic pacing: high-detail page images cost ~26k tokens, so a 200k
# TPM tier allows ~7.7 starts/min. We space request *starts* by MIN_INTERVAL
# (shared across worker threads) to stay under the limit and avoid 429 churn.
MIN_INTERVAL = 9.0  # seconds between starts (~6.7/min -> ~174k TPM, safe margin)
_pace_lock = Lock()
_last_start = [0.0]


def _pace(interval: float = MIN_INTERVAL) -> None:
    """Block until at least ``interval`` has elapsed since the last start."""
    with _pace_lock:
        wait = interval - (time.monotonic() - _last_start[0])
        if wait > 0:
            time.sleep(wait)
        _last_start[0] = time.monotonic()


# gpt-4o bills page images at ~1.1k tokens (vs gpt-4o-mini's ~37k for the
# same image), so it needs far less pacing between call starts.
PACE_BY_MODEL = {"gpt-4o-mini": MIN_INTERVAL, "gpt-4o": 1.5}


def caption_page(image_path: Path, client: OpenAI | None = None, model: str = CAPTION_MODEL) -> str:
    """Generate a structured Korean caption for one page image."""
    client = client or _make_client()
    _pace(PACE_BY_MODEL.get(model, MIN_INTERVAL))
    b64 = _encode_image(image_path)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url",
                 # detail="high": manuals pack small icons/arrows that vanish
                 # under the default low-detail (512px) downsample — confirmed
                 # in a smoke test where the VLM reported "no diagram" on pages
                 # that clearly contain figures.
                 "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ],
        }],
    )
    return response.choices[0].message.content.strip()


def caption_pages(
    pages: list[RasterizedPage],
    cache_path: Path,
    client: OpenAI | None = None,
    force: bool = False,
    max_workers: int = 4,
    model: str = CAPTION_MODEL,
) -> list[PageCaption]:
    """Caption every page, caching results to ``cache_path`` (JSON).

    Resumable: pages already in the cache are skipped unless ``force``.
    Captions are fetched concurrently (the OpenAI client is thread-safe) so
    the full 252-page run finishes inside a single foreground window; the
    cache is flushed under a lock after each completion so a crash never
    loses paid-for captions.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock

    client = client or _make_client()

    # Key the cache on the invariant (category, complexity, page) tuple, NOT the
    # filesystem path: the path is relative when built via `python -m
    # src.multimodal` (cwd=repo root) but absolute when a notebook builds it
    # from ROOT, and a path-keyed cache would miss and re-caption all 252 pages.
    def _key(p) -> tuple[str, str, int]:
        return (p.category, p.complexity, p.page)

    cached: dict[tuple[str, str, int], PageCaption] = {}
    if cache_path.exists() and not force:
        for row in json.loads(cache_path.read_text()):
            pc = PageCaption(**row)
            cached[_key(pc)] = pc

    todo = [p for p in pages if _key(p) not in cached]
    print(f"  {len(cached)} cached, {len(todo)} to caption ({max_workers} workers)")

    lock = Lock()
    done = [0]

    def _flush() -> None:
        cache_path.write_text(json.dumps(
            [asdict(c) for c in cached.values()], ensure_ascii=False, indent=2,
        ))

    def _work(page: RasterizedPage) -> PageCaption | None:
        try:
            caption = caption_page(page.path, client, model=model)
        except Exception as exc:  # keep the batch alive; retry next window
            print(f"  FAILED {page.category}_{page.complexity} p{page.page}: {exc}")
            return None
        return PageCaption(
            category=page.category, complexity=page.complexity,
            page=page.page, image_path=str(page.path), caption=caption,
        )

    if todo:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_work, p): p for p in todo}
            for future in as_completed(futures):
                pc = future.result()
                if pc is None:
                    continue
                with lock:
                    cached[_key(pc)] = pc
                    done[0] += 1
                    _flush()
                    print(f"  captioned [{done[0]}/{len(todo)}] {pc.category}_{pc.complexity} p{pc.page}")

    missing = [p for p in pages if _key(p) not in cached]
    if missing:
        print(f"  {len(missing)} pages still missing — re-run to finish (resumable).")

    # Return captions in the original page order (skip any still-missing).
    return [cached[_key(p)] for p in pages if _key(p) in cached]


def caption_to_document(pc: PageCaption) -> Document:
    """Turn a page caption into an 'image-derived' LangChain chunk.

    Metadata mirrors the C3 text-chunk schema (``category``/``complexity``/
    ``page``) so the same retriever, source tagging, and citation regex work,
    plus ``modality`` and ``figure_ref`` for cross-modal provenance.
    """
    figure_ref = f"{pc.category}_{pc.complexity} p.{pc.page} fig:page"
    return Document(
        page_content=pc.caption,
        metadata={
            "chunk_id": f"{pc.category}_{pc.complexity}_p{pc.page:03d}_caption",
            "source": Path(pc.image_path).name,
            "category": pc.category,
            "complexity": pc.complexity,
            "page": pc.page,
            "section": "",
            "modality": "image-derived",
            "figure_ref": figure_ref,
        },
    )


def _load_c3_documents(c3_dir: Path, c3_collection: str) -> list[Document]:
    """Read the existing C3 text chunks straight out of their Chroma store."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    c3 = Chroma(
        collection_name=c3_collection,
        persist_directory=str(c3_dir),
        embedding_function=embeddings,
    )
    raw = c3._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata={**meta, "modality": "text"})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    return docs


def build_mm_store(
    captions: list[PageCaption],
    persist_dir: Path,
    c3_dir: Path = Path("data/chroma_db_c3"),
    c3_collection: str = "lg_manuals_c3",
    collection_name: str = MM_COLLECTION,
) -> Chroma:
    """Build the cross-modal store = C3 text chunks + image-derived captions.

    The result is a drop-in for the C0 store: same embedding model, same
    metadata keys, plus caption chunks the C0 store never had.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    text_docs = _load_c3_documents(c3_dir, c3_collection)
    caption_docs = [caption_to_document(c) for c in captions]
    all_docs = text_docs + caption_docs
    print(f"mm store: {len(text_docs)} text chunks + {len(caption_docs)} caption chunks = {len(all_docs)}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )
    print(f"Created mm store with {store._collection.count()} documents")
    return store


def load_mm_store(
    persist_dir: Path = Path("data/chroma_db_mm"),
    collection_name: str = MM_COLLECTION,
) -> Chroma:
    """Load the persisted cross-modal store."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


if __name__ == "__main__":
    from .rasterize import rasterize_all

    PDF_DIR = Path("data/raw_pdfs")
    IMG_ROOT = Path("data/sample_images")
    CACHE = Path("data/week9_captions.json")
    MM_DIR = Path("data/chroma_db_mm")

    print("Rasterizing (or reusing) pages...")
    pages = rasterize_all(PDF_DIR, IMG_ROOT)

    print(f"\nCaptioning {len(pages)} pages with {CAPTION_MODEL}...")
    captions = caption_pages(pages, CACHE)

    if len(captions) < len(pages):
        print(f"\n{len(captions)}/{len(pages)} captioned — re-run to finish before building store.")
    else:
        print(f"\nBuilding cross-modal store at {MM_DIR}...")
        build_mm_store(captions, MM_DIR)
