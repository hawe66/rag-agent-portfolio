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

# Structured caption prompt (WEEK9_TASKS §4.1). "보이는 것만" keeps the caption
# consistent with the project's refusal-over-hallucination principle.
CAPTION_PROMPT = """이 이미지는 LG 가전제품 한국어 사용설명서의 한 페이지입니다.
당신의 임무는 이 페이지에 **그려진 그림·도식·아이콘·표를 빠짐없이 시각적으로 묘사**하는 것입니다.
(본문 단락 텍스트는 그대로 옮기지 말고, 그림 옆 라벨/콜아웃/아이콘 설명에 집중하세요.)

페이지를 위에서 아래로 훑으며, 보이는 그래픽 요소마다 다음을 기술하세요:
- 제품/부품 그림: 어떤 부품이 그려졌고 화면상 **어디에**(상/하/좌/우/앞/뒤) 있는지, 콜아웃 번호(①②③…)가 무엇을 가리키는지.
- 화살표/동작 표시: 화살표가 **어느 방향**을 가리키는지(끼움/회전/분리 등 동작 방향).
- 제어판·표시창의 **아이콘/기호**: 각 아이콘이 **어떤 모양인지** 형태 그대로 묘사(선·곡선·도형·빗금 유무 등). 모양을 직접 보고 묘사할 것.
- 표: 행/열 항목과 값.

규칙:
- 이미지에서 **실제로 보이는 것만** 기술하고 추측하지 마세요. 특정 세부가 흐려 판별 불가하면 그 항목만 "판별 불가"로 적으세요.
- 그래픽 요소를 빠뜨리지 마세요. "그림이 없다"고 단정하기 전에 작은 아이콘·콜아웃·선 그림까지 살피세요."""


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


def _pace() -> None:
    """Block until at least MIN_INTERVAL has elapsed since the last start."""
    with _pace_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_start[0])
        if wait > 0:
            time.sleep(wait)
        _last_start[0] = time.monotonic()


def caption_page(image_path: Path, client: OpenAI | None = None) -> str:
    """Generate a structured Korean caption for one page image (gpt-4o-mini)."""
    client = client or _make_client()
    _pace()
    b64 = _encode_image(image_path)
    response = client.chat.completions.create(
        model=CAPTION_MODEL,
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
            caption = caption_page(page.path, client)
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
