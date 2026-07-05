"""
C2 — CLIP image-embedding retrieval (Week 9, WEEK9_TASKS §5).

Joint text-image space (both from sentence-transformers, local):
- images  -> ``clip-ViT-B-32``                       (image encoder)
- queries -> ``clip-ViT-B-32-multilingual-v1``        (Korean-capable text
             encoder distilled into the *same* ViT-B-32 space)

So a Korean question embeds into the same space as the page images and we can
rank pages by cosine similarity. This is the "find a similar picture" path —
it retrieves pages but does **not** read fine detail (that needs a VLM, C2b).

Index is plain numpy + cosine (small: 252 pages × 512-d), persisted as
``embeddings.npy`` + ``meta.json``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rasterize import RasterizedPage

IMAGE_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

_image_model = None
_text_model = None


def _get_image_model():
    """Lazy-load the CLIP image encoder (cached at module level)."""
    global _image_model
    if _image_model is None:
        from sentence_transformers import SentenceTransformer
        _image_model = SentenceTransformer(IMAGE_MODEL, device="cpu")
    return _image_model


def _get_text_model():
    """Lazy-load the multilingual CLIP text encoder (cached at module level)."""
    global _text_model
    if _text_model is None:
        from sentence_transformers import SentenceTransformer
        _text_model = SentenceTransformer(TEXT_MODEL, device="cpu")
    return _text_model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


@dataclass
class ClipIndex:
    """Page-image CLIP index: row-aligned embeddings + metadata."""

    embeddings: np.ndarray  # (N, 512) float32, L2-normalized
    metas: list[dict]       # row-aligned: {category, complexity, page, image_path}

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        (out_dir / "meta.json").write_text(
            json.dumps(self.metas, ensure_ascii=False, indent=2)
        )

    @classmethod
    def load(cls, out_dir: Path) -> "ClipIndex":
        embeddings = np.load(out_dir / "embeddings.npy")
        metas = json.loads((out_dir / "meta.json").read_text())
        return cls(embeddings=embeddings, metas=metas)


def build_clip_index(
    pages: list[RasterizedPage],
    out_dir: Path = Path("data/clip_index"),
    batch_size: int = 16,
) -> ClipIndex:
    """Embed every page image with CLIP and persist the index."""
    from PIL import Image

    model = _get_image_model()
    images = [Image.open(p.path).convert("RGB") for p in pages]
    embeddings = model.encode(
        images, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True,
    )
    embeddings = _normalize(np.asarray(embeddings, dtype=np.float32))

    metas = [
        {
            "category": p.category,
            "complexity": p.complexity,
            "page": p.page,
            "image_path": str(p.path),
        }
        for p in pages
    ]
    index = ClipIndex(embeddings=embeddings, metas=metas)
    index.save(out_dir)
    print(f"Built CLIP index: {len(metas)} pages -> {out_dir}")
    return index


def clip_retrieve(index: ClipIndex, query: str, k: int = 5) -> list[dict]:
    """Retrieve top-k pages for a text query via joint-space cosine similarity.

    Returns a list of metadata dicts (best first), each with a ``score`` field.
    """
    model = _get_text_model()
    q = model.encode([query], convert_to_numpy=True)
    q = _normalize(np.asarray(q, dtype=np.float32))[0]

    scores = index.embeddings @ q  # cosine (both normalized)
    top = np.argsort(-scores)[:k]
    return [{**index.metas[i], "score": float(scores[i])} for i in top]


if __name__ == "__main__":
    from .rasterize import rasterize_all

    PDF_DIR = Path("data/raw_pdfs")
    IMG_ROOT = Path("data/sample_images")
    INDEX_DIR = Path("data/clip_index")

    pages = rasterize_all(PDF_DIR, IMG_ROOT)
    index = build_clip_index(pages, INDEX_DIR)

    # Smoke test: an IR8 query should surface its reference page region.
    demo = "무선청소기 상태 표시창에서 Wi-Fi 연결 끊김 아이콘"
    print(f"\nQuery: {demo}")
    for hit in clip_retrieve(index, demo, k=5):
        print(f"  {hit['category']}_{hit['complexity']} p{hit['page']}  score={hit['score']:.3f}")
