"""
Page rasterization for Phase 2 (Week 9).

Renders each PDF page to a PNG so the cross-modal pipelines (C1 caption,
C2 CLIP) have a per-page image. We rasterize the *whole page* — region crops
are deferred to Week 10 (WEEK9_TASKS §3).

LIM-002 (ADR-011): the manual figures are almost entirely vector graphics, so
raster XObject extraction yields nothing usable. Rasterize → vision is the
confirmed Phase 2 extraction path.

Page numbering: filenames are 1-indexed (``p001.png`` == PDF page index 0),
matching ``parse_pdf_by_page`` (``page_num = index + 1``) and the golden-set
``reference_context`` page numbers.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .parsing import parse_filename

RENDER_DPI = 95  # audit confirmed 95dpi figures are legible (WEEK9_TASKS §3)


@dataclass(frozen=True)
class RasterizedPage:
    """One rendered page image."""

    category: str
    complexity: str
    page: int  # 1-indexed PDF page
    path: Path


def manual_dirname(category: str, complexity: str) -> str:
    """Directory name for a manual's images, e.g. ``airpurifier_complex``."""
    return f"{category}_{complexity}"


def rasterize_pdf(
    pdf_path: Path,
    out_root: Path,
    dpi: int = RENDER_DPI,
) -> list[RasterizedPage]:
    """Render every page of one PDF to ``out_root/{cat}_{cplx}/p{NNN}.png``.

    Returns the list of rendered pages (1-indexed). Existing PNGs are
    overwritten so re-runs stay deterministic.
    """
    meta = parse_filename(pdf_path.name)
    out_dir = out_root / manual_dirname(meta["category"], meta["complexity"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: list[RasterizedPage] = []
    doc = pymupdf.open(str(pdf_path))
    try:
        for index in range(len(doc)):
            page_num = index + 1
            pixmap = doc[index].get_pixmap(dpi=dpi)
            # TODO: 한 페이지를 여러 개의 region으로 나누고 싶으면 여기서 bbox를 지정해서 crop
            out_path = out_dir / f"p{page_num:03d}.png"
            pixmap.save(str(out_path))
            rendered.append(RasterizedPage(
                category=meta["category"],
                complexity=meta["complexity"],
                page=page_num,
                path=out_path,
            ))
    finally:
        doc.close()

    return rendered


def rasterize_all(
    pdf_dir: Path,
    out_root: Path,
    dpi: int = RENDER_DPI,
) -> list[RasterizedPage]:
    """Rasterize all PDFs in ``pdf_dir``."""
    all_pages: list[RasterizedPage] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        pages = rasterize_pdf(pdf_path, out_root, dpi)
        print(f"{pdf_path.name}: {len(pages)} pages -> {pages[0].path.parent}")
        all_pages.extend(pages)
    return all_pages


# IR8 reference pages (golden_set_v2, image-required) — used to verify
# rasterize produced the pages the evaluation depends on.
IR8_REFERENCE_PAGES: dict[str, list[int]] = {
    "airpurifier_complex": [18, 22],
    "waterpurifier_complex": [29],
    "vacuumcleaner_complex": [15, 19],
}


def verify_ir8_pages(out_root: Path) -> dict[str, bool]:
    """Check every IR8 reference page PNG exists. Returns {label: present}."""
    status: dict[str, bool] = {}
    for dirname, pages in IR8_REFERENCE_PAGES.items():
        for page in pages:
            label = f"{dirname} p{page}"
            status[label] = (out_root / dirname / f"p{page:03d}.png").exists()
    return status


if __name__ == "__main__":
    PDF_DIR = Path("data/raw_pdfs")
    OUT_ROOT = Path("data/sample_images")

    pages = rasterize_all(PDF_DIR, OUT_ROOT)
    print(f"\nTotal: {len(pages)} pages rendered at {RENDER_DPI}dpi")

    print("\n--- IR8 reference page check ---")
    status = verify_ir8_pages(OUT_ROOT)
    for label, present in status.items():
        print(f"  {'OK ' if present else 'MISSING'} {label}")
    assert all(status.values()), "Some IR8 reference pages are missing!"
    print("All IR8 reference pages present.")
