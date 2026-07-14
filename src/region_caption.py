"""
Region-crop captions (WEEK11_TASKS §6) — the Week 9 bottleneck breaker.

Week 9/11 both showed the same ceiling: a *full page* (index-side caption or
query-side vision read) never yields icon shapes, part positions, or
directions. This module goes one level down: detect figure regions on a page,
render each region as its own high-res image (the icon fills the frame), and
caption it into structured JSON.

Pipeline:
    1. detect_regions   — occupancy grid over pymupdf vector drawings
                          (figures are vector line-art, LIM-002: no raster
                          XObjects to extract) → connected components → bboxes.
    2. render_crops     — render each bbox at 300dpi straight from the PDF
                          (vector → sharp at any dpi; the 150dpi full-page
                          limit does not apply to clips).
    3. caption_region   — gpt-4o reads one crop + nearby printed text and
                          returns structured JSON:
                          {summary, elements: [{label, shape,
                           position_in_figure, direction}]}.
                          Page-relative position (상/하/좌/우) is computed
                          geometrically — never asked from the VLM.

Caching mirrors week9 captions: JSON keyed by (category, complexity, page,
region_idx); re-runs never re-pay for finished regions.

MVP scope: the IR8 reference pages (rasterize.IR8_REFERENCE_PAGES). Store
integration (`region_caption_to_document`) is provided but the mm-store
rebuild + IR8 re-evaluation belong to Week 12 (then ADR-012).

Run:  .venv/bin/python -m src.region_caption
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pymupdf
from openai import OpenAI
from scipy import ndimage

from .parsing import parse_filename

CAPTION_MODEL = "gpt-4o"
CROP_DPI = 300  # vector art renders sharp at any dpi — icons fill the frame
GRID_CELL_PT = 4.0
DILATE_ITERS = 2  # bridges small gaps between strokes of one figure
MIN_AREA_RATIO = 0.05  # measured: drops header banners (~4%), keeps figures
PAGE_WIDE_RATIO = 0.5  # per-path filter: background/frame rects
CONTEXT_MARGIN_PT = 30.0
CONTEXT_MAX_CHARS = 600

CROPS_ROOT = Path("data/region_crops")
CACHE_PATH = Path("data/week11_captions_crop.json")


@dataclass
class RegionCrop:
    """One rendered figure region."""

    category: str
    complexity: str
    page: int  # 1-indexed
    region_idx: int
    bbox_pdf: tuple[float, float, float, float]  # PDF points (x0, y0, x1, y1)
    position: str  # page-relative, computed (not VLM-guessed)
    image_path: str
    context_text: str  # printed text near the region (label source)


# ---------------------------------------------------------------------------
# 1. Region detection — occupancy grid over vector drawings
# ---------------------------------------------------------------------------


def detect_regions(
    page: pymupdf.Page,
    cell: float = GRID_CELL_PT,
    dilate: int = DILATE_ITERS,
    min_area_ratio: float = MIN_AREA_RATIO,
) -> list[tuple[float, float, float, float]]:
    """Figure-region bboxes (PDF points) from the page's vector drawings.

    Pairwise clustering is infeasible (~9k paths/page measured); instead mark
    a coarse grid cell occupied when any drawing rect covers it, dilate to
    bridge gaps, and take connected components above ``min_area_ratio``.
    """
    width, height = page.rect.width, page.rect.height
    gw, gh = int(width // cell) + 1, int(height // cell) + 1
    grid = np.zeros((gh, gw), dtype=bool)

    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.width * rect.height > PAGE_WIDE_RATIO * width * height:
            continue  # background / page frame
        x0, y0 = max(0, int(rect.x0 // cell)), max(0, int(rect.y0 // cell))
        x1 = min(gw - 1, int(rect.x1 // cell))
        y1 = min(gh - 1, int(rect.y1 // cell))
        grid[y0:y1 + 1, x0:x1 + 1] = True

    labels, count = ndimage.label(ndimage.binary_dilation(grid, iterations=dilate))
    boxes: list[tuple[float, float, float, float]] = []
    for i in range(1, count + 1):
        ys, xs = np.nonzero(labels == i)
        bbox = (
            xs.min() * cell, ys.min() * cell,
            (xs.max() + 1) * cell, (ys.max() + 1) * cell,
        )
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area / (width * height) >= min_area_ratio:
            boxes.append(bbox)
    # top-to-bottom, left-to-right so region_idx is stable across runs
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def position_label(bbox: tuple, page_rect: pymupdf.Rect) -> str:
    """Deterministic page-relative position (상/중/하 × 좌/중앙/우)."""
    cx = (bbox[0] + bbox[2]) / 2 / page_rect.width
    cy = (bbox[1] + bbox[3]) / 2 / page_rect.height
    vertical = "상단" if cy < 1 / 3 else ("중단" if cy < 2 / 3 else "하단")
    horizontal = "좌측" if cx < 1 / 3 else ("중앙" if cx < 2 / 3 else "우측")
    return f"페이지 {vertical} {horizontal}"


def nearby_text(page: pymupdf.Page, bbox: tuple, margin: float = CONTEXT_MARGIN_PT) -> str:
    """Printed text blocks that touch the region (expanded by ``margin``) —
    the source of verbatim callout labels for the caption prompt."""
    zone = pymupdf.Rect(
        bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin
    )
    parts = [
        block[4].strip()
        for block in page.get_text("blocks")
        if pymupdf.Rect(block[:4]).intersects(zone) and block[4].strip()
    ]
    return " / ".join(parts)[:CONTEXT_MAX_CHARS]


# ---------------------------------------------------------------------------
# 2. Crop rendering — 300dpi clips straight from the vector PDF
# ---------------------------------------------------------------------------


def render_crops(
    pdf_path: Path,
    page_num: int,
    out_root: Path = CROPS_ROOT,
    dpi: int = CROP_DPI,
) -> list[RegionCrop]:
    """Detect + render every figure region of one page (1-indexed)."""
    meta = parse_filename(pdf_path.name)
    category, complexity = meta["category"], meta["complexity"]
    out_dir = out_root / f"{category}_{complexity}"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    try:
        page = doc[page_num - 1]
        crops: list[RegionCrop] = []
        for idx, bbox in enumerate(detect_regions(page)):
            out_path = out_dir / f"p{page_num:03d}_r{idx}.png"
            page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=dpi).save(str(out_path))
            crops.append(RegionCrop(
                category=category,
                complexity=complexity,
                page=page_num,
                region_idx=idx,
                bbox_pdf=tuple(round(v, 1) for v in bbox),
                position=position_label(bbox, page.rect),
                image_path=str(out_path),
                context_text=nearby_text(page, bbox),
            ))
        return crops
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 3. Structured caption — gpt-4o on one region
# ---------------------------------------------------------------------------


# v2 principles kept (describe only what exists, transcribe labels verbatim,
# always check hatching/slashes), narrowed to a single region and forced into
# JSON. Page position is injected, never asked.
REGION_CAPTION_PROMPT = """이 이미지는 LG 가전 한국어 매뉴얼 페이지에서 잘라낸 **하나의 그림 영역**입니다 (300dpi 고해상).
영역의 페이지 내 위치: {position}
영역 주변의 인쇄 텍스트 (라벨 대조용): {context}

이 영역만 보고, JSON으로만 답하세요 (다른 텍스트 없이):
{{"summary": "이 영역이 보여주는 것 한 줄",
  "elements": [
    {{"label": "인쇄된 라벨/콜아웃 텍스트 원문 그대로 (없으면 null)",
      "shape": "보이는 모양 묘사 — 기본 도형(원/사각형/부채꼴 등), 내부 요소(점/선/기호), 빗금·사선·X 표시 여부를 반드시 확인해 적기",
      "position_in_figure": "영역 안에서의 위치 (좌/우/상/하/중앙 조합)",
      "direction": "화살표가 있으면 무엇을 어느 방향으로 움직이는지(끼움/회전/분리), 없으면 null"}}
  ]}}

규칙:
- 실제로 보이는 요소만. 없는 것을 만들어내지 마세요.
- label은 추측 금지 — 인쇄된 텍스트를 그대로 옮기고, 없으면 null.
- 확실하지 않은 세부는 "판별 불가"라고 적으세요."""


def caption_region(crop: RegionCrop, client: OpenAI, model: str = CAPTION_MODEL) -> dict:
    """One structured caption. Raises on API failure (caller keeps the batch
    alive); returns ``{"parse_error": raw}`` if the JSON contract is broken."""
    import base64

    b64 = base64.b64encode(Path(crop.image_path).read_bytes()).decode("utf-8")
    prompt = REGION_CAPTION_PROMPT.format(
        position=crop.position, context=crop.context_text or "(없음)"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
            ],
        }],
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": raw}


def caption_crops(
    crops: list[RegionCrop],
    cache_path: Path = CACHE_PATH,
    client: OpenAI | None = None,
    force: bool = False,
) -> list[dict]:
    """Caption all crops with a (category, complexity, page, region_idx)-keyed
    cache, flushed after every region (same discipline as week9 captions)."""
    client = client or OpenAI(max_retries=8)

    def _key(row: dict) -> tuple:
        return (row["category"], row["complexity"], row["page"], row["region_idx"])

    cached: dict[tuple, dict] = {}
    if cache_path.exists() and not force:
        for row in json.loads(cache_path.read_text()):
            cached[_key(row)] = row

    rows: list[dict] = []
    for crop in crops:
        row = {**asdict(crop)}
        key = _key(row)
        if key in cached:
            rows.append(cached[key])
            continue
        row["caption"] = caption_region(crop, client)
        cached[key] = row
        rows.append(row)
        cache_path.write_text(
            json.dumps(list(cached.values()), ensure_ascii=False, indent=2)
        )
        print(f"  captioned {row['category']}_{row['complexity']} "
              f"p{row['page']} r{row['region_idx']}", flush=True)
    return rows


def region_caption_to_document(row: dict):
    """Region caption → 'image-derived' chunk (mm-store forward compat).

    Same metadata contract as week9 page captions plus region fields; the
    Week 12 store rebuild ingests these without retriever changes.
    """
    from langchain_core.documents import Document

    caption = row["caption"]
    lines = [f"[그림 영역 — {row['position']}] {caption.get('summary', '')}"]
    for el in caption.get("elements", []):
        parts = [p for p in (
            el.get("label"),
            el.get("shape"),
            el.get("position_in_figure"),
            el.get("direction"),
        ) if p]
        if parts:
            lines.append(" · ".join(str(p) for p in parts))
    figure_ref = (f"{row['category']}_{row['complexity']} "
                  f"p.{row['page']} fig:r{row['region_idx']}")
    return Document(
        page_content="\n".join(lines),
        metadata={
            "chunk_id": (f"{row['category']}_{row['complexity']}"
                         f"_p{row['page']:03d}_r{row['region_idx']}_caption"),
            "source": Path(row["image_path"]).name,
            "category": row["category"],
            "complexity": row["complexity"],
            "page": row["page"],
            "section": "",
            "modality": "image-derived",
            "figure_ref": figure_ref,
            "region_bbox": json.dumps(row["bbox_pdf"]),
            "region_position": row["position"],
        },
    )


# ---------------------------------------------------------------------------
# MVP driver — IR8 reference pages
# ---------------------------------------------------------------------------


# (manual dirname, pages) derived from rasterize.IR8_REFERENCE_PAGES
IR8_TARGETS = {
    "airpurifier_complex": [18, 22],
    "waterpurifier_complex": [29],
    "vacuumcleaner_complex": [15, 19],
}


def _pdf_for(dirname: str, pdf_dir: Path = Path("data/raw_pdfs")) -> Path:
    """Map manual dirname to its PDF (filenames carry extra suffixes/typos —
    e.g. 'vaccumcleaner' — so match via parse_filename, not string prefix)."""
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        meta = parse_filename(pdf.name)
        if f"{meta['category']}_{meta['complexity']}" == dirname:
            return pdf
    raise FileNotFoundError(f"no PDF for {dirname} in {pdf_dir}")


if __name__ == "__main__":
    all_crops: list[RegionCrop] = []
    for dirname, pages in IR8_TARGETS.items():
        pdf = _pdf_for(dirname)
        for page_num in pages:
            crops = render_crops(pdf, page_num)
            print(f"{dirname} p{page_num}: {len(crops)} regions")
            all_crops.extend(crops)

    print(f"\nCaptioning {len(all_crops)} regions with {CAPTION_MODEL}...")
    rows = caption_crops(all_crops)
    print(f"\nwrote {CACHE_PATH} ({len(rows)} regions)")
