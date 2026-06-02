"""
PDF parsing utilities with metadata extraction.

Key improvement over Week 4: Extract image markers and section headers
BEFORE stripping markdown artifacts.
"""

import re
from pathlib import Path
from dataclasses import dataclass

import pymupdf4llm


@dataclass
class ParsedPage:
    """Represents a parsed page with extracted metadata."""
    page_num: int
    text: str
    clean_text: str
    section: str | None
    image_markers: list[dict]  # [{"marker": str, "position": int, "dimensions": str}]


@dataclass
class ParsedDocument:
    """Represents a fully parsed PDF document."""
    source: str
    category: str
    complexity: str
    pages: list[ParsedPage]
    sections: list[str]  # All unique sections found


# Regex patterns
IMAGE_MARKER_PATTERN = re.compile(r'\*\*==> picture \[(\d+) x (\d+)\] intentionally omitted <==\*\*')
SECTION_HEADER_PATTERN = re.compile(r'^##\s+\*?\*?(.+?)\*?\*?\s*$', re.MULTILINE)

# Noise patterns to filter out from sections
# These are ## headers that aren't meaningful sections
SECTION_NOISE_PATTERNS = [
    re.compile(r'^\d'),           # Starts with number: "5 필터가..."
    re.compile(r'^-\s'),          # Starts with dash: "- 교체 주기..."
    re.compile(r'^[a-zA-Z]\s'),   # Starts with single letter: "R 금지", "j 준수"
    re.compile(r'^[*>]'),         # Starts with * or >
    re.compile(r'모델명'),         # Model names
    re.compile(r'^WD\d'),         # Model number patterns
    re.compile(r'^AS\d'),         # Model number patterns
    re.compile(r'^\|'),           # Table rows
]

# Generic headers that repeat many times - keep only first occurrence contextually
GENERIC_SECTION_NAMES = {
    "경고", "주의", "알아두기", "알아두면 좋은 정보",
    "R 금지 사항", "j 준수 사항", "금지 사항", "준수 사항",
}


def is_valid_section(section_name: str) -> bool:
    """
    Check if a section name is a meaningful section header.

    Filters out:
    - Numbered steps (5 필터가...)
    - Model names (모델명: WD523A...)
    - Special character prefixes (R 금지, j 준수, - 교체...)
    - Generic repeated headers (경고, 주의)
    """
    if not section_name:
        return False

    # Check against noise patterns
    for pattern in SECTION_NOISE_PATTERNS:
        if pattern.search(section_name):
            return False

    # Filter out generic names that appear multiple times
    if section_name in GENERIC_SECTION_NAMES:
        return False

    # Must have at least 2 Korean characters to be meaningful
    korean_chars = len(re.findall(r'[가-힣]', section_name))
    if korean_chars < 2:
        return False

    return True


def parse_filename(filename: str) -> dict:
    """
    Parse metadata from filename.

    Handles both 'vacuumcleaner' and 'vaccumcleaner' (typo in actual files).
    """
    name = Path(filename).stem.lower()

    categories = ["waterpurifier", "airpurifier", "vacuumcleaner", "vaccumcleaner"]
    category = "unknown"
    for cat in categories:
        if cat in name:
            # Normalize vaccumcleaner typo
            category = "vacuumcleaner" if cat == "vaccumcleaner" else cat
            break

    complexity = "complex" if "complex" in name else "simple"

    return {"category": category, "complexity": complexity}


def extract_image_markers(text: str) -> list[dict]:
    """
    Extract image marker information before removing them.

    Returns list of dicts with:
    - marker: The full marker string
    - position: Character position in text
    - width: Image width
    - height: Image height
    """
    markers = []
    for match in IMAGE_MARKER_PATTERN.finditer(text):
        markers.append({
            "marker": match.group(0),
            "position": match.start(),
            "width": int(match.group(1)),
            "height": int(match.group(2)),
        })
    return markers


def extract_sections(text: str) -> list[str]:
    """
    Extract section headers from markdown text.

    Looks for patterns like:
    - ## 안전을 위해 주의하기
    - ## **사용하기**

    Filters out noise patterns (model names, numbered steps, etc.)
    """
    sections = []
    for match in SECTION_HEADER_PATTERN.finditer(text):
        section_name = match.group(1).strip()
        # Remove any remaining bold markers
        section_name = section_name.replace('**', '').strip()
        if section_name and section_name not in sections and is_valid_section(section_name):
            sections.append(section_name)
    return sections


def find_current_section(text: str, position: int) -> str | None:
    """
    Find the section header that applies to a given position in text.

    Searches backwards from position to find the most recent VALID ## header.
    Skips noise patterns (model names, numbered steps, etc.)
    """
    text_before = text[:position]
    matches = list(SECTION_HEADER_PATTERN.finditer(text_before))
    # Search backwards through matches to find most recent valid section
    for match in reversed(matches):
        section_name = match.group(1).strip().replace('**', '').strip()
        if is_valid_section(section_name):
            return section_name
    return None


def clean_markdown(text: str) -> str:
    """
    Remove markdown artifacts for cleaner embeddings.

    IMPORTANT: Call extract_image_markers() BEFORE this function
    if you need to preserve image information.
    """
    # Remove image placeholders
    text = IMAGE_MARKER_PATTERN.sub('', text)
    # Remove bold/italic markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    # Simplify table separators
    text = re.sub(r'\|[-:]+\|', '', text)
    # Remove <br> tags
    text = text.replace('<br>', ' ')
    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def parse_pdf(pdf_path: Path) -> ParsedDocument:
    """
    Parse a PDF file with full metadata extraction.

    Uses pymupdf4llm for layout-aware markdown conversion,
    then extracts:
    - Image markers with positions
    - Section headers
    - Clean text for embedding
    """
    # Get raw markdown
    md_text = pymupdf4llm.to_markdown(str(pdf_path))

    # Parse filename for category/complexity
    meta = parse_filename(pdf_path.name)

    # Extract all sections from full document
    all_sections = extract_sections(md_text)

    # Extract image markers before cleaning
    image_markers = extract_image_markers(md_text)

    # For now, treat entire document as single "page" since pymupdf4llm
    # combines all pages into one markdown string
    # TODO: Consider per-page parsing if needed

    # Determine section for the start of document
    current_section = all_sections[0] if all_sections else None

    page = ParsedPage(
        page_num=1,
        text=md_text,
        clean_text=clean_markdown(md_text),
        section=current_section,
        image_markers=image_markers,
    )

    return ParsedDocument(
        source=pdf_path.name,
        category=meta["category"],
        complexity=meta["complexity"],
        pages=[page],
        sections=all_sections,
    )


def parse_pdf_by_page(pdf_path: Path) -> ParsedDocument:
    """
    Parse a PDF file page by page with metadata extraction.

    This version preserves page boundaries for more granular metadata.
    """
    meta = parse_filename(pdf_path.name)

    # Get page count
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    doc.close()

    pages = []
    all_sections = []
    current_section = None

    for page_num in range(page_count):
        # Parse single page
        md_text = pymupdf4llm.to_markdown(str(pdf_path), pages=[page_num])

        # Extract sections from this page
        page_sections = extract_sections(md_text)
        for s in page_sections:
            if s not in all_sections:
                all_sections.append(s)

        # Update current section if new one found
        if page_sections:
            current_section = page_sections[-1]

        # Extract image markers
        image_markers = extract_image_markers(md_text)

        pages.append(ParsedPage(
            page_num=page_num + 1,
            text=md_text,
            clean_text=clean_markdown(md_text),
            section=current_section,
            image_markers=image_markers,
        ))

    return ParsedDocument(
        source=pdf_path.name,
        category=meta["category"],
        complexity=meta["complexity"],
        pages=pages,
        sections=all_sections,
    )
