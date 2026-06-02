"""
Chunking utilities with proper metadata population.

Key improvement over Week 4: Each chunk includes:
- section: The section header this chunk belongs to
- image_ids: List of image markers that appeared in this chunk's source text
"""

from pathlib import Path
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .parsing import (
    ParsedDocument,
    parse_pdf,
    parse_pdf_by_page,
    find_current_section,
    IMAGE_MARKER_PATTERN,
)


@dataclass
class Chunk:
    """Represents a text chunk with full metadata."""
    text: str
    chunk_id: str
    source: str
    category: str
    complexity: str
    page: int | None
    section: str | None
    chunk_index: int
    char_count: int
    image_count: int  # Number of images in the source region
    image_markers: list[dict]  # Actual image info from source region


def chunk_document(
    parsed_doc: ParsedDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Chunk a parsed document with proper metadata.

    Each chunk includes:
    - section: Inherited from the section header above it
    - image_count: Number of images in the original text region
    - image_markers: Details of images that were in this region
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    chunk_idx = 0

    for page in parsed_doc.pages:
        # Split the clean text
        chunk_texts = splitter.split_text(page.clean_text)

        # We need to map chunk positions back to original text
        # to determine which images belong to which chunks
        # This is approximate since clean_text has markers removed

        # Build a mapping of approximate positions
        # For each chunk, find its start position in clean_text
        # then find the corresponding position in original text

        original_search_pos = 0
        for chunk_text in chunk_texts:
            # Find chunk's actual position in original text using text anchor search
            # Use first 50 chars as anchor (clean text, so should exist in original)
            anchor_len = min(50, len(chunk_text))
            anchor = chunk_text[:anchor_len]

            # Search for anchor in original text
            anchor_pos = page.text.find(anchor, original_search_pos)
            if anchor_pos == -1:
                # Fallback: try shorter anchor
                anchor = chunk_text[:20]
                anchor_pos = page.text.find(anchor, original_search_pos)
            if anchor_pos == -1:
                # Last resort: use ratio estimation
                ratio = len(page.text) / max(len(page.clean_text), 1)
                clean_pos = page.clean_text.find(chunk_text)
                anchor_pos = int(clean_pos * ratio) if clean_pos != -1 else original_search_pos

            chunk_start_original = anchor_pos
            # Estimate end position (chunk length + some buffer for markdown)
            chunk_end_original = chunk_start_original + int(len(chunk_text) * 1.2)
            original_search_pos = chunk_start_original + len(anchor)  # Move forward for next search

            # Find images in this region
            chunk_images = []
            for img in page.image_markers:
                if chunk_start_original <= img["position"] <= chunk_end_original:
                    chunk_images.append(img)

            # Find section for this chunk position
            section = find_current_section(page.text, chunk_start_original)
            if section is None:
                section = page.section  # Fall back to page-level section

            chunk_id = f"{parsed_doc.category}_{parsed_doc.complexity}"
            if page.page_num:
                chunk_id += f"_p{page.page_num:03d}"
            chunk_id += f"_c{chunk_idx:03d}"

            all_chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                source=parsed_doc.source,
                category=parsed_doc.category,
                complexity=parsed_doc.complexity,
                page=page.page_num,
                section=section,
                chunk_index=chunk_idx,
                char_count=len(chunk_text),
                image_count=len(chunk_images),
                image_markers=chunk_images,
            ))
            chunk_idx += 1

    return all_chunks


def chunk_pdf(
    pdf_path: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    by_page: bool = False,
) -> list[Chunk]:
    """
    Parse and chunk a PDF file in one step.

    Args:
        pdf_path: Path to PDF file
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        by_page: If True, parse page-by-page (slower but more granular metadata)
    """
    if by_page:
        parsed = parse_pdf_by_page(pdf_path)
    else:
        parsed = parse_pdf(pdf_path)

    return chunk_document(parsed, chunk_size, chunk_overlap)


def chunks_to_langchain_docs(chunks: list[Chunk]):
    """Convert Chunk objects to LangChain Document format."""
    from langchain_core.documents import Document

    docs = []
    for chunk in chunks:
        # Convert image markers to comma-separated string (Chroma doesn't support lists)
        image_ids_str = ",".join(
            f"img_{i['width']}x{i['height']}" for i in chunk.image_markers
        ) if chunk.image_markers else ""

        metadata = {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "category": chunk.category,
            "complexity": chunk.complexity,
            "page": chunk.page,
            "section": chunk.section or "",  # Chroma doesn't like None
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            "image_count": chunk.image_count,
            "image_ids": image_ids_str,
        }
        docs.append(Document(page_content=chunk.text, metadata=metadata))

    return docs
