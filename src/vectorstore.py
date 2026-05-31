"""
Vector store utilities for creating and managing Chroma collections.
"""

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .chunking import chunk_pdf, chunks_to_langchain_docs, Chunk


def create_vectorstore(
    pdf_dir: Path,
    persist_dir: Path,
    collection_name: str = "lg_manuals",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    by_page: bool = False,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[Chroma, list[Chunk]]:
    """
    Create a Chroma vector store from PDF files.

    Args:
        pdf_dir: Directory containing PDF files
        persist_dir: Directory to persist the vector store
        collection_name: Name of the Chroma collection
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        by_page: If True, parse PDFs page-by-page
        embedding_model: OpenAI embedding model to use

    Returns:
        Tuple of (Chroma vectorstore, list of all chunks)
    """
    import chromadb

    # Delete old collection if exists
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
        print(f"Deleted old {collection_name} collection")
    except Exception:
        print(f"No existing {collection_name} collection to delete")

    # Process all PDFs
    all_chunks = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"Processing {pdf_path.name}...", end=" ")
        chunks = chunk_pdf(pdf_path, chunk_size, chunk_overlap, by_page)
        all_chunks.extend(chunks)

        # Show stats
        sections = set(c.section for c in chunks if c.section)
        images = sum(c.image_count for c in chunks)
        print(f"{len(chunks)} chunks, {len(sections)} sections, {images} images")

    print(f"\nTotal: {len(all_chunks)} chunks")

    # Convert to LangChain docs
    docs = chunks_to_langchain_docs(all_chunks)

    # Create embeddings and vectorstore
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(persist_dir),
    )

    print(f"Created vector store with {vectorstore._collection.count()} documents")

    return vectorstore, all_chunks


def load_vectorstore(
    persist_dir: Path,
    collection_name: str = "lg_manuals",
    embedding_model: str = "text-embedding-3-small",
) -> Chroma:
    """
    Load an existing Chroma vector store.

    Args:
        persist_dir: Directory where vector store is persisted
        collection_name: Name of the Chroma collection
        embedding_model: OpenAI embedding model (must match what was used to create)

    Returns:
        Chroma vectorstore
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    return vectorstore
