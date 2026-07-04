"""
Chunking Module
---------------
Responsible for:
  - Splitting page-level text into smaller overlapping chunks
  - Assigning a unique chunk_id to each chunk
  - Preserving page_number and document_name metadata for source attribution

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph/sentence boundaries first before falling back to hard cuts —
this keeps chunks semantically coherent instead of cutting mid-sentence.
"""

import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 1000      # characters
DEFAULT_CHUNK_OVERLAP = 150    # characters


def chunk_pages(
    pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Takes the output of pdf_loader.load_pdf() (list of page dicts) and
    returns a list of chunk dicts:

        [
            {
                "chunk_id": "uuid-string",
                "document_name": "sample.pdf",
                "page_number": 3,
                "text": "chunk text...",
                "chunk_size": 987
            },
            ...
        ]

    chunk_size and chunk_overlap are exposed as parameters so the UI
    can let the user adjust them (per assignment enhancement requirement).
    """
    if not pages:
        raise ValueError("No pages provided to chunk_pages().")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []

    for page in pages:
        page_text = page["text"]
        splits = splitter.split_text(page_text)

        for split_text in splits:
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "document_name": page["document_name"],
                "page_number": page["page_number"],
                "text": split_text,
                "chunk_size": len(split_text),
            })

    if not all_chunks:
        raise ValueError("Chunking produced no chunks — check input pages.")

    return all_chunks


if __name__ == "__main__":
    # Quick manual test — run: python src/chunker/chunker.py path/to/file.pdf
    import sys
    sys.path.append(".")
    from src.loader.pdf_loader import load_pdf

    if len(sys.argv) < 2:
        print("Usage: python chunker.py <path_to_pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages)

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print("--- Preview of chunk 0 ---")
    print(f"Chunk ID: {chunks[0]['chunk_id']}")
    print(f"Page: {chunks[0]['page_number']}")
    print(f"Size: {chunks[0]['chunk_size']} chars")
    print(f"Text: {chunks[0]['text'][:300]}")
