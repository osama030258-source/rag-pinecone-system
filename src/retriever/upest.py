"""
Pinecone Upsert Module
------------------------
Responsible for:
  - Taking embedded chunks (from embedder.embed_chunks)
  - Formatting them into Pinecone's vector format
  - Upserting them into the index under a namespace (one namespace per document,
    enabling multi-document support and clean separation between PDFs)

Metadata stored per vector:
  - document_name
  - page_number
  - chunk_id
  - text (needed at query time to display the source excerpt)
"""

from src.retriever.pinecone_client import get_or_create_index

BATCH_SIZE = 100  # Pinecone recommends batching upserts


def sanitize_namespace(document_name: str) -> str:
    """
    Converts a document name into a safe namespace string
    (Pinecone namespaces are plain strings, but we normalize
    to avoid issues with spaces/special characters).
    """
    return document_name.replace(" ", "_").replace(".pdf", "").lower()


def upsert_chunks(embedded_chunks: list[dict], namespace: str = None) -> dict:
    """
    Upserts a list of embedded chunks into Pinecone.

    Each embedded chunk must have:
        chunk_id, document_name, page_number, text, embedding

    Returns a summary dict: {"namespace": ..., "upserted_count": ...}
    """
    if not embedded_chunks:
        raise ValueError("No embedded chunks provided to upsert_chunks().")

    index = get_or_create_index()

    doc_name = embedded_chunks[0]["document_name"]
    namespace = namespace or sanitize_namespace(doc_name)

    vectors = []
    for chunk in embedded_chunks:
        vectors.append({
            "id": chunk["chunk_id"],
            "values": chunk["embedding"],
            "metadata": {
                "document_name": chunk["document_name"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            },
        })

    total_upserted = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)

    return {"namespace": namespace, "upserted_count": total_upserted}


if __name__ == "__main__":
    # Quick manual test — run: python src/retriever/upsert.py path/to/file.pdf
    import sys
    sys.path.append(".")
    from src.loader.pdf_loader import load_pdf
    from src.chunker.chunker import chunk_pages
    from src.embedder.embedder import embed_chunks

    if len(sys.argv) < 2:
        print("Usage: python upsert.py <path_to_pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    embedded = embed_chunks(chunks)

    print(f"Upserting {len(embedded)} chunks into Pinecone...")
    result = upsert_chunks(embedded)
    print(f"Done. Namespace: {result['namespace']}, Upserted: {result['upserted_count']}")
