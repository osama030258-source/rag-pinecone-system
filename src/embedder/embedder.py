"""
Embedding Module
-----------------
Responsible for:
  - Loading a Sentence Transformer model
  - Converting chunk text into vector embeddings
  - Attaching the embedding vector to each chunk dict

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - Fast, lightweight, strong general-purpose semantic performance
  - Free / runs locally (no API cost) — good fit for this project
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

_model = None  # lazy-loaded singleton so we don't reload the model repeatedly


def get_model() -> SentenceTransformer:
    """Loads (once) and returns the Sentence Transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Takes a list of raw text strings and returns a list of embedding vectors.
    Batched for efficiency.
    """
    if not texts:
        raise ValueError("No texts provided to embed_texts().")

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Takes the output of chunker.chunk_pages() and returns the same list
    of chunk dicts, each with an added "embedding" key:

        {
            "chunk_id": "...",
            "document_name": "...",
            "page_number": 3,
            "text": "...",
            "chunk_size": 928,
            "embedding": [0.0123, -0.045, ...]   # 384 floats
        }
    """
    if not chunks:
        raise ValueError("No chunks provided to embed_chunks().")

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks


if __name__ == "__main__":
    # Quick manual test — run: python src/embedder/embedder.py path/to/file.pdf
    import sys
    sys.path.append(".")
    from src.loader.pdf_loader import load_pdf
    from src.chunker.chunker import chunk_pages

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <path_to_pdf>")
        sys.exit(1)

    pages = load_pdf(sys.argv[1])
    chunks = chunk_pages(pages)
    print(f"Chunks to embed: {len(chunks)}")
    print("Loading model (first run downloads it, may take a minute)...")

    embedded_chunks = embed_chunks(chunks)

    print("--- Preview of chunk 0 ---")
    print(f"Chunk ID: {embedded_chunks[0]['chunk_id']}")
    print(f"Embedding dimension: {len(embedded_chunks[0]['embedding'])}")
    print(f"First 5 values: {embedded_chunks[0]['embedding'][:5]}")
